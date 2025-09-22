import numpy as np
import optuna
import pandas as pd
import time
import category_encoders as ce
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

def wmape_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true))
    return (numerator / denominator) * 100 if denominator != 0 else np.inf

def cyclical_feature(x, target_column, max_value):
    x_transformed = x.copy()
    x_transformed[target_column + '_SIN'] = np.sin(2 * np.pi * x_transformed[target_column] / max_value)
    x_transformed[target_column + '_COS'] = np.cos(2 * np.pi * x_transformed[target_column] / max_value)
    x_transformed = x_transformed.drop(columns=target_column)
    return x_transformed

def get_best_params(study_best_trials):
    trials = study_best_trials.best_trials
    dists = np.array([np.linalg.norm(np.asarray(t.values, dtype=float)) for t in trials], dtype=float)
    i_best = int(np.argmin(dists))
    return i_best, trials[i_best].params

def sanitize_categories(df):
    df = pd.DataFrame(df).copy()
    subs = {"<": "lt", ">": "gt", "[": "", "]": "", ",": "_", " ": "_"}
    return df.map(lambda x: str(x).translate(str.maketrans(subs)) if pd.notna(x) else x)

class TargetEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, encoder_type='catboost', cols=None, **kwargs):
        self.encoder_type = encoder_type
        self.cols = cols
        self.kwargs = kwargs
        self._enc = None

    def _build_encoder(self):
        enc_map = {'target': ce.TargetEncoder, 'catboost': ce.CatBoostEncoder}
        return enc_map[self.encoder_type](cols=self.cols, **self.kwargs)

    def fit(self, X, y=None):
        if y is None: raise ValueError("TargetEncoderWrapper requires y for fitting.")
        self._enc = self._build_encoder().fit(X, y)
        return self

    def transform(self, X):
        return self._enc.transform(X)

def create_preprocessing_pipeline(preprocessing_configs, numeric_columns, categorical_columns):
    steps = []
    for step, configs in preprocessing_configs.items():
        if step == 'CYCLICAL_FEATURES':
            cyclo_steps = [(f'cyclo_{feat}', FunctionTransformer(cyclical_feature, kw_args=args), [feat]) for feat, args in configs.items()]
            steps.append(('cyclical', ColumnTransformer(cyclo_steps, remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')))
        elif step == 'STANDARDIZE' and configs:
            steps.append(('scaler', ColumnTransformer([('scaler', StandardScaler(), numeric_columns)], remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')))
        elif step == 'TARGET_ENCODE':
            # FIX: Correctly parse the encoder and its parameters
            encoder_type = configs.get('encoder', 'catboost')
            encoder_params = configs.get('params', {})
            
            pipe = Pipeline([
                ('impute', make_column_transformer((SimpleImputer(strategy='constant', fill_value='UNK'), categorical_columns), remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')),
                ('clean', make_column_transformer((FunctionTransformer(sanitize_categories), categorical_columns), remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')),
                ('encode', TargetEncoderWrapper(encoder_type=encoder_type, cols=categorical_columns, **encoder_params))
            ])
            steps.append(('target_encode', pipe))
    return Pipeline(steps)

def objective(trial, hyperparam_ranges, tscv, x_data, columns_to_use, preprocessing, random_state):
    feature_columns, label_columns, numeric_columns, categorical_columns = columns_to_use
    X_train, y_train = x_data[0][feature_columns], x_data[1][label_columns]

    def suggest_param(trial, name, params):
        param_type = 'float' if isinstance(params[0], float) else 'int'
        if param_type == 'float':
            return trial.suggest_float(name, params[0], params[1], log=params[2])
        else:
            return trial.suggest_int(name, params[0], params[1], log=params[2])

    xgb_params = {k: suggest_param(trial, f'xgb_{k}', v) for k, v in hyperparam_ranges['XGBOOST'].items()}
    rf_params = {k: suggest_param(trial, f'rf_{k}', v) for k, v in hyperparam_ranges['RANDOM_FOREST'].items()}
    svr_params = {k: suggest_param(trial, f'svr_{k}', v) for k, v in hyperparam_ranges['SVR'].items()}

    w1 = trial.suggest_float('w_xgb', 0.1, 1.0)
    w2 = trial.suggest_float('w_rf', 0.1, 1.0)
    w3 = trial.suggest_float('w_svr', 0.1, 1.0)

    xgb_model = XGBRegressor(**xgb_params, device='cuda', tree_method='hist', random_state=random_state)
    rf_model = RandomForestRegressor(**rf_params, random_state=random_state, n_jobs=-1)
    svr_model = SVR(**svr_params)
    
    voting_regressor = VotingRegressor(
        estimators=[('xgb', xgb_model), ('rf', rf_model), ('svr', svr_model)],
        weights=[w1, w2, w3],
        n_jobs=-1
    )

    scores_validation, scores_overfitting = [], []
    
    for idx, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipe_preproc = create_preprocessing_pipeline(preprocessing, numeric_columns, categorical_columns)
        
        ttr = TransformedTargetRegressor(regressor=voting_regressor, func=np.log1p, inverse_func=np.expm1)
        pipeline = Pipeline(steps=[('preprocessing', pipe_preproc), ('model', ttr)])

        start_time = time.time()
        pipeline.fit(X_train_fold, y_train_fold.values.ravel())
        print(f"[Fold {idx}] Fit finished in {time.time() - start_time:.2f}s")
        
        y_pred_train = pipeline.predict(X_train_fold)
        y_pred_val = pipeline.predict(X_val_fold)
        
        score_wmape_train = wmape_score(y_train_fold.values, y_pred_train)
        score_wmape_val = wmape_score(y_val_fold.values, y_pred_val)
        
        scores_validation.append(score_wmape_val)
        scores_overfitting.append(score_wmape_val - score_wmape_train)

        trial.report(score_wmape_val, idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores_validation), np.mean(scores_overfitting)

def create_objective_function(hyperparam_ranges, tscv, x_data, columns_to_use, preprocessing, random_state):
    return lambda trial: objective(trial, hyperparam_ranges, tscv, x_data, columns_to_use, preprocessing, random_state)

