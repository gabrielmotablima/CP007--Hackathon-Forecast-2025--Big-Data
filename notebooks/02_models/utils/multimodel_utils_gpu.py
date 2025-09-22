import numpy as np
import optuna
import pandas as pd
import time
import category_encoders as ce
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

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
        enc_map = {'target': ce.TargetEncoder, 'm_estimate': ce.MEstimateEncoder,
                   'loo': ce.LeaveOneOutEncoder, 'catboost': ce.CatBoostEncoder}
        return enc_map[self.encoder_type](cols=self.cols, **self.kwargs)

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("TargetEncoderWrapper requires y for fitting.")
        self._enc = self._build_encoder().fit(X, y)
        return self

    def transform(self, X):
        return self._enc.transform(X)

def create_preprocessing_pipeline(preprocessing_configs, numeric_columns, categorical_columns):
    pre_processing_steps = []
    for step, configs in preprocessing_configs.items():
        if step == 'CYCLICAL_FEATURES':
            cyclical_steps = [(f'cyclical_{feat}', FunctionTransformer(cyclical_feature, kw_args=args), [feat])
                              for feat, args in configs.items()]
            pre_processing_steps.append(('cyclical_features', ColumnTransformer(transformers=cyclical_steps, remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')))
        
        elif step == 'STANDARDIZE' and configs:
            pre_processing_steps.append(('scaler', ColumnTransformer(transformers=[('scaler', StandardScaler(), numeric_columns)], remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')))
        
        elif step == 'TARGET_ENCODE':
            te_cfg = configs
            fill_missing = make_column_transformer((SimpleImputer(strategy='constant', fill_value='UNK'), categorical_columns), remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')
            cast_and_clean = make_column_transformer((FunctionTransformer(sanitize_categories), categorical_columns), remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')
            target_encoder = TargetEncoderWrapper(encoder_type=te_cfg.get('encoder', 'catboost'), cols=categorical_columns, **te_cfg.get('params', {}))
            pre_processing_steps.append(('target_encoding', Pipeline(steps=[('impute', fill_missing), ('clean', cast_and_clean), ('encode', target_encoder)])))
            
    return Pipeline(steps=pre_processing_steps)

def _define_model_and_params(trial, hyperparam_ranges):
    model_name = trial.suggest_categorical('model', list(hyperparam_ranges.keys()))
    model_params = {}

    params = hyperparam_ranges[model_name]

    if model_name == 'XGBOOST':
        # Correctly parse [low, high, log] lists for XGBoost
        model_params = {
            'n_estimators': trial.suggest_int('n_estimators', params['n_estimators'][0], params['n_estimators'][1], log=params['n_estimators'][2]),
            'max_depth': trial.suggest_int('max_depth', params['max_depth'][0], params['max_depth'][1], log=params['max_depth'][2]),
            'learning_rate': trial.suggest_float('learning_rate', params['learning_rate'][0], params['learning_rate'][1], log=params['learning_rate'][2]),
            'min_child_weight': trial.suggest_float('min_child_weight', params['min_child_weight'][0], params['min_child_weight'][1], log=params['min_child_weight'][2]),
            'reg_alpha': trial.suggest_float('reg_alpha', params['reg_alpha'][0], params['reg_alpha'][1], log=params['reg_alpha'][2]),
            'reg_lambda': trial.suggest_float('reg_lambda', params['reg_lambda'][0], params['reg_lambda'][1], log=params['reg_lambda'][2]),
            'subsample': trial.suggest_float('subsample', params['subsample'][0], params['subsample'][1], log=params['subsample'][2]),
            'colsample_bytree': trial.suggest_float('colsample_bytree', params['colsample_bytree'][0], params['colsample_bytree'][1], log=params['colsample_bytree'][2]),
        }
        model = xgb.XGBRegressor(**model_params, device='cuda', tree_method='hist', random_state=42)

    elif model_name == 'RANDOM_FOREST':
        # Correctly parse [low, high, log] lists for RandomForest
        model_params = {
            'n_estimators': trial.suggest_int('n_estimators', params['n_estimators'][0], params['n_estimators'][1], log=params['n_estimators'][2]),
            'max_depth': trial.suggest_int('max_depth', params['max_depth'][0], params['max_depth'][1], log=params['max_depth'][2]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', params['min_samples_leaf'][0], params['min_samples_leaf'][1], log=params['min_samples_leaf'][2]),
            'min_samples_split': trial.suggest_int('min_samples_split', params['min_samples_split'][0], params['min_samples_split'][1], log=params['min_samples_split'][2]),
        }
        model = RandomForestRegressor(**model_params, random_state=42, n_jobs=-1)

    elif model_name == 'SVR':
        # Correctly parse parameters for SVR
        model_params = {
            'C': trial.suggest_float('C', params['C'][0], params['C'][1], log=params['C'][2]),
            'gamma': trial.suggest_float('gamma', params['gamma'][0], params['gamma'][1], log=params['gamma'][2]),
            'kernel': trial.suggest_categorical('kernel', params['kernel']),
        }
        model = SVR(**model_params)

    elif model_name == 'KNEIGHBORS':
        # Correctly parse parameters for KNeighbors
        model_params = {
            'n_neighbors': trial.suggest_int('n_neighbors', params['n_neighbors'][0], params['n_neighbors'][1], log=params['n_neighbors'][2]),
            'weights': trial.suggest_categorical('weights', params['weights']),
            'p': trial.suggest_int('p', params['p'][0], params['p'][1]), # p is not logarithmic
        }
        model = KNeighborsRegressor(**model_params, n_jobs=-1)
        
    return model_name, model, model_params


def objective(trial, hyperparam_ranges, tscv, x_data, columns_to_use, preprocessing, random_state, multi_scores=False):
    feature_columns, label_columns, numeric_columns, categorical_columns = columns_to_use
    X_train, y_train = x_data[0][feature_columns], x_data[1][label_columns]

    model_name, model, _ = _define_model_and_params(trial, hyperparam_ranges)
    
    scores_validation, scores_overfitting = [], []

    for idx, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipe_preproc = create_preprocessing_pipeline(preprocessing, numeric_columns, categorical_columns)
        
        ttr = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
        pipeline = Pipeline(steps=[('preprocessing', pipe_preproc), ('model', ttr)])

        start_time = time.time()
        pipeline.fit(X_train_fold, y_train_fold.values.ravel())
        print(f"[{model_name} - Fold {idx}] Fit finished in {time.time() - start_time:.2f}s")
        
        y_pred_train = pipeline.predict(X_train_fold)
        y_pred_val = pipeline.predict(X_val_fold)
        
        score_wmape_train = wmape_score(y_train_fold, y_pred_train)
        score_wmape_val = wmape_score(y_val_fold, y_pred_val)
        
        scores_validation.append(score_wmape_val)
        scores_overfitting.append(score_wmape_val - score_wmape_train)

        trial.report(score_wmape_val, idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    if multi_scores:
        return np.mean(scores_validation), np.mean(scores_overfitting)
    return np.mean(scores_validation)

def create_objective_function(hyperparam_ranges, tscv, x_data, columns_to_use, preprocessing, random_state, multi_scores=False):
    return lambda trial: objective(trial, hyperparam_ranges, tscv, x_data, columns_to_use, preprocessing, random_state, multi_scores)

