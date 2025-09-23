import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import KNNImputer, SimpleImputer
import optuna
import category_encoders as ce

def wmape_score(y_true, y_pred, epsilon=1e-8):
    '''
    Function to calculate the Weighted Mean Absolute Percentage Error (WMAPE) between true and predicted values.
    Parameters:
    - y_true: array-like of shape (n_samples,), true values.
    - y_pred: array-like of shape (n_samples,), predicted values.
    Returns:
    - float, the WMAPE score.
    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        return np.inf  # Or handle as appropriate for your use case

    return (numerator / denominator) * 100 # Often expressed as a percentage


def cyclical_feature(x, target_column, max_value):
    '''
    Function to transform a feature into its cyclical representation using sine and cosine transformations.
    Parameters:
    - x: pd.DataFrame, the input dataframe containing the target column.
    - target_column: str, the name of the column to be transformed.
    - max_value: int or float, the maximum value of the target column to scale the transformations.
    Returns:
    - pd.DataFrame, the dataframe with the original target column replaced by its sine and cosine transformations.
    '''
    x_transformed = x.copy()
    x_transformed[target_column + '_SIN'] = np.sin(2 * np.pi * x_transformed[target_column] / max_value)
    x_transformed[target_column + '_COS'] = np.cos(2 * np.pi * x_transformed[target_column] / max_value)
    x_transformed = x_transformed.drop(columns=target_column)
    return x_transformed


def get_best_params(study_best_trials):

    trials = study_best_trials.best_trials

    # Calculate the euclidian distance between best trials and the point (0,0)
    dists = np.array([np.linalg.norm(np.asarray(t.values, dtype=float)) for t in trials], dtype=float)

    # Selecting the best
    i_best = int(np.argmin(dists))

    # Return the index and the best parameters
    return i_best, trials[i_best].params


def sanitize_categories(df):
    # df pode vir como DataFrame ou array; garantimos DataFrame
    df = pd.DataFrame(df).copy()
    subs = {
    "<": "less than",
    ">": "greater than",
    "[": "",
    "]": "",
    ",": "_",
    " ": "_"
    }

    df = df.map(lambda x: str(x).translate(str.maketrans(subs)) if pd.notna(x) else x)
    return df


class TargetEncoderWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper para category_encoders com integração limpa em Pipelines do sklearn.
    encoder_type: 'target' | 'm_estimate' | 'loo' | 'catboost'
    cols: lista de colunas categóricas
    **kwargs: params do encoder escolhido (ex.: smoothing, m, sigma, random_state, etc.)
    """
    def __init__(self, encoder_type='catboost', cols=None, **kwargs):
        self.encoder_type = encoder_type
        self.cols = cols
        self.kwargs = kwargs
        self._enc = None

    def _build_encoder(self):
        enc_map = {
            'target': ce.TargetEncoder,
            'm_estimate': ce.MEstimateEncoder,
            'loo': ce.LeaveOneOutEncoder,
            'catboost': ce.CatBoostEncoder,
        }
        Enc = enc_map[self.encoder_type]
        return Enc(cols=self.cols, **self.kwargs)

    def fit(self, X, y=None):
        # category_encoders precisa de y
        if y is None:
            raise ValueError("TargetEncoderWrapper requer y no fit para calcular as estatísticas do target.")
        self._enc = self._build_encoder()
        self._enc = self._enc.fit(X, y)
        return self

    def transform(self, X):
        return self._enc.transform(X)
    

def create_preprocessing_pipeline(preprocessing_configs, numeric_columns, categorical_columns):
    '''
    Function to create a preprocessing pipeline based on the provided configuration dictionary.
    Parameters:
    - preprocessing_configs: dict, configuration dictionary specifying the preprocessing steps and their parameters.
    - numeric_columns: list of str, names of the numeric columns to be processed.
    - categorical_columns: list of str, names of the categorical columns to be processed.
    Returns:
    - sklearn.pipeline.Pipeline, the constructed preprocessing pipeline.
    '''
    # creating a dictionary to map the steps that should be implemented sequentially
    pre_processing_steps = []
    # iterate preprocessing steps and add them to the dictionary if they are not None
    for step in preprocessing_configs:
            # imputation of missing values in numerical columns
        if step == "KNN_IMPUTER":
            # creating the KNNImputer steps
            knn_imputation_steps = []
            # interating over the imputations that will be applied in sequence, appending transformations sequentially
            for imputation in preprocessing_configs['KNN_IMPUTER']:
                knn_imputation_steps.append(
                    (imputation,
                        make_column_transformer(
                            (KNNImputer(), 
                            make_column_selector(pattern = '|'.join(preprocessing_configs['KNN_IMPUTER'][imputation]))
                            ),  verbose_feature_names_out = False, remainder = 'passthrough').set_output(transform='pandas')
                    )
                )
            pre_processing_steps.append(('knn_imputation', Pipeline(steps = knn_imputation_steps)))

        # creating cyclical features, such that the users should not worry about with engineering these
        # steps themselves
        if step == 'CYCLICAL_FEATURES':
            # creating an empty list to store each of the cyclical feature transformations that are going to be created by
            # the pipeline object dynamically
            cyclical_features_steps = []
            # iterating across the cyclical features to implement the function that will parse the current
            # representation of the feature into its cyclical counterpart
            for feature in preprocessing_configs['CYCLICAL_FEATURES']:
                cyclical_features_steps.append(
                    (f'cyclical_{feature.lower()}',
                    FunctionTransformer(cyclical_feature, kw_args = preprocessing_configs['CYCLICAL_FEATURES'][feature]), [feature])
                )
            # appending this preprocessing steps to the list of steps that will compose the pipeline
            pre_processing_steps.append(
                ('cyclical_features', ColumnTransformer(transformers = cyclical_features_steps,
                                                        remainder = 'passthrough', verbose_feature_names_out = False
                                                        ).set_output(transform='pandas'))
            )

        # scanling the numeric features if this was included in the configuration dictionary and set to True
        if step == 'STANDARDIZE' and preprocessing_configs['STANDARDIZE']:
            if numeric_columns is None:
                raise ValueError("Numeric columns must be provided for StandardScaler.")
            # appending this pre-processing steps to the list of steps that will compose the pipeline
            pre_processing_steps.append(
                ('scaler', ColumnTransformer(transformers = [('scaler', StandardScaler(), numeric_columns)],
                                                            remainder = 'passthrough', verbose_feature_names_out = False
                                                            ).set_output(transform='pandas'))
            )

        if step == "ONE_HOT":
                if categorical_columns is None:
                    raise ValueError("Categorical columns must be provided for OneHotEncoder.")
                # adding any information to a categorical column
                fill_missing = make_column_transformer((SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='UNK'), categorical_columns),
                                                        verbose_feature_names_out = False, remainder = 'passthrough').set_output(transform='pandas')
                # sanitizing the categories to avoid issues with special characters
                cast_and_clean = make_column_transformer(
                    (FunctionTransformer(sanitize_categories), categorical_columns),
                    verbose_feature_names_out=False, remainder='passthrough'
                ).set_output(transform='pandas')
                # one hot enconding categorical features will be implemented after filling in the missing values
                encode_categorical = make_column_transformer((OneHotEncoder(**preprocessing_configs['ONE_HOT']), categorical_columns),
                                                                verbose_feature_names_out = False, remainder = 'passthrough').set_output(transform='pandas')
                # appending this pre-processing steps to the list of steps that will compose the pipeline
                pre_processing_steps.append(('ohe', Pipeline(steps = [('impute', fill_missing), ('clean', cast_and_clean), ('encode', encode_categorical)])))

        # ====== NOVO BLOCO: TARGET ENCODE (substitui o ONE_HOT) ======
        if step == "TARGET_ENCODE":
            if categorical_columns is None:
                raise ValueError("Categorical columns must be provided for Target Encoding.")

            te_cfg = preprocessing_configs['TARGET_ENCODE']
            encoder_type = te_cfg.get('encoder', 'catboost')  # 'catboost' | 'target' | 'm_estimate' | 'loo'
            encoder_params = te_cfg.get('params', {})

            # 1) imputação simples de missing nas categóricas (ex.: UNK)
            fill_missing = make_column_transformer(
                (SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='UNK'), categorical_columns),
                verbose_feature_names_out=False, remainder='passthrough'
            ).set_output(transform='pandas')

            # 2) sanitização (opcional, mantém seu comportamento)
            cast_and_clean = make_column_transformer(
                (FunctionTransformer(sanitize_categories), categorical_columns),
                verbose_feature_names_out=False, remainder='passthrough'
            ).set_output(transform='pandas')

            # 3) target encoding seguro (o encoder atua apenas nas colunas categóricas)
            # O wrapper aplica o encoder apenas nas cols especificadas e preserva o resto.
            target_encoder = TargetEncoderWrapper(
                encoder_type=encoder_type,
                cols=categorical_columns,
                **encoder_params
            )

            pre_processing_steps.append((
                'target_encoding',
                Pipeline(steps=[
                    ('impute', fill_missing),
                    ('clean', cast_and_clean),
                    ('encode', target_encoder),
                ])
            ))

    # returning a pipeline object containing all the pre-processing steps
    return Pipeline(steps = pre_processing_steps)


def create_objective_function(hyperparam_ranges, tscv, x_data, regressor_model, columns_to_use, eval_features,
                              preprocessing, random_state, defining_hyperparams_function, multi_scores = False, pruner = False):
    # Creating objective function that will be used by Optuna to optimize the hyperparameters of the model
    return lambda trial: objective(trial, hyperparam_ranges, tscv, x_data, regressor_model, columns_to_use, eval_features,
                                   preprocessing, random_state, defining_hyperparams_function, multi_scores, pruner)


def objective(trial, hyperparam_ranges, tscv, x_data, regressor_model, columns_to_use, eval_features,
              preprocessing, random_state, defining_hyperparams_function, multi_scores = False, pruner = False):
    
    # separate the columns to use and the input data
    feature_columns, label_columns, numeric_columns, categorical_columns = columns_to_use
    X_train, y_train = x_data[0][feature_columns], x_data[1][label_columns]

    # hyperparameters that will be optimized by Optuna
    dict_params = defining_hyperparams_function(trial, hyperparam_ranges)

    # Defining scores validation for Optuna optimization
    # This evaluates the validation scores by params trials Optuna
    scores_validation = []
    # This evaluates the underfit or overfit of the model
    scores_overfitting = []

    for idx, fold in enumerate(tscv.split(X_train), start=1):
        ###########################################################################################
        #                 GETTING THE LOGS FROM THE TRAINING AND VALIDATION FOLDS                 #
        ###########################################################################################
        # getting indexes of the training and validation fold
        training_index, validation_index = fold
        # getting the logs from training and validation folds
        X_train_fold, X_val_fold = X_train.iloc[training_index, :], X_train.iloc[validation_index, :]
        y_train_fold, y_val_fold = y_train.iloc[training_index], y_train.iloc[validation_index]
        print(f"[Fold {idx}] Training fold: {X_train_fold.shape[0]} samples")
        
        ###########################################################################################
        #                               CREATING THE PIPELINE OBJECT                              #
        ###########################################################################################
        # Creating the pipeline object
        print(f"[Fold {idx}] Creating the preprocessing and modeling pipeline.")

        # ======================================================= #
        #                IMPUTATION PIPELINE STEPS                #
        # ======================================================= #

        # creating the pipeline object that will be used to pre-process the input data before fitting the model
        pipe_preproc = create_preprocessing_pipeline(preprocessing_configs = preprocessing, 
                                                        numeric_columns = numeric_columns, 
                                                        categorical_columns = categorical_columns) 
        
        ###########################################################################################
        #                              FITTING THE MODEL TO THE DATA                              #
        ###########################################################################################
        print(f"[Fold {idx}] Fitting the model to the data.")

        # creating a pipeline object to accomodate both: pre-processing steps and the model
        pipeline = Pipeline(steps=[
            ('preprocessing', pipe_preproc),
            ('model', regressor_model(**dict_params, random_state = random_state))
        ])

        # training the model
        pipeline.fit(X_train_fold, y_train_fold.values.ravel())
        
        ###########################################################################################
        #                            EVALUATING THE MODEL PERFORMANCE                             #
        ###########################################################################################
        # predicting the target values for the training and validation folds
        y_pred_train = pipeline.predict(X_train_fold)
        y_pred_val = pipeline.predict(X_val_fold)
        
        # calculating the WMAPE score for the training and validation folds
        score_wmape_train = wmape_score(y_train_fold, y_pred_train)
        score_wmape_val = wmape_score(y_val_fold, y_pred_val)
        print(f"[Fold {idx}] WMAPE Training:   {score_wmape_train}.")
        print(f"[Fold {idx}] WMAPE Validation: {score_wmape_val}.")
        # stores the diff between the training and validation scores
        scores_overfitting.append(score_wmape_val - score_wmape_train)
        # stores validation score
        scores_validation.append(score_wmape_val)

        if pruner:
            # report intermediate objective value
            trial.report(score_wmape_val, idx)
            # handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    # returning multi scores
    if multi_scores:
        return np.mean(scores_validation), np.mean(scores_overfitting)
    # returning single score
    return np.mean(scores_validation)


def min_max_param(params, min_param, max_param):
    if params < min_param:
        min_param = params
    if params > max_param:
        max_param = params
    return min_param, max_param

