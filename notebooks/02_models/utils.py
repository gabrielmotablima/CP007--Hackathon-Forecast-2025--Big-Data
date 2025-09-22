"""
Utilities for Sales Forecasting Model - XGBoost Implementation

Este módulo contém funções utilitárias e classes customizadas para o projeto
de previsão de vendas, incluindo métricas de avaliação, transformações de 
features e encoders categóricos.

Autor: Equipe FGY - Hackathon Forecast 2025
Data: 2025
"""

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
    """
    Calcula o WMAPE (Weighted Mean Absolute Percentage Error) entre valores reais e preditos.
    
    O WMAPE é uma métrica mais robusta que o MAPE tradicional, especialmente para
    séries temporais com valores próximos de zero, pois pondera os erros pelo volume total.
    
    Fórmula: WMAPE = (Σ|y_true - y_pred| / Σ|y_true|) × 100
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Valores reais/observados
    y_pred : array-like of shape (n_samples,)
        Valores preditos pelo modelo
    epsilon : float, default=1e-8
        Pequeno valor para evitar divisão por zero
        
    Returns
    -------
    float
        Pontuação WMAPE em percentual (0-100)
        
    Examples
    --------
    >>> y_true = [100, 200, 300]
    >>> y_pred = [90, 210, 290]
    >>> wmape_score(y_true, y_pred)
    5.0
    
    Notes
    -----
    - Valores menores indicam melhor performance
    - Retorna inf se denominador for zero
    - Amplamente usado para avaliação de modelos de forecast de vendas
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        return np.inf  # Or handle as appropriate for your use case

    return (numerator / denominator) * 100 # Often expressed as a percentage


def cyclical_feature(x, target_column, max_value):
    """
    Transforma uma feature em sua representação cíclica usando transformações 
    seno e cosseno para capturar a natureza cíclica de variáveis temporais.
    
    Esta transformação é especialmente útil para features como mês, dia da semana,
    hora do dia, etc., onde existe uma continuidade entre o valor máximo e mínimo.
    
    Parameters
    ----------
    x : pd.DataFrame
        DataFrame de entrada contendo a coluna target
    target_column : str
        Nome da coluna a ser transformada ciclicamente
    max_value : int or float
        Valor máximo da feature para normalizar as transformações
        (ex: 12 para meses, 7 para dias da semana, 24 para horas)
        
    Returns
    -------
    pd.DataFrame
        DataFrame com a coluna original substituída por suas 
        representações seno e cosseno (_SIN e _COS)
        
    Examples
    --------
    >>> df = pd.DataFrame({'month': [1, 6, 12]})
    >>> result = cyclical_feature(df, 'month', 12)
    >>> print(result.columns)
    Index(['month_SIN', 'month_COS'], dtype='object')
    
    Notes
    -----
    - Remove a coluna original após a transformação
    - Preserva a continuidade cíclica (dezembro conecta com janeiro)
    - Facilita o aprendizado de padrões sazonais pelos modelos ML
    """
    x_transformed = x.copy()
    x_transformed[target_column + '_SIN'] = np.sin(2 * np.pi * x_transformed[target_column] / max_value)
    x_transformed[target_column + '_COS'] = np.cos(2 * np.pi * x_transformed[target_column] / max_value)
    x_transformed = x_transformed.drop(columns=target_column)
    return x_transformed


def get_best_params(study_best_trials):
    """
    Seleciona os melhores parâmetros de um estudo Optuna multi-objetivo
    baseado na menor distância euclidiana ao ponto de origem (0,0).
    
    Em otimização multi-objetivo, pode haver múltiplas soluções Pareto-ótimas.
    Esta função usa distância euclidiana para selecionar uma solução única
    que equilibra todos os objetivos.
    
    Parameters
    ----------
    study_best_trials : optuna.study.Study
        Objeto de estudo Optuna com trials completados
        
    Returns
    -------
    tuple
        (índice_melhor_trial, parâmetros_melhor_trial)
        
    Examples
    --------
    >>> study = optuna.create_study(directions=['minimize', 'minimize'])
    >>> # ... executar trials ...
    >>> idx, best_params = get_best_params(study)
    >>> print(f"Melhor trial: {idx}, Parâmetros: {best_params}")
    
    Notes
    -----
    - Útil quando há trade-offs entre múltiplos objetivos
    - Assume que todos os objetivos devem ser minimizados
    - A distância euclidiana favoriza soluções balanceadas
    """

    trials = study_best_trials.best_trials

    # Calculate the euclidian distance between best trials and the point (0,0)
    dists = np.array([np.linalg.norm(np.asarray(t.values, dtype=float)) for t in trials], dtype=float)

    # Selecting the best
    i_best = int(np.argmin(dists))

    # Return the index and the best parameters
    return i_best, trials[i_best].params


def sanitize_categories(df):
    """
    Sanitiza valores categóricos removendo caracteres especiais que podem
    causar problemas em algoritmos de machine learning.
    
    Esta função é especialmente útil para limpar categorias que contêm
    caracteres como <, >, [, ], vírgulas e espaços, que podem interferir
    no processamento por alguns algoritmos ou bibliotecas.
    
    Parameters
    ----------
    df : pd.DataFrame or array-like
        DataFrame ou array contendo valores categóricos para sanitizar
        
    Returns
    -------
    pd.DataFrame
        DataFrame com valores categóricos sanitizados
        
    Examples
    --------
    >>> df = pd.DataFrame({'cat': ['A,B', 'C<D', '[E]']})
    >>> sanitized = sanitize_categories(df)
    >>> print(sanitized['cat'].tolist())
    ['A_B', 'C less than D', 'E']
    
    Notes
    -----
    - Converte todos os valores para string antes da sanitização
    - Preserva valores NaN/None
    - Útil antes de aplicar encoders categóricos
    """
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
    
    Esta classe encapsula diferentes tipos de target encoders do category_encoders,
    permitindo uso transparente em Pipelines do scikit-learn com validação cruzada
    e evitando vazamento de dados (data leakage).
    
    Parameters
    ----------
    encoder_type : str, default='catboost'
        Tipo de encoder a ser usado:
        - 'target': TargetEncoder básico
        - 'm_estimate': MEstimateEncoder com regularização
        - 'loo': LeaveOneOutEncoder 
        - 'catboost': CatBoostEncoder (mais robusto)
    cols : list or None, default=None
        Lista de colunas categóricas a serem codificadas.
        Se None, aplica a todas as colunas
    **kwargs : dict
        Parâmetros específicos do encoder escolhido
        (ex: smoothing, m, sigma, random_state, etc.)
        
    Attributes
    ----------
    _enc : category_encoders.BaseEncoder
        Instância do encoder configurado
        
    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> encoder = TargetEncoderWrapper(encoder_type='catboost', cols=['categoria'])
    >>> pipeline = Pipeline([('encoder', encoder), ('model', XGBRegressor())])
    >>> pipeline.fit(X_train, y_train)
    
    Notes
    -----
    - Compatível com Pipeline e cross_val_score do sklearn
    - Evita overfitting usando técnicas internas de cada encoder
    - CatBoost encoder é geralmente mais robusto para production
    """
    def __init__(self, encoder_type='catboost', cols=None, **kwargs):
        self.encoder_type = encoder_type
        self.cols = cols
        self.kwargs = kwargs
        self._enc = None

    def _build_encoder(self):
        """Constrói a instância do encoder baseado no tipo especificado."""
        enc_map = {
            'target': ce.TargetEncoder,
            'm_estimate': ce.MEstimateEncoder,
            'loo': ce.LeaveOneOutEncoder,
            'catboost': ce.CatBoostEncoder,
        }
        Enc = enc_map[self.encoder_type]
        return Enc(cols=self.cols, **self.kwargs)

    def fit(self, X, y=None):
        """
        Treina o encoder nas features categóricas usando a variável target.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features de entrada
        y : array-like
            Variável target (obrigatória para target encoding)
            
        Returns
        -------
        self : TargetEncoderWrapper
            Retorna self para permitir method chaining
        """
        if y is None:
            raise ValueError("TargetEncoderWrapper requer y no fit para calcular as estatísticas do target.")
        self._enc = self._build_encoder()
        self._enc = self._enc.fit(X, y)
        return self

    def transform(self, X):
        """
        Aplica a transformação de encoding nas features categóricas.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features de entrada para transformar
            
        Returns
        -------
        pd.DataFrame
            Features transformadas com encoding aplicado
        """
        return self._enc.transform(X)
    

def create_preprocessing_pipeline(preprocessing_configs, numeric_columns, categorical_columns):
    """
    Cria um pipeline de pré-processamento baseado na configuração fornecida.
    
    Esta função constrói dinamicamente um pipeline do scikit-learn com diferentes
    etapas de pré-processamento baseadas em um dicionário de configuração,
    permitindo flexibilidade na definição de transformações.
    
    Parameters
    ----------
    preprocessing_configs : dict
        Dicionário de configuração especificando as etapas de pré-processamento
        e seus parâmetros. Estrutura esperada:
        {
            'KNN_IMPUTER': {...},
            'SCALER': {...},
            'CATEGORICAL_ENCODER': {...}
        }
    numeric_columns : list of str
        Nomes das colunas numéricas a serem processadas
    categorical_columns : list of str
        Nomes das colunas categóricas a serem processadas
        
    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline de pré-processamento construído com as transformações especificadas
        
    Examples
    --------
    >>> config = {
    ...     'KNN_IMPUTER': {'numeric': ['col1', 'col2']},
    ...     'SCALER': {'standard': ['col1', 'col2']}
    ... }
    >>> pipeline = create_preprocessing_pipeline(config, ['col1', 'col2'], ['cat1'])
    
    Notes
    -----
    - Suporta diferentes tipos de imputação (KNN, Simple)
    - Múltiplos tipos de scaling (Standard, MinMax, Robust)
    - Diversos encoders categóricos (OneHot, Target, etc.)
    - Etapas são aplicadas sequencialmente conforme configuração
    """
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

