# 🛒 Previsão de Vendas para Varejo - Hackathon Forecast 2025

## 📋 Visão Geral

Este projeto desenvolve um modelo de **previsão de vendas (forecast)** para apoiar o varejo na reposição de produtos. O objetivo é prever a **quantidade semanal de vendas por PDV (Ponto de Venda) e SKU (Stock Keeping Unit)** utilizando técnicas avançadas de machine learning.

### 🎯 Objetivo Principal
Desenvolver um modelo preditivo robusto que permita:
- Prever vendas semanais com alta precisão
- Otimizar a gestão de estoque
- Reduzir perdas por ruptura ou excesso de produtos
- Melhorar a eficiência operacional do varejo

### 🔧 Tecnologias Utilizadas
- **Python 3.x** - Linguagem principal
- **XGBoost** - Algoritmo de machine learning principal
- **Polars** - Manipulação eficiente de dados
- **Pandas** - Análise de dados
- **Scikit-learn** - Ferramentas de ML e métricas
- **Optuna** - Otimização de hiperparâmetros
- **Category Encoders** - Codificação de variáveis categóricas

## 📁 Estrutura do Projeto

```
CP007--Hackathon-Forecast-2025--Big-Data/
├── LICENSE                              # Licença do projeto
├── README.md                           # Documentação principal
├── requirements.txt                    # Dependências Python
├── code/                              # Scripts auxiliares (se existir)
├── data/                              # Diretório de dados (não versionado)
│   ├── raw/                           # Dados brutos
│   └── processed/                     # Dados processados
└── notebooks/                         # Jupyter Notebooks
    ├── 01_data_preparation/           # Preparação e limpeza dos dados
    │   ├── 01_prepare_data.ipynb      # Notebook principal de preparação
    │   └── 02_build_production.ipynb  # Pipeline de produção
    └── 02_models/                     # Modelagem e treinamento
        ├── 01_xgboost.ipynb           # Modelo XGBoost padrão
        ├── 02_xgboost_gpu.ipynb       # Modelo XGBoost com GPU
        ├── model_params.yml           # Configurações do modelo
        ├── utils.py                   # Funções utilitárias
        └── utils_gpu.py               # Funções para GPU
```

## 📊 Dados e Features

### Fontes de Dados
O projeto utiliza os seguintes datasets:
- **Transações**: Dados históricos de vendas por PDV/SKU
- **Lojas**: Informações dos pontos de venda
- **Produtos**: Catálogo de produtos e características
- **CEP**: Dados geográficos para contextualização
- **Feriados**: Calendário de feriados para sazonalidade

### Features Principais
- `internal_store_id`: ID único da loja/PDV
- `internal_product_id`: ID único do produto/SKU
- `quantity`: Quantidade vendida (variável target)
- `week_of_year`: Semana do ano (sazonalidade)
- `month`: Mês (sazonalidade)
- `holiday`: Indicador de feriado
- `previous_month_*`: Features de vendas do mês anterior
- Variables categóricas: categoria, marca, fabricante, etc.

## 🚀 Como Executar

### 1. Pré-requisitos
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Jupyter Notebook ou JupyterLab
- GPU NVIDIA (opcional, para versão GPU do XGBoost)

### 2. Instalação do Ambiente

```bash
# Clone o repositório
git clone https://github.com/gabrielmotablima/CP007--Hackathon-Forecast-2025--Big-Data.git
cd CP007--Hackathon-Forecast-2025--Big-Data

# Crie um ambiente virtual (recomendado)
python -m venv venv

# Ative o ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt

# Instale o XGBoost com suporte GPU (opcional)
pip install xgboost[gpu]
```

### 3. Preparação dos Dados

```bash
# Certifique-se que os dados estão no diretório correto
mkdir -p data/raw data/processed

# Execute o notebook de preparação
jupyter notebook notebooks/01_data_preparation/01_prepare_data.ipynb
```

### 4. Treinamento do Modelo

```bash
# Para modelo padrão (CPU)
jupyter notebook notebooks/02_models/01_xgboost.ipynb

# Para modelo com GPU (se disponível)
jupyter notebook notebooks/02_models/02_xgboost_gpu.ipynb
```

## 📈 Métricas e Avaliação

### Métrica Principal: WMAPE
O projeto utiliza **WMAPE (Weighted Mean Absolute Percentage Error)** como métrica principal:

```
WMAPE = (Σ|y_true - y_pred| / Σ|y_true|) × 100
```

### Estratégia de Validação
- **Validação Cruzada**: 4 folds estratificados
- **Estratificação**: Por `internal_store_id` e `internal_product_id`
- **Otimização**: Hiperparâmetros via Optuna

## ⚙️ Configurações

As configurações do modelo estão centralizadas no arquivo `model_params.yml`:

```yaml
RANDOM_STATE: 33              # Semente para reprodutibilidade
NUMBER_OF_FOLDS: 4           # Número de folds na validação cruzada
TARGET: quantity             # Variável target
EVAL_FEATURES: [...]         # Features para avaliação
FEATURES: [...]              # Features do modelo
```

## 🔄 Pipeline de Produção

1. **Preparação dos Dados** (`01_prepare_data.ipynb`)
   - Carregamento e unificação dos datasets
   - Limpeza e tratamento de valores ausentes
   - Engenharia de features
   - Features cíclicas (mês, semana)

2. **Modelagem** (`01_xgboost.ipynb` ou `02_xgboost_gpu.ipynb`)
   - Divisão treino/validação estratificada
   - Otimização de hiperparâmetros
   - Treinamento do modelo final
   - Avaliação e métricas

3. **Produção** (`02_build_production.ipynb`)
   - Pipeline automatizado
   - Exportação do modelo treinado
   - Scripts de inferência

## 🤝 Contribuição

Para contribuir com o projeto:

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença especificada no arquivo `LICENSE`.

## 📞 Contato

Para dúvidas ou sugestões sobre o projeto, entre em contato através do GitHub.

---

**Desenvolvido para o Hackathon Forecast 2025 - Big Data** 🏆