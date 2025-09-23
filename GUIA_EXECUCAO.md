# 🚀 Guia de Execução Completo - Previsão de Vendas

Este guia fornece instruções detalhadas para executar todo o pipeline de previsão de vendas, desde a configuração inicial até a obtenção dos resultados finais.

## 📋 Pré-requisitos

### Sistema Operacional
- Windows 10/11, macOS 10.15+, ou Ubuntu 18.04+
- Pelo menos 8GB de RAM (16GB recomendado)
- 10GB de espaço livre em disco

### Software Necessário
- **Python 3.8+** (recomendado: 3.9 ou 3.10)
- **Git** para controle de versão
- **Jupyter Notebook** ou **VS Code com extensão Python**

### Hardware Opcional
- **GPU NVIDIA** com drivers CUDA para aceleração (opcional)

## 🛠️ Configuração do Ambiente

### Passo 1: Clonar o Repositório

```bash
# Clone o repositório
git clone https://github.com/gabrielmotablima/CP007--Hackathon-Forecast-2025--Big-Data.git

# Entre no diretório
cd CP007--Hackathon-Forecast-2025--Big-Data
```

### Passo 2: Criar Ambiente Virtual

```bash
# Criar ambiente virtual
python -m venv forecast_env

# Ativar ambiente (Windows)
forecast_env\Scripts\activate

# Ativar ambiente (Linux/Mac)
source forecast_env/bin/activate
```

### Passo 3: Instalar Dependências

```bash
# Atualizar pip
python -m pip install --upgrade pip

# Instalar dependências principais
pip install -r requirements.txt

# Para suporte GPU (opcional)
pip install xgboost[gpu]
```

### Passo 4: Verificar Instalação

```bash
# Verificar Python e bibliotecas
python -c "import pandas, numpy, xgboost, sklearn, optuna; print('✅ Todas as bibliotecas instaladas com sucesso!')"
```

## 📂 Preparação dos Dados

### Passo 1: Estrutura de Diretórios

Certifique-se de que a estrutura esteja organizada:

```
CP007--Hackathon-Forecast-2025--Big-Data/
├── data/
│   ├── raw/                    # Dados brutos (você deve colocar aqui)
│   │   ├── part-00000-tid-2779033056155408584-*.parquet  # Dados de lojas
│   │   ├── part-00000-tid-5196563791502273604-*.parquet  # Dados de transações
│   │   ├── part-00000-tid-7173294866425216458-*.parquet  # Dados de produtos
│   │   ├── georef-zipcode.csv                            # Dados de CEP
│   │   └── processed_usa_holiday.csv                     # Feriados
│   └── processed/              # Dados processados (será criado)
└── notebooks/                  # Notebooks de execução
```

### Passo 2: Colocar os Dados

1. **Obtenha os arquivos de dados** (conforme instruções do hackathon)
2. **Coloque todos os arquivos** no diretório `data/raw/`
3. **Verifique os nomes** dos arquivos conforme esperado pelos notebooks

## 🔄 Execução do Pipeline

### Etapa 1: Preparação dos Dados (OBRIGATÓRIA)

```bash
# Iniciar Jupyter Notebook
jupyter notebook

# Ou usar VS Code
code notebooks/01_data_preparation/01_prepare_data.ipynb
```

**Executar o notebook completo:**
1. Abra `notebooks/01_data_preparation/01_prepare_data.ipynb`
2. Execute todas as células sequencialmente (`Kernel → Restart & Run All`)
3. Aguarde conclusão (pode levar 10-30 minutos dependendo do tamanho dos dados)
4. Verifique se foi criado: `data/processed/processed_data.parquet`

**Indicadores de sucesso:**
- ✅ Arquivo `processed_data.parquet` criado
- ✅ Sem erros nas células de execução
- ✅ Mensagens de log indicando conclusão

### Etapa 2: Treinamento do Modelo (PRINCIPAL)

```bash
# Abrir notebook de modelagem
code notebooks/02_models/01_xgboost.ipynb
```

**Executar passo a passo:**

1. **Configuração e Carregamento (Células 1-5)**
   ```python
   # Verifique se carregou corretamente
   print(f"Dataset shape: {df.shape}")
   print(f"Colunas: {df.columns.tolist()}")
   ```

2. **Divisão dos Dados (Células 6-8)**
   ```python
   # Verifique as divisões
   print(f"Treino: {X_train.shape}")
   print(f"Teste: {X_test.shape}")
   ```

3. **Otimização de Hiperparâmetros (Células 9-12)**
   - ⚠️ **IMPORTANTE**: Esta etapa pode levar 1-3 horas
   - Monitore o progresso através dos logs do Optuna
   - Você pode ajustar `n_trials` para reduzir tempo

4. **Treinamento Final (Células 13-15)**
   ```python
   # Verifique métricas de treino
   print(f"WMAPE Validação: {wmape_cv}")
   print(f"WMAPE Teste: {wmape_test}")
   ```

5. **Avaliação e Exportação (Células 16-18)**
   - Gere gráficos de performance
   - Exporte modelo treinado
   - Calcule feature importance

### Etapa 3: Versão GPU (OPCIONAL)

Se você tem GPU NVIDIA disponível:

```bash
# Abrir versão GPU
code notebooks/02_models/02_xgboost_gpu.ipynb
```

**Vantagens da versão GPU:**
- 🚀 Treinamento 5-10x mais rápido
- ⚡ Otimização de hiperparâmetros acelerada
- 💻 Melhor uso de recursos de hardware

## 📊 Interpretação dos Resultados

### Métricas Principais

**WMAPE (Weighted Mean Absolute Percentage Error)**
- `< 10%`: Excelente performance
- `10-20%`: Boa performance 
- `20-30%`: Performance aceitável
- `> 30%`: Necessita melhorias

### Arquivos de Saída

Após execução completa, você terá:

```
data/processed/
├── processed_data.parquet          # Dados limpos
├── model_xgboost.pkl              # Modelo treinado
├── best_params.json               # Melhores hiperparâmetros
└── feature_importance.csv         # Importância das features

results/
├── validation_results.csv         # Resultados validação cruzada
├── test_predictions.csv           # Predições no conjunto teste
└── performance_plots.png          # Gráficos de performance
```

## 🔧 Solução de Problemas

### Erros Comuns

**1. ModuleNotFoundError**
```bash
# Solução: Reinstalar dependências
pip install -r requirements.txt --force-reinstall
```

**2. Arquivo não encontrado**
```bash
# Solução: Verificar estrutura de diretórios
ls data/raw/  # Deve mostrar todos os arquivos .parquet e .csv
```

**3. Memória insuficiente**
```python
# Solução: Reduzir tamanho do dataset temporariamente
df_sample = df.sample(frac=0.1)  # Usar 10% dos dados
```

**4. GPU não reconhecida**
```bash
# Verificar instalação CUDA
python -c "import xgboost as xgb; print(xgb.XGBRegressor(tree_method='gpu_hist'))"
```

### Otimizações de Performance

**1. Reduzir tempo de execução:**
```yaml
# Ajustar em model_params.yml
OPTUNA_PARAMS:
  n_trials: 50  # Reduzir de 100 para 50
```

**2. Usar menos folds:**
```yaml
NUMBER_OF_FOLDS: 3  # Reduzir de 4 para 3
```

**3. Limitar features:**
```yaml
FEATURES: ['internal_store_id', 'internal_product_id', 'month', 'week_of_year']
```

## 📈 Próximos Passos

### Após Execução Bem-sucedida

1. **Análise de Resultados**
   - Revisar métricas de performance
   - Analisar feature importance
   - Identificar padrões nos erros

2. **Validação de Negócio**
   - Comparar com dados históricos
   - Validar com especialistas de domínio
   - Testar em cenários específicos

3. **Deployment (Opcional)**
   - Criar API de predição
   - Implementar monitoramento
   - Automatizar retreinamento

### Melhorias Possíveis

1. **Feature Engineering Avançada**
   - Features de lag mais complexas
   - Interações entre variáveis
   - Features de sazonalidade específicas

2. **Modelos Ensemble**
   - Combinar XGBoost com outros algoritmos
   - Stacking de múltiplos modelos
   - Voting ensemble

3. **Otimização Temporal**
   - Modelos específicos por categoria
   - Segmentação por PDV
   - Forecasting multivariado

## 📞 Suporte

Em caso de dúvidas ou problemas:

1. **Verificar logs** dos notebooks para mensagens de erro
2. **Consultar documentação** no README.md
3. **Revisar configurações** em model_params.yml
4. **Abrir issue** no repositório GitHub

---

**Tempo Estimado Total: 2-4 horas**
- Configuração: 30 minutos
- Preparação dados: 30 minutos  
- Modelagem: 1-3 horas
- Análise: 30 minutos

**🎯 Meta: Obter WMAPE < 15% no conjunto de teste**