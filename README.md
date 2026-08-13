# AI Data Analyst

A local AI-assisted (Agentic AI) data analysis framework built with **Python**, **Streamlit**, **SciPy**, **statsmodels**, **scikit-learn**, **PyTorch**, **Ollama**, and **LangChain**.

## Overview

1. Load dataset and profile its variables.
2. Compute Descriptive Statistics
3. Conduct Exploratory Analysis
4. Perform Cluster Analysis
5. Run Text Analysis
6. Train Random Forest and PyTorch models
7. Expose all capabilities through Agentic AI.

## Screenshots
You can see some representative views of the current APP in screenshots/

## Main Features

### Dataset Loading
Supports tabular formats such as:
- CSV
- Excel
- ODS
- Parquet

### Automatic Data Profiling
Columns are classified into:
- numeric
- categorical
- boolean
- text
- datetime
- other

### Descriptive Statistics
Type-aware summaries for numeric, categorical, text, datetime, boolean, and other variables.

### Exploratory Data Analysis

```text
Numeric ↔ Numeric
→ Spearman correlation

Numeric ↔ Binary
→ Mann-Whitney U

Numeric ↔ Categorical (3+ groups)
→ Kruskal-Wallis
→ corrected Mann-Whitney post-hoc comparisons

Categorical ↔ Categorical
→ Chi-square
→ Cramér's V

Binary ↔ Binary
→ Chi-square
→ Phi
```

### K-Means Clustering
Includes:
- numeric feature preprocessing;
- categorical one-hot encoding;
- missing-value imputation;
- configurable scaling;
- best-K evaluation;
- cluster quality metrics.

### Text Analysis
Includes:
- text statistics;
- word frequencies;
- n-grams;
- TF-IDF keywords;
- LDA topic modelling.

### Random Forest

Classification:
- `RandomForestClassifier`
- class imbalance handling
- stratified K-fold CV
- macro F1, weighted F1, balanced accuracy, accuracy

Regression:
- `RandomForestRegressor`
- K-fold CV
- R², MAE, RMSE

Also returns native feature importance.

### PyTorch Feed-Forward Neural Networks

Supports:
- regression;
- binary classification;
- multiclass classification;
- configurable hidden layers;
- configurable activations;
- dropout;
- optional batch normalization;
- multiple optimizers;
- early stopping;
- class-imbalance handling;
- K-fold cross-validation.


## AI Data Analyst Agent

The agent uses:
- Ollama
- LangChain
- tool calling
- deterministic Python analysis functions

Available tools include:

```text
get_dataset_overview()
get_column_statistics(column)
calculate_spearman(first_column, second_column)
compare_two_groups(numeric_column, group_column)
compare_multiple_groups(numeric_column, group_column)
compare_categorical_columns(first_column, second_column)
compare_binary_columns(first_column, second_column)
run_kmeans_analysis(...)
analyze_text_column(...)
train_random_forest_model(...)
train_neural_network_model(...)
```

The full DataFrame remains in Python memory. The LLM receives metadata and compact structured results rather than raw datasets or fitted model objects.

## Agent Architecture

```text
User
  ↓
Streamlit
  ↓
AgentExecutor
  ↓
Ollama
  ↓
Select tool
  ↓
analysis_tools.py
  ↓
statistics / EDA / clustering / text / ML / neural networks
  ↓
Structured result
  ↓
Ollama interpretation
  ↓
User-facing answer
```

## Streamlit Interface

Tabs:

```text
Overview
Statistics
EDA
Clustering
Text Analysis
Random Forest
Neural Network
AI Data Analyst
```

Heavy operations run only after explicit user actions. Dataset/profile state is retained with `st.session_state`.

## Example Agent Questions

```text
What columns are in the dataset?
```

```text
Give me statistics for Weight.
```

```text
Is Age related to Weight?
```

```text
Does Weight differ between smokers and non-smokers?
```

```text
Cluster the observations using age, height, and weight.
```

```text
Train a Random Forest to predict genre as a target.
```

```text
Train a neural network to predict age as a target.
```

## Installation


```bash
pip install -r requirements.txt
```

Ollama version
```bash
ollama --version
0.32.7
```

## Ollama

```bash
ollama pull llama3.2:3b
```

## Run

```bash
streamlit run main.py
```

## Testing

```bash
python -m pytest -v
```
