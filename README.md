# AI Data Analyst

A local AI-assisted data analysis platform built with **Python**, **Streamlit**, **SciPy**, **scikit-learn**, **PyTorch**, **Ollama**, and **LangChain tool calling**.

## Overview

The application loads a dataset, profiles its variables, computes descriptive statistics, runs exploratory tests, performs clustering and text analysis, trains Random Forest and PyTorch models, and exposes those capabilities through an AI data analyst interface.

Core principle:

```text
LLM decides WHAT is needed
Python calculates the result
LLM explains the result
```

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

Example:

```text
Input
  ↓
Linear(64)
  ↓
ReLU
  ↓
Linear(32)
  ↓
ReLU
  ↓
Output
```

Direct input-to-output weights are available with:

```python
hidden_layers=[]
```

## Neural-Network Cross-Validation

```text
Complete dataset
       ↓
Outer K-fold CV
       ↓
Outer training set
       ↓
Internal train/validation split
       ↓
Early stopping
       ↓
Evaluate untouched outer fold
       ↓
Aggregate metrics
       ↓
Median best epoch
       ↓
Train final model on all data
```

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

## Project Structure

```text
AI-Data-Analyst/
├── main.py
├── data/
│   ├── __init__.py
│   ├── data_loader.py
│   └── data_profiler.py
├── analysis/
│   ├── __init__.py
│   ├── statistics.py
│   ├── eda.py
│   ├── clustering.py
│   ├── text_analysis.py
│   ├── machine_learning.py
│   └── neural_networks.py
├── agents/
│   ├── __init__.py
│   ├── tool_context.py
│   ├── analysis_tools.py
│   ├── prompts.py
│   └── data_analyst_agent.py
├── tests/
├── requirements.txt
└── README.md
```

## Streamlit Interface

Recommended tabs:

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

## Dataset Context

```text
Streamlit
   ↓
DatasetContext
   ├── DataFrame
   └── profile
        ↓
Agent tools
```

This prevents the full DataFrame from being passed through the LLM.

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
Cluster the observations using age, height and weight.
```

```text
Train a Random Forest to predict the target.
```

```text
Train a neural network to predict the target.
```

## Installation

```bash
git clone <repository-url>
cd AI-Data-Analyst

python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For the current LangChain setup:

```bash
pip install langchain==0.3.30 langchain-ollama
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

Tests should cover both deterministic analysis functions and agent-facing tool wrappers.

## Reliability Rules

The system prompt instructs the LLM to:
- never invent columns or statistical results;
- use tools for dataset-dependent claims;
- use exact column names;
- prefer statistically valid methods;
- distinguish association from causation;
- avoid expensive analyses unless requested;
- avoid tool calls with missing required arguments.

Agent-facing outputs are compacted so the local model does not receive fitted pipelines, tensors, preprocessors, or huge weight matrices.

## Current Limitations

- Small local models can occasionally choose the wrong tool.
- Tool arguments may still require validation and normalization.
- Neural-network K-fold CV can be computationally expensive.
- Statistical routing still depends partly on the LLM.

## Future Improvements

### Deterministic statistical router
Let the LLM identify intent/columns, while Python selects the valid statistical test.

### Model persistence
Save trained scikit-learn and PyTorch models.

### Prediction interface
Allow users to submit new rows for prediction.

### Monitoring
Track training time, inference latency, model scores, and dataset drift.

### Better visualizations
Add confusion matrices, regression diagnostics, cluster plots, feature-importance plots, and training curves.

### LangGraph later
Introduce LangGraph only when the workflow requires stronger multi-step routing, retries, approvals, persistence, or pause/resume.

## Technologies

- Python
- pandas
- NumPy
- SciPy
- statsmodels
- scikit-learn
- PyTorch
- Streamlit
- LangChain
- Ollama
- pytest

## Portfolio Skills Demonstrated

```text
Automated data profiling
Statistical inference
EDA
Clustering
NLP analysis
Classical machine learning
PyTorch neural networks
Cross-validation
Class imbalance handling
Model interpretation
Tool-calling agents
Local LLM integration
Streamlit UI design
Testing
```

## Engineering Philosophy

The project separates natural-language reasoning from numerical computation.

```text
Language understanding → LLM
Statistical / ML computation → Python
Result interpretation → LLM
```

That separation is the central architectural idea of the AI Data Analyst.
