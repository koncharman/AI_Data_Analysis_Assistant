from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from langchain.tools import tool

from agents.tool_context import dataset_context
from analysis.statistics import get_dataset_statistics
from analysis.eda import (
    binary_association,
    categorical_association,
    kruskal_wallis_with_posthoc,
    mann_whitney_test,
    spearman_correlation,
)
from analysis.clustering import find_best_k, prepare_clustering_data, run_kmeans
from analysis.text_analysis import (
    get_ngrams,
    get_text_statistics,
    get_tfidf_keywords,
    get_word_frequencies,
    run_topic_modeling,
)
from analysis.machine_learning import train_random_forest
from analysis.neural_networks import train_feed_forward_network

def _json_safe(value: Any, max_list_items: int = 250) -> Any:
    """Convert results to compact JSON-safe values for the LLM."""
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, max_list_items)
            for key, item in value.items()
            if key not in {
                "model", "fitted_pipeline", "preprocessor", "label_encoder",
                "vectorizer", "training_history", "final_training_history",
            }
        }
    if isinstance(value, (list, tuple)):
        values = list(value)
        result = [_json_safe(item, max_list_items) for item in values[:max_list_items]]
        if len(values) > max_list_items:
            result.append({"truncated_items": len(values) - max_list_items})
        return result
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.head(100).to_dict(orient="records"), max_list_items)
    if isinstance(value, pd.Series):
        return _json_safe(value.head(100).tolist(), max_list_items)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist(), max_list_items)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def _require_column(column: str) -> None:
    dataframe = dataset_context.get_dataframe()
    if column not in dataframe.columns:
        raise ValueError("Column '{}' does not exist.".format(column))


@tool
def get_dataset_overview() -> Dict[str, Any]:
    """Return dataset dimensions, columns, semantic types, missing counts, and a tiny preview."""
    dataframe = dataset_context.get_dataframe()
    profile = dataset_context.get_profile()
    return _json_safe({
        "dataset_id": dataset_context.get_dataset_id(),
        "rows": dataframe.shape[0],
        "columns": dataframe.shape[1],
        "column_names": dataframe.columns.tolist(),
        "numeric_columns": profile.get("numeric_columns", []),
        "categorical_columns": profile.get("categorical_columns", []),
        "boolean_columns": profile.get("boolean_columns", []),
        "text_columns": profile.get("text_columns", []),
        "datetime_columns": profile.get("datetime_columns", []),
        "other_columns": profile.get("other_columns", []),
        "missing_values": dataframe.isna().sum().sort_values(ascending=False).head(30).to_dict(),
        "preview": dataframe.head(3).to_dict(orient="records"),
    })


@tool
def get_column_statistics(column: str) -> Dict[str, Any]:
    """Return deterministic descriptive statistics for one exact column name."""
    _require_column(column)
    result = get_dataset_statistics(
        dataset_context.get_dataframe(), dataset_context.get_profile()
    )
    for section_name, section in result.items():
        if isinstance(section, dict) and column in section:
            return _json_safe({
                "column": column,
                "variable_type": section_name,
                "statistics": section[column],
            })
    raise ValueError("No statistics were produced for '{}'.".format(column))


@tool
def calculate_spearman(first_column: str, second_column: str) -> Dict[str, Any]:
    """Calculate Spearman correlation for two numeric or ordinal-compatible columns."""
    return _json_safe(spearman_correlation(
        dataset_context.get_dataframe(), first_column, second_column
    ))


@tool
def compare_two_groups(numeric_column: str, group_column: str) -> Dict[str, Any]:
    """Compare a numeric variable across exactly two independent groups with Mann-Whitney U."""
    return _json_safe(mann_whitney_test(
        dataset_context.get_dataframe(), numeric_column, group_column
    ))


@tool
def compare_multiple_groups(numeric_column: str, group_column: str) -> Dict[str, Any]:
    """Compare a numeric variable across three or more groups with Kruskal-Wallis and post-hoc tests."""
    return _json_safe(kruskal_wallis_with_posthoc(
        dataset_context.get_dataframe(), numeric_column, group_column
    ))


@tool
def compare_categorical_columns(first_column: str, second_column: str) -> Dict[str, Any]:
    """Measure association between categorical variables using chi-square and Cramer's V."""
    return _json_safe(categorical_association(
        dataset_context.get_dataframe(), first_column, second_column
    ))


@tool
def compare_binary_columns(first_column: str, second_column: str) -> Dict[str, Any]:
    """Measure association between two binary variables using chi-square and Phi."""
    return _json_safe(binary_association(
        dataset_context.get_dataframe(), first_column, second_column
    ))


@tool
def run_kmeans_analysis(
    numeric_columns: List[str],
    categorical_columns: Optional[List[str]] = None,
    scaling: str = "standard",
    minimum_k: int = 2,
    maximum_k: int = 8,
) -> Dict[str, Any]:
    """Find the best K by silhouette score and run K-Means on selected features."""
    prepared = prepare_clustering_data(
        dataframe=dataset_context.get_dataframe(),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns or [],
        scaling=scaling,
    )
    best = find_best_k(prepared["data"], min_k=minimum_k, max_k=maximum_k)
    final = run_kmeans(prepared["data"], n_clusters=best["best_k"])
    labels = pd.Series(final["labels"])
    return _json_safe({
        "feature_names": prepared["feature_names"],
        "removed_columns": prepared["removed_columns"],
        "scaling": scaling,
        "best_k_analysis": best,
        "final_metrics": final["metrics"],
        "cluster_sizes": labels.value_counts().sort_index().to_dict(),
        "cluster_centers": final["cluster_centers"],
    })


@tool
def analyze_text_column(
    column: str,
    analysis: str = "statistics",
    top_n: int = 20,
    ngram_size: int = 2,
    n_topics: int = 5,
    words_per_topic: int = 10,
    min_df: int = 2,
) -> Dict[str, Any]:
    """Analyze a text column. analysis: statistics, frequencies, ngrams, tfidf, or topics."""
    dataframe = dataset_context.get_dataframe()
    name = analysis.lower()
    if name == "statistics":
        result = get_text_statistics(dataframe, column)
    elif name == "frequencies":
        result = get_word_frequencies(dataframe, column, top_n=top_n)
    elif name == "ngrams":
        result = get_ngrams(dataframe, column, n=ngram_size, top_n=top_n)
    elif name == "tfidf":
        result = get_tfidf_keywords(dataframe, column, top_n=top_n)
    elif name == "topics":
        result = run_topic_modeling(
            dataframe, column, n_topics=n_topics, words_per_topic=words_per_topic,
            min_df=min_df,
        )
    else:
        raise ValueError("analysis must be statistics, frequencies, ngrams, tfidf, or topics.")
    return _json_safe(result)


@tool
def train_random_forest_model(
    target_column: str,
    feature_columns: Optional[Any] = None,
    task_type: Optional[str] = None,
    cv_folds: int = 10,
    n_estimators: int = 300,
    handle_class_imbalance: bool = True,
) -> Dict[str, Any]:
    """
    Train and cross-validate a Random Forest.

    You can omit everything apart from target_column. Omit every variable not mentioned, do not include them in the query.

    target_column must be an exact existing dataset column name.

    feature_columns must contain only exact existing dataset column names.

    If the user does not explicitly specify feature_columns,
    omit feature_columns. The tool will automatically use all
    eligible predictor columns from the dataset profile, excluding
    the target.

    feature_columns: format is a Python list (example: []), it can be omited if the user does not give input.

    Never invent, infer, rename, abbreviate, or generate feature
    names that are not present in the dataset.
    """

    if isinstance(cv_folds, str):
        cv_folds=int(cv_folds)

    if isinstance(n_estimators, str):
        n_estimators = int(n_estimators)


    dataframe = dataset_context.get_dataframe()
    profile = dataset_context.get_profile()

    # --------------------------------------------------------
    # Validate target
    # --------------------------------------------------------

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Unknown target column: {target_column}"
        )

    # --------------------------------------------------------
    # Resolve feature columns
    # --------------------------------------------------------

    if not isinstance(feature_columns, list) or len(feature_columns) == 0:

        resolved_features = (
            profile.get("numeric_columns", [])
            + profile.get("categorical_columns", [])
            + profile.get("boolean_columns", [])
        )

        resolved_features = [
            column
            for column in resolved_features
            if column != target_column
        ]

    else:

        unknown_features = [
            column
            for column in feature_columns
            if column not in dataframe.columns
        ]

        if unknown_features:
            raise ValueError(
                "Unknown feature columns: {}".format(
                    unknown_features
                )
            )

        resolved_features = [
            column
            for column in feature_columns
            if column != target_column
        ]

    if isinstance(handle_class_imbalance, bool):

        resolved_imbalance = handle_class_imbalance

    elif isinstance(handle_class_imbalance, str):

        resolved_imbalance = (
            handle_class_imbalance
            .strip()
            .lower()
            in {
                "true",
                "1",
                "yes",
            }
        )

    else:
        resolved_imbalance = True



    result = train_random_forest(
        dataframe=dataset_context.get_dataframe(),
        profile=dataset_context.get_profile(),
        target_column=target_column,
        feature_columns=resolved_features,
        task_type=task_type,
        cv_folds=cv_folds,
        n_estimators=n_estimators,
        handle_class_imbalance=resolved_imbalance,
    )
    return _json_safe(result)


@tool
def train_neural_network_model(
    target_column: str,
    feature_columns: Optional[Any] = None,
    task_type: Optional[str] = None,
    hidden_layers: Optional[Any] = None,
    activation: Optional[str] = None,
    cv_folds: Optional[int] = 10,
    max_epochs: Optional[int] = 100,
    patience: Optional[int] = None,
    learning_rate: Optional[float] = 0.05,
    batch_size: Optional[int] = 16,
    handle_class_imbalance: Optional[bool] = True,
) -> Dict[str, Any]:
    """
    Train and evaluate a PyTorch feed-forward neural network.

    You can omit everything apart from target_column. Omit every variable not mentioned, do not include them in the query.

    target_column must be an exact existing dataset column name.

    feature_columns must contain only exact existing dataset column names.

    If the user does not explicitly specify feature_columns,
    omit feature_columns. The tool will automatically use all
    eligible predictor columns from the dataset profile, excluding
    the target.

    feature_columns: format is a Python list (example: []), it can be omited if the user does not give input.

    activation: format is a Python string, it can be omited if the user does not give input.

    hidden_layers: format is a Python list (example: []), it can be omited if the user does not give input.

    Never invent, infer, rename, abbreviate, or generate feature
    names that are not present in the dataset.

    Returns a compact summary of model performance suitable
    for interpretation by the agent. Raw neural-network
    weight matrices are not returned.

    activation and hidden_layers are related to the layers of the network. They can be empty or lists.
    """

    if isinstance(cv_folds, str):
        cv_folds = int(cv_folds)

    if isinstance(max_epochs, str):
        max_epochs = int(max_epochs)

    if isinstance(learning_rate, str):
        learning_rate = int(learning_rate)

    if isinstance(batch_size, str):
        learning_rate = int(learning_rate)

    # --------------------------------------------------------
    # Normalize task type
    # --------------------------------------------------------

    dataframe = dataset_context.get_dataframe()
    profile = dataset_context.get_profile()

    # --------------------------------------------------------
    # Validate target
    # --------------------------------------------------------

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Unknown target column: {target_column}"
        )

    # --------------------------------------------------------
    # Resolve feature columns
    # --------------------------------------------------------

    if not isinstance(feature_columns, list) or len(feature_columns) == 0:

        resolved_features = (
            profile.get("numeric_columns", [])
            + profile.get("categorical_columns", [])
            + profile.get("boolean_columns", [])
        )

        resolved_features = [
            column
            for column in resolved_features
            if column != target_column
        ]

    else:

        unknown_features = [
            column
            for column in feature_columns
            if column not in dataframe.columns
        ]

        if unknown_features:
            raise ValueError(
                "Unknown feature columns: {}".format(
                    unknown_features
                )
            )

        resolved_features = [
            column
            for column in feature_columns
            if column != target_column
        ]

    if isinstance(handle_class_imbalance, bool):

        resolved_imbalance = handle_class_imbalance

    elif isinstance(handle_class_imbalance, str):

        resolved_imbalance = (
            handle_class_imbalance
            .strip()
            .lower()
            in {
                "true",
                "1",
                "yes",
            }
        )

    else:
        resolved_imbalance = True

    resolved_task_type = None

    if isinstance(task_type, str):
        cleaned_task_type = task_type.strip().lower()

        if cleaned_task_type in {
            "classification",
            "regression",
        }:
            resolved_task_type = cleaned_task_type

    # --------------------------------------------------------
    # Normalize hidden layers
    # --------------------------------------------------------

    if isinstance(hidden_layers, list):

        resolved_hidden_layers = [
            int(size)
            for size in hidden_layers
            if isinstance(size, (int, float))
            and int(size) > 0
        ]

    else:
        # Default architecture when the model supplies
        # None, {}, "", or another invalid value.
        resolved_hidden_layers = []

    # --------------------------------------------------------
    # Normalize activation
    # --------------------------------------------------------

    resolved_activation = (
        activation.strip().lower()
        if isinstance(activation, str)
        and activation.strip()
        else "identity"
    )

    valid_activations = {
        "relu",
        "leaky_relu",
        "elu",
        "gelu",
        "selu",
        "tanh",
        "sigmoid",
        "identity",
    }

    if resolved_activation not in valid_activations:
        resolved_activation = "identity"

    # --------------------------------------------------------
    # Normalize numeric options
    # --------------------------------------------------------

    resolved_cv_folds = (
        int(cv_folds)
        if isinstance(cv_folds, int)
        and cv_folds >= 2
        else 10
    )

    resolved_max_epochs = (
        int(max_epochs)
        if isinstance(max_epochs, int)
        and max_epochs >= 1
        else 100
    )

    resolved_patience = (
        int(patience)
        if isinstance(patience, int)
        and patience >= 1
        else 10
    )

    resolved_learning_rate = (
        float(learning_rate)
        if isinstance(
            learning_rate,
            (int, float),
        )
        and learning_rate > 0
        else 0.001
    )

    resolved_batch_size = (
        int(batch_size)
        if isinstance(batch_size, int)
        and batch_size >= 1
        else 32
    )


    # --------------------------------------------------------
    # Run the actual neural-network analysis
    # --------------------------------------------------------

    result = train_feed_forward_network(
        dataframe=dataset_context.get_dataframe(),
        profile=dataset_context.get_profile(),
        target_column=target_column,
        feature_columns=resolved_features,
        task_type=resolved_task_type,
        hidden_layers=resolved_hidden_layers,
        activations=resolved_activation,
        cv_folds=resolved_cv_folds,
        max_epochs=resolved_max_epochs,
        patience=resolved_patience,
        learning_rate=resolved_learning_rate,
        batch_size=resolved_batch_size,
        handle_class_imbalance=resolved_imbalance,
    )

    # --------------------------------------------------------
    # Build compact cross-validation metrics
    # --------------------------------------------------------
    # The complete result may contain large PyTorch weight
    # matrices. Those are useful internally but should not be
    # sent to the LLM because they can overwhelm its context.
    # --------------------------------------------------------

    cross_validation = result.get(
        "cross_validation",
        {},
    )

    metrics = cross_validation.get(
        "metrics",
        {},
    )

    compact_metrics = {}

    for metric_name, metric_values in metrics.items():

        if not isinstance(metric_values, dict):
            continue

        compact_metrics[metric_name] = {
            key: float(value)
            if isinstance(value, (int, float))
            else value
            for key, value in metric_values.items()
            if key in {
                "mean",
                "standard_deviation",
                "minimum",
                "maximum",
            }
        }

    # --------------------------------------------------------
    # Build the result returned to the agent
    # --------------------------------------------------------

    compact_result = {
        "method": "feed_forward_neural_network",

        "task_type": result.get(
            "task_type",
            resolved_task_type,
        ),

        "target_column": result.get(
            "target_column",
            target_column,
        ),

        "feature_columns": result.get(
            "feature_columns",
            feature_columns,
        ),

        "class_names": result.get(
            "class_names"
        ),

        "number_of_classes": result.get(
            "number_of_classes"
        ),

        "rows_used": result.get(
            "rows_used"
        ),

        "architecture": result.get(
            "architecture"
        ),

        "requested_cv_folds": result.get(
            "requested_cv_folds",
            resolved_cv_folds,
        ),

        "actual_cv_folds": result.get(
            "actual_cv_folds"
        ),

        "cv_warning": result.get(
            "cv_warning"
        ),

        "primary_metric": result.get(
            "primary_metric"
        ),

        "primary_score": result.get(
            "primary_score"
        ),

        "cross_validation_metrics":
            compact_metrics,

        "final_epochs": result.get(
            "final_epochs"
        ),

        "handle_class_imbalance":
            resolved_imbalance,
    }

    # --------------------------------------------------------
    # Handle direct input-output weights
    # --------------------------------------------------------
    # Direct weights are interpretable only when there are no
    # hidden layers between the inputs and outputs.
    #
    # For networks with hidden layers, raw matrices are omitted.
    # --------------------------------------------------------

    weights_result = result.get(
        "weights",
        {},
    )

    direct_weights_available = (
        weights_result.get(
            "direct_input_output_weights",
            False,
        )
        if isinstance(weights_result, dict)
        else False
    )

    compact_result[
        "direct_input_output_weights"
    ] = direct_weights_available

    if direct_weights_available:

        direct_weights = weights_result.get(
            "direct_weights"
        )

        # Regression or binary classification
        if (
            isinstance(direct_weights, dict)
            and isinstance(
                direct_weights.get("weights"),
                list,
            )
        ):
            compact_result[
                "top_direct_weights"
            ] = direct_weights[
                "weights"
            ][:15]

        # Multiclass classification
        elif isinstance(
            direct_weights,
            dict,
        ):
            compact_result[
                "top_direct_weights_by_class"
            ] = {}

            for (
                class_name,
                class_result,
            ) in direct_weights.items():

                if not isinstance(
                    class_result,
                    dict,
                ):
                    continue

                class_weights = (
                    class_result.get(
                        "weights",
                        [],
                    )
                )

                compact_result[
                    "top_direct_weights_by_class"
                ][class_name] = (
                    class_weights[:10]
                )

    else:
        compact_result[
            "weight_note"
        ] = (
            "The neural network contains hidden layers. "
            "Raw layer weight matrices were omitted because "
            "they are not direct feature effects."
        )

    return compact_result

ANALYSIS_TOOLS = [
    get_dataset_overview,
    get_column_statistics,
    calculate_spearman,
    compare_two_groups,
    compare_multiple_groups,
    compare_categorical_columns,
    compare_binary_columns,
    run_kmeans_analysis,
    analyze_text_column,
    train_random_forest_model,
    train_neural_network_model,
]
