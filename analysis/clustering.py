
from typing import Any,Optional

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
)


# ============================================================
# Scaling
# ============================================================

def get_scaler(
    method: str = "standard",
):
    """
    Return the requested numeric scaler.

    Available methods:
        standard:
            Mean = 0 and standard deviation = 1.

        robust:
            Uses the median and interquartile range.
            Useful when numeric variables contain outliers.

        minmax:
            Scales values approximately between 0 and 1.

        none:
            Do not scale numeric variables.
            Usually not recommended for K-Means.
    """

    method = method.lower()

    if method == "standard":
        return StandardScaler()

    if method == "robust":
        return RobustScaler()

    if method == "minmax":
        return MinMaxScaler()

    if method == "none":
        return "passthrough"

    raise ValueError(
        "Unknown scaling method. "
        "Choose from: standard, robust, minmax, none."
    )


# ============================================================
# Data preparation
# ============================================================

def prepare_clustering_data(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
        categorical_columns: Optional[list[str]] = None,
        scaling: str = "standard",
) -> dict[str, Any]:
    """
    Prepare numeric and categorical variables for K-Means.

    Steps:
        1. Select requested variables.
        2. Remove completely missing columns.
        3. Remove constant columns.
        4. Median-impute missing numeric values.
        5. Scale numeric variables.
        6. Impute categorical missing values.
        7. One-hot encode categorical variables.

    The original DataFrame is never modified.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    if categorical_columns is None:
        categorical_columns = []

    selected_columns = (
        numeric_columns
        + categorical_columns
    )

    if len(selected_columns) == 0:
        raise ValueError(
            "At least one clustering feature is required."
        )

    # --------------------------------------------------------
    # Validate column names
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in selected_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Columns do not exist: {missing_columns}"
        )

    # Work on a copy so the original dataset remains unchanged.
    data = dataframe[
        selected_columns
    ].copy()

    # --------------------------------------------------------
    # Convert requested numeric columns to numeric
    # --------------------------------------------------------

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Remove unusable columns
    # --------------------------------------------------------

    removed_columns = []

    for column in selected_columns:

        # Completely missing columns cannot contribute
        # information to clustering.
        if data[column].isna().all():
            removed_columns.append(column)

        # Constant columns also contain no clustering
        # information.
        elif data[column].nunique(
            dropna=True
        ) <= 1:
            removed_columns.append(column)

    if removed_columns:
        data = data.drop(
            columns=removed_columns
        )

    numeric_columns_used = [
        column
        for column in numeric_columns
        if column not in removed_columns
    ]

    categorical_columns_used = [
        column
        for column in categorical_columns
        if column not in removed_columns
    ]

    if data.shape[1] == 0:
        raise ValueError(
            "No usable clustering features remain."
        )

    # --------------------------------------------------------
    # Numeric preprocessing
    # --------------------------------------------------------

    transformers = []

    if numeric_columns_used:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    get_scaler(
                        scaling
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns_used,
            )
        )

    # --------------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------------

    if categorical_columns_used:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns_used,
            )
        )

    # --------------------------------------------------------
    # Combine numeric + categorical transformations
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    transformed_data = (
        preprocessor.fit_transform(data)
    )

    feature_names = (
        preprocessor.get_feature_names_out()
        .tolist()
    )

    return {
        "data": transformed_data,
        "feature_names": feature_names,
        "numeric_columns": numeric_columns_used,
        "categorical_columns":
            categorical_columns_used,
        "removed_columns": removed_columns,
        "scaling": scaling,
        "preprocessor": preprocessor,
    }


# ============================================================
# Evaluate one K
# ============================================================

def evaluate_kmeans(
    data: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Fit K-Means for one value of K and calculate
    clustering quality metrics.
    """

    if n_clusters < 2:
        raise ValueError(
            "n_clusters must be at least 2."
        )

    if n_clusters >= len(data):
        raise ValueError(
            "n_clusters must be smaller than "
            "the number of observations."
        )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )

    labels = model.fit_predict(data)

    # Some pathological datasets may produce fewer actual
    # clusters than requested.
    unique_clusters = np.unique(labels)

    if len(unique_clusters) < 2:
        raise ValueError(
            "K-Means produced fewer than two clusters."
        )

    return {
        "k": int(n_clusters),

        # Higher is generally better.
        "silhouette_score": float(
            silhouette_score(
                data,
                labels,
            )
        ),

        # Higher is generally better.
        "calinski_harabasz_score": float(
            calinski_harabasz_score(
                data,
                labels,
            )
        ),

        # Lower is generally better.
        "davies_bouldin_score": float(
            davies_bouldin_score(
                data,
                labels,
            )
        ),

        # Within-cluster sum of squared distances.
        # Useful for an elbow analysis.
        "inertia": float(
            model.inertia_
        ),
    }


# ============================================================
# Find best K
# ============================================================

def find_best_k(
    data: np.ndarray,
    min_k: int = 2,
    max_k: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Evaluate several values of K.

    The best K is selected using the highest silhouette score.

    Other metrics are also returned so that the application
    or LLM can explain the clustering quality.
    """

    number_of_rows = len(data)

    if number_of_rows < 3:
        raise ValueError(
            "At least three observations are required "
            "to evaluate K-Means clustering."
        )

    if min_k < 2:
        raise ValueError(
            "min_k must be at least 2."
        )

    if max_k < min_k:
        raise ValueError(
            "max_k must be greater than or equal to min_k."
        )

    # K must always be smaller than the number of samples.
    effective_max_k = min(
        max_k,
        number_of_rows - 1,
    )

    results = []

    for k in range(
        min_k,
        effective_max_k + 1,
    ):

        result = evaluate_kmeans(
            data=data,
            n_clusters=k,
            random_state=random_state,
        )

        results.append(result)

    if not results:
        raise ValueError(
            "No valid K values could be evaluated."
        )

    # For V1 we use silhouette score as the primary
    # criterion for choosing K.
    best_result = max(
        results,
        key=lambda result:
            result["silhouette_score"],
    )

    return {
        "best_k": best_result["k"],
        "selection_metric": "silhouette_score",
        "best_silhouette_score":
            best_result["silhouette_score"],
        "results": results,
    }


# ============================================================
# Run final K-Means
# ============================================================

def run_kmeans(
    data: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Fit the final K-Means model.

    Returns cluster labels, centers, and evaluation metrics.
    """

    if n_clusters < 2:
        raise ValueError(
            "n_clusters must be at least 2."
        )

    if n_clusters >= len(data):
        raise ValueError(
            "n_clusters must be smaller than "
            "the number of observations."
        )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )

    labels = model.fit_predict(data)

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "K-Means produced fewer than two clusters."
        )

    metrics = {
        "silhouette_score": float(
            silhouette_score(
                data,
                labels,
            )
        ),
        "calinski_harabasz_score": float(
            calinski_harabasz_score(
                data,
                labels,
            )
        ),
        "davies_bouldin_score": float(
            davies_bouldin_score(
                data,
                labels,
            )
        ),
        "inertia": float(
            model.inertia_
        ),
    }

    return {
        "method": "kmeans",
        "n_clusters": int(n_clusters),
        "labels": labels.tolist(),
        "cluster_centers":
            model.cluster_centers_.tolist(),
        "metrics": metrics,
        "model": model,
    }