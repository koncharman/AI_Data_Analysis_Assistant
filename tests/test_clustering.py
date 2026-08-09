import numpy as np
import pandas as pd
import pytest

from analysis.clustering import (
    get_scaler,
    prepare_clustering_data,
    evaluate_kmeans,
    find_best_k,
    run_kmeans,
)

from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
)


# ============================================================
# Test data
# ============================================================

@pytest.fixture
def sample_dataframe():
    """
    Dataset containing numeric and categorical features.

    It also contains missing values so we can verify that
    the preprocessing pipeline imputes them correctly.
    """

    return pd.DataFrame(
        {
            "age": [
                20, 21, 22,
                40, 41, 42,
                60, 61, 62,
            ],
            "income": [
                1000,
                1100,
                np.nan,
                3000,
                3100,
                3200,
                5000,
                np.nan,
                5200,
            ],
            "department": [
                "A", "A", None,
                "B", "B", "B",
                "C", None, "C",
            ],
        }
    )


# ============================================================
# Scalers
# ============================================================

def test_standard_scaler():
    """
    Verify that the standard scaling option returns
    StandardScaler.
    """

    scaler = get_scaler("standard")

    assert isinstance(
        scaler,
        StandardScaler,
    )


def test_robust_scaler():
    """
    Verify that the robust scaling option returns
    RobustScaler.
    """

    scaler = get_scaler("robust")

    assert isinstance(
        scaler,
        RobustScaler,
    )


def test_minmax_scaler():
    """
    Verify that the min-max option returns
    MinMaxScaler.
    """

    scaler = get_scaler("minmax")

    assert isinstance(
        scaler,
        MinMaxScaler,
    )


def test_invalid_scaler():
    """
    Unknown scaling methods should produce
    a clear error.
    """

    with pytest.raises(
        ValueError,
        match="Unknown scaling method",
    ):
        get_scaler("invalid")


# ============================================================
# Data preparation
# ============================================================

def test_prepare_clustering_data(
    sample_dataframe,
):
    """
    Verify that numeric and categorical variables
    are transformed into a numeric clustering matrix.
    """

    result = prepare_clustering_data(
        dataframe=sample_dataframe,
        numeric_columns=[
            "age",
            "income",
        ],
        categorical_columns=[
            "department",
        ],
        scaling="standard",
    )

    transformed = result["data"]

    # Same number of observations as the original dataset.
    assert transformed.shape[0] == len(
        sample_dataframe
    )

    # The result must contain multiple transformed features.
    assert transformed.shape[1] >= 3

    assert result["numeric_columns"] == [
        "age",
        "income",
    ]

    assert result[
        "categorical_columns"
    ] == [
        "department",
    ]


def test_missing_values_are_imputed(
    sample_dataframe,
):
    """
    Missing numeric and categorical values should be
    imputed before clustering.

    Therefore the transformed matrix should contain
    no missing values.
    """

    result = prepare_clustering_data(
        dataframe=sample_dataframe,
        numeric_columns=[
            "age",
            "income",
        ],
        categorical_columns=[
            "department",
        ],
    )

    transformed = result["data"]

    assert not np.isnan(
        transformed
    ).any()


def test_categorical_is_one_hot_encoded(
    sample_dataframe,
):
    """
    Categorical variables should become binary
    one-hot encoded features.
    """

    result = prepare_clustering_data(
        dataframe=sample_dataframe,
        numeric_columns=["age"],
        categorical_columns=[
            "department",
        ],
    )

    feature_names = result[
        "feature_names"
    ]

    department_features = [
        feature
        for feature in feature_names
        if feature.startswith(
            "department_"
        )
    ]

    assert len(
        department_features
    ) >= 2


def test_constant_column_is_removed():
    """
    Constant variables contain no clustering information
    and should therefore be removed.
    """

    dataframe = pd.DataFrame(
        {
            "age": [
                20, 30, 40, 50
            ],
            "constant": [
                1, 1, 1, 1
            ],
        }
    )

    result = prepare_clustering_data(
        dataframe=dataframe,
        numeric_columns=[
            "age",
            "constant",
        ],
    )

    assert "constant" in result[
        "removed_columns"
    ]

    assert "constant" not in result[
        "numeric_columns"
    ]


# ============================================================
# K-Means evaluation
# ============================================================

def test_evaluate_kmeans():
    """
    Verify that K-Means evaluation returns all
    expected clustering metrics.
    """

    data = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],

            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2],
        ]
    )

    result = evaluate_kmeans(
        data,
        n_clusters=2,
    )

    assert result["k"] == 2

    assert "silhouette_score" in result
    assert "calinski_harabasz_score" in result
    assert "davies_bouldin_score" in result
    assert "inertia" in result

    assert (
        -1
        <= result["silhouette_score"]
        <= 1
    )


# ============================================================
# Best K
# ============================================================

def test_find_best_k():
    """
    Create three clearly separated groups.

    Silhouette score should identify three clusters
    as the best solution.
    """

    data = np.array(
        [
            # Cluster 1
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],

            # Cluster 2
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [10.1, 10.1],

            # Cluster 3
            [20.0, 0.0],
            [20.1, 0.0],
            [20.0, 0.1],
            [20.1, 0.1],
        ]
    )

    result = find_best_k(
        data,
        min_k=2,
        max_k=5,
    )

    assert result["best_k"] == 3

    assert (
        result["selection_metric"]
        == "silhouette_score"
    )

    # k = 2, 3, 4, 5
    assert len(result["results"]) == 4


# ============================================================
# Final K-Means
# ============================================================

def test_run_kmeans():
    """
    Verify that final K-Means returns one cluster label
    for every observation.
    """

    data = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],

            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2],
        ]
    )

    result = run_kmeans(
        data,
        n_clusters=2,
    )

    assert result["method"] == "kmeans"

    assert result["n_clusters"] == 2

    assert len(
        result["labels"]
    ) == len(data)

    assert len(
        result["cluster_centers"]
    ) == 2

    assert "silhouette_score" in result[
        "metrics"
    ]


def test_run_kmeans_invalid_k():
    """
    K must be smaller than the number of observations.
    """

    data = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
    )

    with pytest.raises(
        ValueError,
        match="smaller than",
    ):
        run_kmeans(
            data,
            n_clusters=3,
        )