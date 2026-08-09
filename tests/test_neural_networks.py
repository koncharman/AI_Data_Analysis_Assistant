import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from analysis.neural_networks import (
    FeedForwardNetwork,
    build_neural_network_preprocessor,
    calculate_metrics,
    create_neural_cross_validator,
    extract_network_weights,
    get_activation,
    predict_feed_forward_network,
    prepare_neural_network_data,
    resolve_activations,
    train_feed_forward_network,
)
from data.data_profiler import profile_dataset


@pytest.fixture
def classification_dataframe():
    rng = np.random.RandomState(42)
    n = 120

    age = rng.randint(18, 70, n)
    income = rng.normal(35000, 9000, n)
    department = rng.choice(["Sales", "IT", "HR"], n)
    active = rng.choice([True, False], n)

    score = (
        age
        + income / 2500
        + (department == "IT") * 8
        + active * 3
        + rng.normal(0, 3, n)
    )
    target = np.where(score > 65, "yes", "no")

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "department": department,
            "active": active,
            "target": target,
        }
    )
    df.loc[[2, 10], "income"] = np.nan
    df.loc[[4, 20], "department"] = None
    return df


@pytest.fixture
def regression_dataframe():
    rng = np.random.RandomState(42)
    n = 120

    experience = rng.uniform(0, 20, n)
    education = rng.choice(["Bachelor", "Master", "PhD"], n)
    remote = rng.choice([True, False], n)

    effect = pd.Series(education).map(
        {"Bachelor": 0, "Master": 8000, "PhD": 16000}
    ).to_numpy()

    salary = (
        25000
        + 4500 * experience
        + effect
        + remote * 2500
        + rng.normal(0, 1500, n)
    )

    df = pd.DataFrame(
        {
            "experience": experience,
            "education": education,
            "remote": remote,
            "salary": salary,
        }
    )
    df.loc[[3, 18], "experience"] = np.nan
    df.loc[[5, 27], "education"] = None
    return df


# ============================================================
# Activations / architecture
# ============================================================

@pytest.mark.parametrize(
    "name, expected",
    [
        ("relu", nn.ReLU),
        ("leaky_relu", nn.LeakyReLU),
        ("elu", nn.ELU),
        ("gelu", nn.GELU),
        ("selu", nn.SELU),
        ("tanh", nn.Tanh),
        ("sigmoid", nn.Sigmoid),
        ("identity", nn.Identity),
    ],
)
def test_get_activation(name, expected):
    assert isinstance(get_activation(name), expected)


def test_resolve_activations():
    assert resolve_activations([8, 4], "relu") == ["relu", "relu"]
    assert resolve_activations([8, 4], ["relu", "tanh"]) == ["relu", "tanh"]


def test_direct_network():
    model = FeedForwardNetwork(5, 1, hidden_layers=[])
    linear = [x for x in model.network if isinstance(x, nn.Linear)]
    assert len(linear) == 1
    assert linear[0].in_features == 5
    assert linear[0].out_features == 1


def test_hidden_network():
    model = FeedForwardNetwork(
        5,
        3,
        hidden_layers=[8, 4],
        activations=["relu", "tanh"],
        dropout=0.1,
    )
    output = model(torch.randn(7, 5))
    assert output.shape == (7, 3)
    assert model.activation_names == ["relu", "tanh"]


# ============================================================
# Preprocessing / preparation
# ============================================================

def test_preprocessor_handles_missing(classification_dataframe):
    preprocessor = build_neural_network_preprocessor(
        ["age", "income"],
        ["department", "active"],
    )
    X = preprocessor.fit_transform(
        classification_dataframe[
            ["age", "income", "department", "active"]
        ]
    )
    X = np.asarray(X, dtype=np.float32)
    assert not np.isnan(X).any()


def test_prepare_classification(classification_dataframe):
    profile = profile_dataset(classification_dataframe)
    result = prepare_neural_network_data(
        classification_dataframe,
        profile,
        "target",
        task_type="classification",
    )
    assert result["task_type"] == "classification"
    assert result["number_of_classes"] == 2
    assert result["output_size"] == 1
    assert len(result["X"]) == len(classification_dataframe)


def test_prepare_regression(regression_dataframe):
    profile = profile_dataset(regression_dataframe)
    result = prepare_neural_network_data(
        regression_dataframe,
        profile,
        "salary",
        task_type="regression",
    )
    assert result["task_type"] == "regression"
    assert result["output_size"] == 1
    assert result["number_of_classes"] is None


# ============================================================
# 10-fold CV configuration
# ============================================================

def test_classification_uses_ten_folds(classification_dataframe):
    profile = profile_dataset(classification_dataframe)
    prepared = prepare_neural_network_data(
        classification_dataframe,
        profile,
        "target",
        task_type="classification",
    )
    _, actual_folds, warning = create_neural_cross_validator(
        prepared["y"],
        "classification",
        requested_folds=10,
    )
    assert actual_folds == 10
    assert warning is None


def test_classification_reduces_folds():
    y = np.array([0] * 20 + [1] * 4)
    _, actual_folds, warning = create_neural_cross_validator(
        y,
        "classification",
        requested_folds=10,
    )
    assert actual_folds == 4
    assert warning is not None


def test_regression_uses_ten_folds(regression_dataframe):
    y = regression_dataframe["salary"].to_numpy()
    _, actual_folds, warning = create_neural_cross_validator(
        y,
        "regression",
        requested_folds=10,
    )
    assert actual_folds == 10
    assert warning is None


# ============================================================
# Metrics / weights
# ============================================================

def test_classification_metrics():
    metrics = calculate_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
        "classification",
    )
    assert "f1_macro" in metrics
    assert "balanced_accuracy" in metrics


def test_regression_metrics():
    metrics = calculate_metrics(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.1, 1.9, 3.2]),
        "regression",
    )
    assert "r2" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics


def test_direct_weights():
    model = FeedForwardNetwork(3, 1, hidden_layers=[])
    result = extract_network_weights(
        model,
        ["a", "b", "c"],
    )
    assert result["direct_input_output_weights"] is True
    assert len(result["direct_weights"]["weights"]) == 3


def test_hidden_weights_are_not_direct():
    model = FeedForwardNetwork(3, 1, hidden_layers=[5])
    result = extract_network_weights(
        model,
        ["a", "b", "c"],
    )
    assert result["direct_input_output_weights"] is False
    assert result["direct_weights"] is None
    assert len(result["layers"]) == 2


# ============================================================
# Complete CV training
#
# Use 3 folds and few epochs in tests for speed.
# Production default remains 10 folds.
# ============================================================

def test_train_classification_cv(classification_dataframe):
    profile = profile_dataset(classification_dataframe)

    result = train_feed_forward_network(
        dataframe=classification_dataframe,
        profile=profile,
        target_column="target",
        task_type="classification",
        hidden_layers=[],
        cv_folds=3,
        max_epochs=8,
        patience=3,
        learning_rate=0.01,
        batch_size=32,
        device_name="cpu",
        random_state=42,
    )

    assert result["task_type"] == "classification"
    assert result["actual_cv_folds"] == 3
    assert result["primary_metric"] == "f1_macro"
    assert len(result["cross_validation"]["folds"]) == 3
    assert len(
        result["cross_validation"]["metrics"]["f1_macro"]["fold_scores"]
    ) == 3
    assert result["final_epochs"] >= 1
    assert result["weights"]["direct_input_output_weights"] is True


def test_train_regression_cv(regression_dataframe):
    profile = profile_dataset(regression_dataframe)

    result = train_feed_forward_network(
        dataframe=regression_dataframe,
        profile=profile,
        target_column="salary",
        task_type="regression",
        hidden_layers=[],
        cv_folds=3,
        max_epochs=8,
        patience=3,
        learning_rate=0.01,
        batch_size=32,
        device_name="cpu",
        random_state=42,
    )

    assert result["task_type"] == "regression"
    assert result["actual_cv_folds"] == 3
    assert result["primary_metric"] == "r2"
    assert len(result["cross_validation"]["folds"]) == 3
    assert "r2" in result["cross_validation"]["metrics"]
    assert "mae" in result["cross_validation"]["metrics"]
    assert "rmse" in result["cross_validation"]["metrics"]


def test_hidden_network_cv(classification_dataframe):
    profile = profile_dataset(classification_dataframe)

    result = train_feed_forward_network(
        dataframe=classification_dataframe,
        profile=profile,
        target_column="target",
        task_type="classification",
        hidden_layers=[12, 6],
        activations=["relu", "tanh"],
        cv_folds=3,
        max_epochs=5,
        patience=2,
        batch_size=32,
        device_name="cpu",
        random_state=42,
    )

    assert result["architecture"]["hidden_layers"] == [12, 6]
    assert result["architecture"]["activations"] == ["relu", "tanh"]
    assert result["weights"]["direct_input_output_weights"] is False


# ============================================================
# Prediction from final model
# ============================================================

def test_predict_classification(classification_dataframe):
    profile = profile_dataset(classification_dataframe)

    result = train_feed_forward_network(
        dataframe=classification_dataframe,
        profile=profile,
        target_column="target",
        task_type="classification",
        hidden_layers=[],
        cv_folds=2,
        max_epochs=4,
        patience=2,
        batch_size=32,
        device_name="cpu",
        random_state=42,
    )

    new_data = classification_dataframe[
        result["feature_columns"]
    ].head(4)

    predictions = predict_feed_forward_network(
        result,
        new_data,
    )

    assert len(predictions["predictions"]) == 4
    assert len(predictions["positive_probabilities"]) == 4


def test_predict_regression(regression_dataframe):
    profile = profile_dataset(regression_dataframe)

    result = train_feed_forward_network(
        dataframe=regression_dataframe,
        profile=profile,
        target_column="salary",
        task_type="regression",
        hidden_layers=[],
        cv_folds=2,
        max_epochs=4,
        patience=2,
        batch_size=32,
        device_name="cpu",
        random_state=42,
    )

    new_data = regression_dataframe[
        result["feature_columns"]
    ].head(5)

    predictions = predict_feed_forward_network(
        result,
        new_data,
    )

    assert len(predictions["predictions"]) == 5
    assert all(isinstance(x, float) for x in predictions["predictions"])
