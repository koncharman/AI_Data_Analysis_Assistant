import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.pipeline import Pipeline

from analysis.machine_learning import (
    build_preprocessor,
    build_random_forest,
    create_cross_validator,
    detect_task_type,
    get_feature_importance,
    prepare_supervised_data,
    select_feature_columns,
    train_random_forest,
)
from data.data_profiler import profile_dataset


# ============================================================
# Reusable datasets
# ============================================================

@pytest.fixture
def classification_dataframe():
    """
    Create a binary classification dataset containing:

    - numeric features;
    - a categorical feature;
    - a boolean feature;
    - missing feature values;
    - an imbalanced target.

    The target contains a learnable relationship with
    age, income and department.
    """

    rng = np.random.RandomState(42)

    number_of_rows = 120

    age = rng.randint(
        18,
        70,
        size=number_of_rows,
    )

    income = rng.normal(
        loc=35000,
        scale=12000,
        size=number_of_rows,
    )

    department = rng.choice(
        ["Sales", "IT", "HR"],
        size=number_of_rows,
        p=[0.5, 0.3, 0.2],
    )

    active = rng.choice(
        [True, False],
        size=number_of_rows,
        p=[0.8, 0.2],
    )

    target_score = (
        age
        + income / 2500
        + (department == "IT") * 8
        + rng.normal(
            0,
            4,
            size=number_of_rows,
        )
    )

    churn = np.where(
        target_score > 72,
        "yes",
        "no",
    )

    dataframe = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "department": department,
            "active": active,
            "churn": churn,
        }
    )

    # Add missing feature values. The preprocessing
    # pipeline should impute these values.
    dataframe.loc[
        [2, 12, 35],
        "income",
    ] = np.nan

    dataframe.loc[
        [5, 41],
        "department",
    ] = None

    return dataframe


@pytest.fixture
def regression_dataframe():
    """
    Create a regression dataset with numeric,
    categorical and boolean features.

    Salary follows a learnable nonlinear relationship
    with experience and education.
    """

    rng = np.random.RandomState(42)

    number_of_rows = 120

    experience = rng.uniform(
        0,
        20,
        size=number_of_rows,
    )

    education = rng.choice(
        ["Bachelor", "Master", "PhD"],
        size=number_of_rows,
    )

    remote = rng.choice(
        [True, False],
        size=number_of_rows,
    )

    education_effect = pd.Series(
        education
    ).map(
        {
            "Bachelor": 0,
            "Master": 10000,
            "PhD": 20000,
        }
    ).to_numpy()

    salary = (
        25000
        + 4000 * experience
        + 120 * experience ** 2
        + education_effect
        + remote * 3000
        + rng.normal(
            0,
            2500,
            size=number_of_rows,
        )
    )

    dataframe = pd.DataFrame(
        {
            "experience": experience,
            "education": education,
            "remote": remote,
            "salary": salary,
        }
    )

    dataframe.loc[
        [4, 20, 65],
        "experience",
    ] = np.nan

    dataframe.loc[
        [8, 44],
        "education",
    ] = None

    return dataframe


# ============================================================
# Task detection
# ============================================================

def test_detect_classification_from_string_target(
    classification_dataframe,
):
    """
    A string target should be detected as classification.
    """

    result = detect_task_type(
        classification_dataframe,
        "churn",
    )

    assert result["task_type"] == "classification"
    assert result["target_column"] == "churn"
    assert result["unique_target_values"] == 2


def test_detect_regression_from_continuous_target(
    regression_dataframe,
):
    """
    A continuous numeric target should be detected
    as regression.
    """

    result = detect_task_type(
        regression_dataframe,
        "salary",
    )

    assert result["task_type"] == "regression"
    assert result["target_column"] == "salary"
    assert result["unique_target_values"] > 20


def test_detect_numeric_classification_warning():
    """
    A numeric target with few unique values should be
    treated as classification with a warning.
    """

    dataframe = pd.DataFrame(
        {
            "feature": range(30),
            "target": [
                0,
                1,
                2,
            ] * 10,
        }
    )

    result = detect_task_type(
        dataframe,
        "target",
    )

    assert result["task_type"] == "classification"
    assert result["warning"] is not None


def test_detect_task_rejects_unknown_target():
    """
    A target that does not exist should be rejected.
    """

    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    with pytest.raises(
        ValueError,
        match="does not exist",
    ):
        detect_task_type(
            dataframe,
            "target",
        )


def test_detect_task_rejects_constant_target():
    """
    A target with only one distinct value cannot be
    used for supervised machine learning.
    """

    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [1, 1, 1, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="at least two distinct values",
    ):
        detect_task_type(
            dataframe,
            "target",
        )


# ============================================================
# Feature selection
# ============================================================

def test_select_feature_columns(
    classification_dataframe,
):
    """
    Numeric, categorical and boolean features should
    be selected, while the target is excluded.
    """

    profile = profile_dataset(
        classification_dataframe
    )

    result = select_feature_columns(
        dataframe=classification_dataframe,
        profile=profile,
        target_column="churn",
    )

    assert "age" in result["numeric_columns"]
    assert "income" in result["numeric_columns"]

    assert "department" in result[
        "categorical_columns"
    ]

    assert "active" in result[
        "categorical_columns"
    ]

    assert "churn" not in result[
        "feature_columns"
    ]


def test_select_only_requested_features(
    classification_dataframe,
):
    """
    Explicit feature selection should be respected.
    """

    profile = profile_dataset(
        classification_dataframe
    )

    result = select_feature_columns(
        dataframe=classification_dataframe,
        profile=profile,
        target_column="churn",
        feature_columns=[
            "age",
            "department",
        ],
    )

    assert result["feature_columns"] == [
        "age",
        "department",
    ]

    assert result["numeric_columns"] == [
        "age",
    ]

    assert result["categorical_columns"] == [
        "department",
    ]


def test_select_features_rejects_unknown_column(
    classification_dataframe,
):
    """
    Unknown feature names should raise a clear error.
    """

    profile = profile_dataset(
        classification_dataframe
    )

    with pytest.raises(
        ValueError,
        match="do not exist",
    ):
        select_feature_columns(
            dataframe=classification_dataframe,
            profile=profile,
            target_column="churn",
            feature_columns=[
                "age",
                "unknown_feature",
            ],
        )


# ============================================================
# Supervised data preparation
# ============================================================

def test_prepare_supervised_data(
    classification_dataframe,
):
    """
    Preparation should return X, y and the selected
    feature groups.
    """

    profile = profile_dataset(
        classification_dataframe
    )

    result = prepare_supervised_data(
        dataframe=classification_dataframe,
        profile=profile,
        target_column="churn",
    )

    assert len(result["X"]) == len(
        classification_dataframe
    )

    assert len(result["y"]) == len(
        classification_dataframe
    )

    assert "churn" not in result["X"].columns

    assert result["removed_target_rows"] == 0


def test_prepare_supervised_data_removes_missing_targets():
    """
    Rows with missing target values should be removed.

    Missing feature values should remain because they are
    handled inside the preprocessing pipeline.
    """

    dataframe = pd.DataFrame(
        {
            "age": [
                20,
                np.nan,
                40,
                50,
            ],
            "target": [
                "A",
                "B",
                None,
                "A",
            ],
        }
    )

    profile = profile_dataset(
        dataframe
    )

    result = prepare_supervised_data(
        dataframe=dataframe,
        profile=profile,
        target_column="target",
    )

    assert len(result["X"]) == 3
    assert len(result["y"]) == 3
    assert result["removed_target_rows"] == 1

    # The feature imputer has not run yet.
    assert result["X"]["age"].isna().sum() == 1


# ============================================================
# Preprocessing
# ============================================================

def test_preprocessor_handles_missing_values(
    classification_dataframe,
):
    """
    Numeric and categorical missing values should be
    imputed before model training.
    """

    preprocessor = build_preprocessor(
        numeric_columns=[
            "age",
            "income",
        ],
        categorical_columns=[
            "department",
            "active",
        ],
    )

    transformed = preprocessor.fit_transform(
        classification_dataframe[
            [
                "age",
                "income",
                "department",
                "active",
            ]
        ]
    )

    assert transformed.shape[0] == len(
        classification_dataframe
    )

    assert not np.isnan(
        transformed
    ).any()


def test_preprocessor_returns_feature_names(
    classification_dataframe,
):
    """
    The fitted preprocessor should expose numeric and
    one-hot encoded feature names.
    """

    preprocessor = build_preprocessor(
        numeric_columns=[
            "age",
            "income",
        ],
        categorical_columns=[
            "department",
        ],
    )

    preprocessor.fit(
        classification_dataframe[
            [
                "age",
                "income",
                "department",
            ]
        ]
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    assert "age" in feature_names
    assert "income" in feature_names

    assert any(
        name.startswith("department_")
        for name in feature_names
    )


# ============================================================
# Random Forest creation
# ============================================================

def test_build_random_forest_classifier():
    """
    Classification should create a RandomForestClassifier
    with balanced class weights by default.
    """

    model = build_random_forest(
        task_type="classification",
    )

    assert isinstance(
        model,
        RandomForestClassifier,
    )

    assert model.class_weight == "balanced"

    assert model.n_estimators == 300


def test_build_classifier_without_balance():
    """
    Class weighting should be disabled when requested.
    """

    model = build_random_forest(
        task_type="classification",
        handle_class_imbalance=False,
    )

    assert isinstance(
        model,
        RandomForestClassifier,
    )

    assert model.class_weight is None


def test_build_random_forest_regressor():
    """
    Regression should create a RandomForestRegressor.
    """

    model = build_random_forest(
        task_type="regression",
    )

    assert isinstance(
        model,
        RandomForestRegressor,
    )

    assert model.n_estimators == 300


def test_build_random_forest_custom_estimators():
    """
    The requested number of trees should be respected.
    """

    model = build_random_forest(
        task_type="classification",
        n_estimators=25,
    )

    assert model.n_estimators == 25


def test_build_random_forest_rejects_invalid_task():
    """
    Unsupported task types should be rejected.
    """

    with pytest.raises(
        ValueError,
        match="classification.*regression",
    ):
        build_random_forest(
            task_type="clustering",
        )


def test_build_random_forest_rejects_invalid_estimators():
    """
    A Random Forest requires at least one tree.
    """

    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        build_random_forest(
            task_type="classification",
            n_estimators=0,
        )


# ============================================================
# Cross-validation configuration
# ============================================================

def test_classification_uses_ten_folds(
    classification_dataframe,
):
    """
    Ten folds should be used when every class contains
    at least ten observations.
    """

    y = classification_dataframe[
        "churn"
    ]

    _, actual_folds, warning = (
        create_cross_validator(
            y=y,
            task_type="classification",
            requested_folds=10,
        )
    )

    assert actual_folds == 10
    assert warning is None


def test_classification_reduces_fold_count():
    """
    Classification folds should not exceed the size
    of the smallest target class.
    """

    y = pd.Series(
        ["majority"] * 20
        + ["minority"] * 4
    )

    _, actual_folds, warning = (
        create_cross_validator(
            y=y,
            task_type="classification",
            requested_folds=10,
        )
    )

    assert actual_folds == 4
    assert warning is not None
    assert "reduced" in warning


def test_classification_rejects_single_observation_class():
    """
    Stratified cross-validation requires at least two
    observations in every class.
    """

    y = pd.Series(
        ["A"] * 10
        + ["B"]
    )

    with pytest.raises(
        ValueError,
        match="at least two observations",
    ):
        create_cross_validator(
            y=y,
            task_type="classification",
            requested_folds=10,
        )


def test_regression_uses_ten_folds(
    regression_dataframe,
):
    """
    Regression should use ten folds when enough rows exist.
    """

    y = regression_dataframe[
        "salary"
    ]

    _, actual_folds, warning = (
        create_cross_validator(
            y=y,
            task_type="regression",
            requested_folds=10,
        )
    )

    assert actual_folds == 10
    assert warning is None


def test_regression_reduces_folds_for_small_dataset():
    """
    Regression folds cannot exceed the number
    of observations.
    """

    y = pd.Series(
        [1.0, 2.0, 3.0, 4.0]
    )

    _, actual_folds, warning = (
        create_cross_validator(
            y=y,
            task_type="regression",
            requested_folds=10,
        )
    )

    assert actual_folds == 4
    assert warning is not None


# ============================================================
# Complete classification flow
# ============================================================

def test_train_random_forest_classification(
    classification_dataframe,
):
    """
    The complete classification flow should return
    cross-validation metrics and a fitted pipeline.
    """

    profile = profile_dataset(
        classification_dataframe
    )

    result = train_random_forest(
        dataframe=classification_dataframe,
        profile=profile,
        target_column="churn",
        task_type="classification",

        # Use fewer trees and folds in the test suite
        # to keep tests quick.
        n_estimators=30,
        cv_folds=3,
        random_state=42,
    )

    assert result["task_type"] == "classification"

    assert result["selection_metric"] == "f1_macro"

    assert result["actual_cv_folds"] == 3

    assert result["n_estimators"] == 30

    assert result[
        "handle_class_imbalance"
    ] is True

    metrics = result[
        "cross_validation"
    ]["metrics"]

    assert "f1_weighted" in metrics
    assert "f1_macro" in metrics
    assert "balanced_accuracy" in metrics
    assert "accuracy" in metrics

    assert len(
        metrics["f1_macro"]["fold_scores"]
    ) == 3

    assert (
        0
        <= metrics["f1_macro"]["mean"]
        <= 1
    )

    assert result["primary_score"] == (
        metrics["f1_macro"]["mean"]
    )

    assert isinstance(
        result["fitted_pipeline"],
        Pipeline,
    )


def test_train_classification_without_balance(
    classification_dataframe,
):
    """
    The imbalance-handling option should be configurable.
    """

    profile = profile_dataset(
        classification_dataframe
    )

    result = train_random_forest(
        dataframe=classification_dataframe,
        profile=profile,
        target_column="churn",
        task_type="classification",
        n_estimators=20,
        cv_folds=3,
        handle_class_imbalance=False,
    )

    assert result[
        "handle_class_imbalance"
    ] is False

    fitted_model = result[
        "fitted_pipeline"
    ].named_steps["model"]

    assert fitted_model.class_weight is None


# ============================================================
# Complete regression flow
# ============================================================

def test_train_random_forest_regression(
    regression_dataframe,
):
    """
    The complete regression flow should return R²,
    MAE and RMSE across cross-validation folds.
    """

    profile = profile_dataset(
        regression_dataframe
    )

    result = train_random_forest(
        dataframe=regression_dataframe,
        profile=profile,
        target_column="salary",
        task_type="regression",
        n_estimators=30,
        cv_folds=3,
        random_state=42,
    )

    assert result["task_type"] == "regression"

    assert result["selection_metric"] == "r2"

    assert result["actual_cv_folds"] == 3

    metrics = result[
        "cross_validation"
    ]["metrics"]

    assert "r2" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics

    assert len(
        metrics["r2"]["fold_scores"]
    ) == 3

    assert metrics["mae"]["mean"] >= 0
    assert metrics["rmse"]["mean"] >= 0

    assert result["primary_score"] == (
        metrics["r2"]["mean"]
    )

    assert isinstance(
        result["fitted_pipeline"],
        Pipeline,
    )


def test_train_uses_automatic_task_detection(
    regression_dataframe,
):
    """
    Task detection should be used when task_type
    is not supplied.
    """

    profile = profile_dataset(
        regression_dataframe
    )

    result = train_random_forest(
        dataframe=regression_dataframe,
        profile=profile,
        target_column="salary",
        n_estimators=20,
        cv_folds=3,
    )

    assert result["task_type"] == "regression"


# ============================================================
# Feature importance
# ============================================================

def test_classification_feature_importance(
    classification_dataframe,
):
    """
    Classification should return ranked feature importance.
    """

    profile = profile_dataset(
        classification_dataframe
    )

    result = train_random_forest(
        dataframe=classification_dataframe,
        profile=profile,
        target_column="churn",
        task_type="classification",
        n_estimators=30,
        cv_folds=3,
    )

    importance = result[
        "feature_importance"
    ]

    assert len(importance) > 0
    assert importance[0]["rank"] == 1

    assert all(
        "feature" in item
        and "importance" in item
        and "rank" in item
        for item in importance
    )

    scores = [
        item["importance"]
        for item in importance
    ]

    # Results should be ordered from highest
    # to lowest importance.
    assert scores == sorted(
        scores,
        reverse=True,
    )

    # Random Forest importance normally sums to one.
    assert sum(scores) == pytest.approx(
        1.0
    )


def test_regression_feature_importance(
    regression_dataframe,
):
    """
    Regression should return importance for numeric
    and transformed categorical features.
    """

    profile = profile_dataset(
        regression_dataframe
    )

    result = train_random_forest(
        dataframe=regression_dataframe,
        profile=profile,
        target_column="salary",
        task_type="regression",
        n_estimators=30,
        cv_folds=3,
    )

    importance = result[
        "feature_importance"
    ]

    feature_names = {
        item["feature"]
        for item in importance
    }

    assert "experience" in feature_names

    assert any(
        name.startswith("education_")
        for name in feature_names
    )

    assert sum(
        item["importance"]
        for item in importance
    ) == pytest.approx(1.0)


def test_get_feature_importance_rejects_non_pipeline():
    """
    The feature-importance function requires a fitted
    sklearn Pipeline.
    """

    with pytest.raises(
        TypeError,
        match="sklearn Pipeline",
    ):
        get_feature_importance(
            "not a pipeline"
        )


def test_get_feature_importance_rejects_unfitted_pipeline():
    """
    Importance cannot be retrieved before fitting
    the preprocessing and model steps.
    """

    preprocessor = build_preprocessor(
        numeric_columns=["age"],
        categorical_columns=[],
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=5,
                    random_state=42,
                ),
            ),
        ]
    )

    # Scikit-learn may raise NotFittedError or AttributeError
    # depending on which unfitted component is accessed first.
    with pytest.raises(Exception):
        get_feature_importance(
            pipeline
        )


# ============================================================
# Invalid regression target
# ============================================================

def test_regression_rejects_non_numeric_target():
    """
    Explicit regression should reject a target that cannot
    be converted to numeric values.
    """

    dataframe = pd.DataFrame(
        {
            "age": list(range(30)),
            "target": [
                "low",
                "medium",
                "high",
            ] * 10,
        }
    )

    profile = profile_dataset(
        dataframe
    )

    with pytest.raises(
        ValueError,
        match="numeric target",
    ):
        train_random_forest(
            dataframe=dataframe,
            profile=profile,
            target_column="target",
            task_type="regression",
            n_estimators=10,
            cv_folds=3,
        )