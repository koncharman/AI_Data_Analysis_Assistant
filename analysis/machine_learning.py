from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# General validation
# ============================================================

def _validate_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate the DataFrame used for machine learning.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    if dataframe.shape[1] == 0:
        raise ValueError(
            "The DataFrame does not contain any columns."
        )

    if dataframe.shape[0] < 3:
        raise ValueError(
            "At least three observations are required."
        )


def _validate_target(
    dataframe: pd.DataFrame,
    target_column: str,
) -> None:
    """
    Validate the selected target column.
    """

    if not isinstance(target_column, str):
        raise TypeError(
            "target_column must be a string."
        )

    if target_column not in dataframe.columns:
        raise ValueError(
            "Target column '{}' does not exist.".format(
                target_column
            )
        )

    non_missing_target = dataframe[
        target_column
    ].dropna()

    if non_missing_target.empty:
        raise ValueError(
            "The target column contains only missing values."
        )

    if non_missing_target.nunique() < 2:
        raise ValueError(
            "The target column must contain at least "
            "two distinct values."
        )


# ============================================================
# Task detection
# ============================================================

def detect_task_type(
    dataframe: pd.DataFrame,
    target_column: str,
    max_class_values: int = 20,
) -> Dict[str, Any]:
    """
    Determine whether a target suggests classification
    or regression.

    Rules:
        - Boolean, object, category and string targets:
          classification.

        - Numeric targets with at most max_class_values:
          classification with an ambiguity warning.

        - Numeric targets with more distinct values:
          regression.

    The recommendation can be overridden by passing
    task_type explicitly to train_random_forest().
    """

    _validate_dataframe(dataframe)
    _validate_target(dataframe, target_column)

    target = dataframe[
        target_column
    ].dropna()

    unique_count = int(
        target.nunique()
    )

    warning = None

    if (
        pd.api.types.is_bool_dtype(target)
        or pd.api.types.is_object_dtype(target)
        or isinstance(
            target.dtype,
            pd.CategoricalDtype,
        )
        or pd.api.types.is_string_dtype(target)
    ):
        task_type = "classification"

    elif pd.api.types.is_numeric_dtype(target):

        if unique_count <= max_class_values:
            task_type = "classification"

            warning = (
                "The target is numeric but has only {} "
                "distinct values. It was treated as "
                "classification. Set task_type='regression' "
                "to override this."
            ).format(unique_count)

        else:
            task_type = "regression"

    else:
        task_type = "classification"

        warning = (
            "The target has an uncommon dtype and was "
            "treated as classification."
        )

    return {
        "task_type": task_type,
        "target_column": target_column,
        "target_dtype": str(target.dtype),
        "unique_target_values": unique_count,
        "warning": warning,
    }


# ============================================================
# Feature selection
# ============================================================

def select_feature_columns(
    dataframe: pd.DataFrame,
    profile: Dict[str, Any],
    target_column: str,
    feature_columns: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Select numeric and categorical features.

    Text and datetime columns are excluded from V1 because
    they require specialized feature engineering.

    Boolean columns are treated as categorical variables.
    """

    if not isinstance(profile, dict):
        raise TypeError(
            "profile must be a dictionary."
        )

    required_profile_keys = [
        "numeric_columns",
        "categorical_columns",
        "boolean_columns",
    ]

    for key in required_profile_keys:

        if key not in profile:
            raise ValueError(
                "The profile is missing '{}'.".format(
                    key
                )
            )

    # By default, use all supported columns identified
    # by the data profiler.
    if feature_columns is None:
        feature_columns = (
            profile["numeric_columns"]
            + profile["categorical_columns"]
            + profile["boolean_columns"]
        )

    if not isinstance(feature_columns, list):
        raise TypeError(
            "feature_columns must be a list or None."
        )

    # Remove duplicate feature names while retaining order.
    feature_columns = list(
        dict.fromkeys(feature_columns)
    )

    # Ensure the target is never accidentally used
    # as an input feature.
    feature_columns = [
        column
        for column in feature_columns
        if column != target_column
    ]

    unknown_columns = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if unknown_columns:
        raise ValueError(
            "Feature columns do not exist: {}".format(
                unknown_columns
            )
        )

    numeric_columns = [
        column
        for column in feature_columns
        if column in profile["numeric_columns"]
    ]

    categorical_columns = [
        column
        for column in feature_columns
        if (
            column in profile["categorical_columns"]
            or column in profile["boolean_columns"]
        )
    ]

    selected_columns = (
        numeric_columns
        + categorical_columns
    )

    if not selected_columns:
        raise ValueError(
            "No supported machine-learning features "
            "were selected."
        )

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "feature_columns": selected_columns,
    }


# ============================================================
# One-hot encoder compatibility
# ============================================================

def _create_one_hot_encoder() -> OneHotEncoder:
    """
    Create a dense OneHotEncoder.

    Newer scikit-learn versions use sparse_output=False.
    Older versions use sparse=False.
    """

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


# ============================================================
# Preprocessing
# ============================================================

def build_preprocessor(
    numeric_columns: List[str],
    categorical_columns: List[str],
) -> ColumnTransformer:
    """
    Build preprocessing for numeric and categorical features.

    Numeric features:
        - Replace missing values with the median.

    Categorical features:
        - Replace missing values with the most frequent value.
        - Convert categories to one-hot encoded columns.

    Scaling is not required for Random Forest models.
    """

    transformers = []

    if numeric_columns:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            )
        )

    if categorical_columns:

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
                    _create_one_hot_encoder(),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError(
            "At least one numeric or categorical "
            "feature is required."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ============================================================
# Prepare supervised data
# ============================================================

def prepare_supervised_data(
    dataframe: pd.DataFrame,
    profile: Dict[str, Any],
    target_column: str,
    feature_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Prepare features and the target for supervised learning.

    Rows with missing target values are removed.

    Missing feature values are retained because they are
    handled by the preprocessing pipeline inside each
    cross-validation fold.
    """

    _validate_dataframe(dataframe)
    _validate_target(
        dataframe,
        target_column,
    )

    selected = select_feature_columns(
        dataframe=dataframe,
        profile=profile,
        target_column=target_column,
        feature_columns=feature_columns,
    )

    all_columns = (
        selected["feature_columns"]
        + [target_column]
    )

    data = dataframe[
        all_columns
    ].copy()

    original_row_count = len(data)

    # We do not impute target values because doing so would
    # introduce artificial labels or outcomes.
    data = data.dropna(
        subset=[target_column]
    )

    if len(data) < 3:
        raise ValueError(
            "Too few observations remain after removing "
            "missing target values."
        )

    X = data[
        selected["feature_columns"]
    ]

    y = data[target_column]

    return {
        "X": X,
        "y": y,
        "numeric_columns":
            selected["numeric_columns"],
        "categorical_columns":
            selected["categorical_columns"],
        "feature_columns":
            selected["feature_columns"],
        "removed_target_rows": int(
            original_row_count - len(data)
        ),
        "row_indices": data.index.tolist(),
    }


# ============================================================
# Random Forest model
# ============================================================

def build_random_forest(
    task_type: str,
    random_state: int = 42,
    n_estimators: int = 300,
    handle_class_imbalance: bool = True,
) -> Any:
    """
    Create a Random Forest model.

    Classification uses class_weight='balanced' when
    imbalance handling is enabled.

    Regression does not use class weights.
    """

    if n_estimators < 1:
        raise ValueError(
            "n_estimators must be at least 1."
        )

    if task_type == "classification":

        class_weight = (
            "balanced"
            if handle_class_imbalance
            else None
        )

        return RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )

    if task_type == "regression":

        return RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(
        "task_type must be 'classification' "
        "or 'regression'."
    )


# ============================================================
# Cross-validation configuration
# ============================================================

def create_cross_validator(
    y: pd.Series,
    task_type: str,
    requested_folds: int = 10,
    random_state: int = 42,
) -> Tuple[Any, int, Optional[str]]:
    """
    Create an appropriate cross-validation splitter.

    Classification:
        - Stratified K-Fold.
        - The number of folds cannot exceed the size
          of the smallest class.

    Regression:
        - Shuffled K-Fold.
        - The number of folds cannot exceed the number
          of observations.
    """

    if not isinstance(requested_folds, int):
        raise TypeError(
            "requested_folds must be an integer."
        )

    if requested_folds < 2:
        raise ValueError(
            "requested_folds must be at least 2."
        )

    warning = None

    if task_type == "classification":

        class_counts = y.value_counts()

        smallest_class_count = int(
            class_counts.min()
        )

        if smallest_class_count < 2:
            raise ValueError(
                "Every target class must contain at least "
                "two observations for cross-validation."
            )

        actual_folds = min(
            requested_folds,
            smallest_class_count,
        )

        if actual_folds < requested_folds:
            warning = (
                "Cross-validation was reduced from {} to {} "
                "folds because the smallest class contains "
                "{} observations."
            ).format(
                requested_folds,
                actual_folds,
                smallest_class_count,
            )

        cross_validator = StratifiedKFold(
            n_splits=actual_folds,
            shuffle=True,
            random_state=random_state,
        )

        return (
            cross_validator,
            actual_folds,
            warning,
        )

    if task_type == "regression":

        actual_folds = min(
            requested_folds,
            len(y),
        )

        if actual_folds < 3:
            raise ValueError(
                "At least three observations are required "
                "for regression cross-validation."
            )

        if actual_folds < requested_folds:
            warning = (
                "Cross-validation was reduced from {} to {} "
                "folds because only {} observations are "
                "available."
            ).format(
                requested_folds,
                actual_folds,
                len(y),
            )

        cross_validator = KFold(
            n_splits=actual_folds,
            shuffle=True,
            random_state=random_state,
        )

        return (
            cross_validator,
            actual_folds,
            warning,
        )

    raise ValueError(
        "task_type must be 'classification' "
        "or 'regression'."
    )


# ============================================================
# Evaluation helpers
# ============================================================

def _evaluate_classification_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate classification metrics for one fold.
    """

    return {
        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
    }


def _evaluate_regression_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate regression metrics for one fold.
    """

    return {
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
    }


def _summarize_fold_metrics(
    fold_metrics: List[Dict[str, float]],
) -> Dict[str, Any]:
    """
    Summarize metrics across all cross-validation folds.
    """

    if not fold_metrics:
        raise ValueError(
            "No fold metrics were supplied."
        )

    metric_names = list(
        fold_metrics[0].keys()
    )

    summary = {}

    for metric_name in metric_names:

        values = [
            fold[metric_name]
            for fold in fold_metrics
        ]

        summary[metric_name] = {
            "mean": float(
                np.mean(values)
            ),
            "standard_deviation": float(
                np.std(
                    values,
                    ddof=1,
                )
                if len(values) > 1
                else 0.0
            ),
            "minimum": float(
                np.min(values)
            ),
            "maximum": float(
                np.max(values)
            ),
            "fold_scores": [
                float(value)
                for value in values
            ],
        }

    return summary


# ============================================================
# Random Forest cross-validation
# ============================================================

def cross_validate_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_columns: List[str],
    categorical_columns: List[str],
    model: Any,
    task_type: str,
    cross_validator: Any,
) -> Dict[str, Any]:
    """
    Evaluate Random Forest with manual cross-validation.

    Preprocessing is fitted separately inside every training
    fold, preventing information leakage from validation data.
    """

    fold_metrics = []

    split_target = (
        y
        if task_type == "classification"
        else None
    )

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        cross_validator.split(
            X,
            split_target,
        ),
        start=1,
    ):

        X_train = X.iloc[
            train_indices
        ]

        X_validation = X.iloc[
            validation_indices
        ]

        y_train = y.iloc[
            train_indices
        ]

        y_validation = y.iloc[
            validation_indices
        ]

        # Build a new preprocessor for every fold so that
        # medians, modes and categories are learned only
        # from the fold's training data.
        preprocessor = build_preprocessor(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    clone(model),
                ),
            ]
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        predictions = pipeline.predict(
            X_validation
        )

        if task_type == "classification":

            metrics = (
                _evaluate_classification_predictions(
                    y_validation,
                    predictions,
                )
            )

        else:

            metrics = (
                _evaluate_regression_predictions(
                    y_validation,
                    predictions,
                )
            )

        fold_metrics.append(
            metrics
        )

    return {
        "model_name": "random_forest",
        "folds": int(
            len(fold_metrics)
        ),
        "metrics":
            _summarize_fold_metrics(
                fold_metrics
            ),
        "fold_results":
            fold_metrics,
    }


# ============================================================
# Fit final model
# ============================================================

def fit_final_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_columns: List[str],
    categorical_columns: List[str],
    model: Any,
) -> Pipeline:
    """
    Fit Random Forest on all available valid observations.
    """

    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                clone(model),
            ),
        ]
    )

    pipeline.fit(
        X,
        y,
    )

    return pipeline


# ============================================================
# Feature importance
# ============================================================

def get_feature_importance(
    fitted_pipeline: Pipeline,
) -> List[Dict[str, Any]]:
    """
    Extract native Random Forest feature importance.

    Importance values correspond to transformed features,
    including individual one-hot encoded categories.
    """

    if not isinstance(
        fitted_pipeline,
        Pipeline,
    ):
        raise TypeError(
            "fitted_pipeline must be a sklearn Pipeline."
        )

    preprocessor = fitted_pipeline.named_steps[
        "preprocessor"
    ]

    model = fitted_pipeline.named_steps[
        "model"
    ]

    if not hasattr(
        model,
        "feature_importances_",
    ):
        raise ValueError(
            "The fitted model does not expose "
            "feature_importances_."
        )

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    importances = model.feature_importances_

    if len(feature_names) != len(importances):
        raise RuntimeError(
            "Feature names and importance values "
            "have different lengths."
        )

    results = [
        {
            "feature": str(feature_name),
            "importance": float(importance),
        }
        for feature_name, importance
        in zip(
            feature_names,
            importances,
        )
    ]

    results.sort(
        key=lambda item:
            item["importance"],
        reverse=True,
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        result["rank"] = int(rank)

    return results


# ============================================================
# Main machine-learning function
# ============================================================

def train_random_forest(
    dataframe: pd.DataFrame,
    profile: Dict[str, Any],
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    task_type: Optional[str] = None,
    cv_folds: int = 10,
    handle_class_imbalance: bool = True,
    n_estimators: int = 300,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train and evaluate a Random Forest model.

    Classification:
        - RandomForestClassifier
        - balanced class weights when enabled
        - stratified cross-validation
        - primary metric: macro F1

    Regression:
        - RandomForestRegressor
        - shuffled K-fold cross-validation
        - primary metric: R²

    After cross-validation, the model is fitted on all valid
    observations and feature importance is returned.
    """

    # --------------------------------------------------------
    # Detect or validate the task type
    # --------------------------------------------------------

    task_detection = detect_task_type(
        dataframe=dataframe,
        target_column=target_column,
    )

    if task_type is None:

        resolved_task_type = task_detection[
            "task_type"
        ]

    else:

        resolved_task_type = task_type.lower()

        if resolved_task_type not in {
            "classification",
            "regression",
        }:
            raise ValueError(
                "task_type must be 'classification' "
                "or 'regression'."
            )

    # --------------------------------------------------------
    # Prepare features and target
    # --------------------------------------------------------

    prepared = prepare_supervised_data(
        dataframe=dataframe,
        profile=profile,
        target_column=target_column,
        feature_columns=feature_columns,
    )

    X = prepared["X"]
    y = prepared["y"]

    numeric_columns = prepared[
        "numeric_columns"
    ]

    categorical_columns = prepared[
        "categorical_columns"
    ]

    # A regression target must contain valid numbers.
    if resolved_task_type == "regression":

        numeric_y = pd.to_numeric(
            y,
            errors="coerce",
        )

        invalid_target_count = int(
            numeric_y.isna().sum()
        )

        if invalid_target_count > 0:
            raise ValueError(
                "Regression requires a numeric target. "
                "{} target values could not be converted."
                .format(invalid_target_count)
            )

        y = numeric_y

    # --------------------------------------------------------
    # Create cross-validation folds
    # --------------------------------------------------------

    (
        cross_validator,
        actual_folds,
        cv_warning,
    ) = create_cross_validator(
        y=y,
        task_type=resolved_task_type,
        requested_folds=cv_folds,
        random_state=random_state,
    )

    # --------------------------------------------------------
    # Create the Random Forest model
    # --------------------------------------------------------

    model = build_random_forest(
        task_type=resolved_task_type,
        random_state=random_state,
        n_estimators=n_estimators,
        handle_class_imbalance=(
            handle_class_imbalance
        ),
    )

    # --------------------------------------------------------
    # Run cross-validation
    # --------------------------------------------------------

    cross_validation_result = (
        cross_validate_random_forest(
            X=X,
            y=y,
            numeric_columns=numeric_columns,
            categorical_columns=(
                categorical_columns
            ),
            model=model,
            task_type=resolved_task_type,
            cross_validator=cross_validator,
        )
    )

    if resolved_task_type == "classification":

        # Macro F1 gives every target class equal importance,
        # making it useful when classes are imbalanced.
        selection_metric = "f1_macro"

    else:

        selection_metric = "r2"

    primary_score = float(
        cross_validation_result[
            "metrics"
        ][selection_metric]["mean"]
    )

    # --------------------------------------------------------
    # Fit final model using all valid rows
    # --------------------------------------------------------

    fitted_pipeline = fit_final_random_forest(
        X=X,
        y=y,
        numeric_columns=numeric_columns,
        categorical_columns=(
            categorical_columns
        ),
        model=model,
    )

    # --------------------------------------------------------
    # Extract native feature importance
    # --------------------------------------------------------

    feature_importance = get_feature_importance(
        fitted_pipeline
    )

    # --------------------------------------------------------
    # Return structured result
    # --------------------------------------------------------

    return {
        "task_type": resolved_task_type,
        "task_detection": task_detection,
        "target_column": target_column,
        "feature_columns":
            prepared["feature_columns"],
        "numeric_columns":
            numeric_columns,
        "categorical_columns":
            categorical_columns,
        "rows_used": int(
            len(X)
        ),
        "removed_target_rows":
            prepared["removed_target_rows"],
        "requested_cv_folds": int(
            cv_folds
        ),
        "actual_cv_folds": int(
            actual_folds
        ),
        "cv_warning": cv_warning,
        "handle_class_imbalance": bool(
            handle_class_imbalance
        ),
        "n_estimators": int(
            n_estimators
        ),
        "selection_metric":
            selection_metric,
        "cross_validation":
            cross_validation_result,
        "primary_score":
            primary_score,
        "feature_importance":
            feature_importance,
        "fitted_pipeline":
            fitted_pipeline,
    }