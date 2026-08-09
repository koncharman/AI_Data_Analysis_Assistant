from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from analysis.machine_learning import detect_task_type, select_feature_columns


# ============================================================
# General helpers
# ============================================================

def set_random_seed(random_state: int = 42) -> None:
    """Set NumPy/PyTorch seeds for reproducible training."""
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


def _create_one_hot_encoder() -> OneHotEncoder:
    """Create a dense encoder compatible with old/new sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_activation(name: str) -> nn.Module:
    """Return a supported PyTorch activation."""
    mapping = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "selu": nn.SELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "identity": nn.Identity,
    }
    if not isinstance(name, str):
        raise TypeError("activation must be a string.")
    key = name.lower()
    if key not in mapping:
        raise ValueError("Unsupported activation '{}'.".format(name))
    return mapping[key]()


def resolve_activations(
    hidden_layers: List[int],
    activations: Union[str, Sequence[str]],
) -> List[str]:
    """Resolve one activation name for every hidden layer."""
    if isinstance(activations, str):
        return [activations] * len(hidden_layers)
    values = list(activations)
    if len(values) != len(hidden_layers):
        raise ValueError(
            "When activations is a sequence, it must contain "
            "one activation for every hidden layer."
        )
    return values


# ============================================================
# Network
# ============================================================

class FeedForwardNetwork(nn.Module):
    """
    Configurable feed-forward network.

    hidden_layers=[] gives direct input -> output weights.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: Optional[List[int]] = None,
        activations: Union[str, Sequence[str]] = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
    ):
        super().__init__()

        hidden_layers = [] if hidden_layers is None else hidden_layers

        if input_size < 1 or output_size < 1:
            raise ValueError("input_size and output_size must be at least 1.")
        if any((not isinstance(x, int) or x < 1) for x in hidden_layers):
            raise ValueError("Every hidden layer size must be a positive integer.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be between 0 and 1.")

        activation_names = resolve_activations(hidden_layers, activations)

        layers = []
        previous_size = input_size

        for hidden_size, activation_name in zip(hidden_layers, activation_names):
            layers.append(nn.Linear(previous_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(get_activation(activation_name))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_size = hidden_size

        # Raw output only: losses below handle logits correctly.
        layers.append(nn.Linear(previous_size, output_size))

        self.network = nn.Sequential(*layers)
        self.hidden_layers = list(hidden_layers)
        self.activation_names = activation_names

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


# ============================================================
# Data preparation
# ============================================================

def build_neural_network_preprocessor(
    numeric_columns: List[str],
    categorical_columns: List[str],
) -> ColumnTransformer:
    """
    Numeric: median imputation + standard scaling.
    Categorical: mode imputation + one-hot encoding.
    """
    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", _create_one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("At least one usable feature is required.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_neural_network_data(
    dataframe: pd.DataFrame,
    profile: Dict[str, Any],
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Select X/y and encode the target.

    Feature preprocessing is NOT fitted here because it must
    be fitted independently inside each cross-validation fold.
    """
    if target_column not in dataframe.columns:
        raise ValueError("Target column '{}' does not exist.".format(target_column))

    detection = detect_task_type(dataframe, target_column)

    resolved_task = detection["task_type"] if task_type is None else task_type.lower()
    if resolved_task not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification' or 'regression'.")

    selected = select_feature_columns(
        dataframe=dataframe,
        profile=profile,
        target_column=target_column,
        feature_columns=feature_columns,
    )

    data = dataframe[selected["feature_columns"] + [target_column]].copy()
    original_rows = len(data)
    data = data.dropna(subset=[target_column])

    if len(data) < 3:
        raise ValueError("At least 3 rows with non-missing targets are required.")

    X = data[selected["feature_columns"]].reset_index(drop=True)
    raw_y = data[target_column].reset_index(drop=True)

    label_encoder = None
    class_names = None
    number_of_classes = None

    if resolved_task == "regression":
        y_numeric = pd.to_numeric(raw_y, errors="coerce")
        if y_numeric.isna().any():
            raise ValueError("Regression requires a numeric target.")
        y = y_numeric.to_numpy(dtype=np.float32)
        output_size = 1
    else:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(raw_y.astype(str))
        class_names = label_encoder.classes_.astype(str).tolist()
        number_of_classes = len(class_names)
        if number_of_classes < 2:
            raise ValueError("Classification requires at least two classes.")
        output_size = 1 if number_of_classes == 2 else number_of_classes

    return {
        "X": X,
        "y": np.asarray(y),
        "task_type": resolved_task,
        "task_detection": detection,
        "feature_columns": selected["feature_columns"],
        "numeric_columns": selected["numeric_columns"],
        "categorical_columns": selected["categorical_columns"],
        "label_encoder": label_encoder,
        "class_names": class_names,
        "number_of_classes": number_of_classes,
        "output_size": output_size,
        "rows_used": len(data),
        "removed_target_rows": original_rows - len(data),
    }


# ============================================================
# Cross-validation
# ============================================================

def create_neural_cross_validator(
    y: np.ndarray,
    task_type: str,
    requested_folds: int = 10,
    random_state: int = 42,
):
    """
    Create StratifiedKFold for classification or KFold for
    regression. Reduce folds safely when the dataset requires it.
    """
    if requested_folds < 2:
        raise ValueError("requested_folds must be at least 2.")

    if task_type == "classification":
        counts = pd.Series(y).value_counts()
        smallest_class = int(counts.min())

        if smallest_class < 2:
            raise ValueError(
                "Every class needs at least two observations "
                "for stratified cross-validation."
            )

        actual_folds = min(requested_folds, smallest_class)
        warning = None
        if actual_folds < requested_folds:
            warning = (
                "Cross-validation folds reduced from {} to {} "
                "because of the smallest class."
            ).format(requested_folds, actual_folds)

        return (
            StratifiedKFold(
                n_splits=actual_folds,
                shuffle=True,
                random_state=random_state,
            ),
            actual_folds,
            warning,
        )

    if task_type == "regression":
        actual_folds = min(requested_folds, len(y))
        if actual_folds < 2:
            raise ValueError("At least two observations are required.")
        warning = None
        if actual_folds < requested_folds:
            warning = "Cross-validation folds reduced from {} to {}.".format(
                requested_folds, actual_folds
            )

        return (
            KFold(
                n_splits=actual_folds,
                shuffle=True,
                random_state=random_state,
            ),
            actual_folds,
            warning,
        )

    raise ValueError("task_type must be 'classification' or 'regression'.")


def _split_internal_validation(
    X_train_outer: pd.DataFrame,
    y_train_outer: np.ndarray,
    task_type: str,
    validation_fraction: float,
    random_state: int,
):
    """
    Split the outer training fold into internal train/validation.

    The outer test fold is never used for early stopping.
    """
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5.")

    stratify = None
    if task_type == "classification":
        counts = pd.Series(y_train_outer).value_counts()
        # Stratification is used when it is feasible.
        if int(counts.min()) >= 2:
            stratify = y_train_outer

    try:
        return train_test_split(
            X_train_outer,
            y_train_outer,
            test_size=validation_fraction,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        # Very small folds can make stratification impossible.
        return train_test_split(
            X_train_outer,
            y_train_outer,
            test_size=validation_fraction,
            random_state=random_state,
            stratify=None,
        )


# ============================================================
# PyTorch helpers
# ============================================================

def _target_tensor(
    y: np.ndarray,
    task_type: str,
    number_of_classes: Optional[int],
) -> torch.Tensor:
    if task_type == "regression" or number_of_classes == 2:
        return torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    return torch.tensor(y, dtype=torch.long)


def _loader(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    number_of_classes: Optional[int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        _target_tensor(y, task_type, number_of_classes),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_loss_function(
    task_type: str,
    y_train: np.ndarray,
    number_of_classes: Optional[int],
    handle_class_imbalance: bool,
    device: torch.device,
) -> nn.Module:
    """Build MSE, weighted BCE, or weighted cross-entropy."""
    if task_type == "regression":
        return nn.MSELoss()

    if number_of_classes == 2:
        pos_weight = None
        if handle_class_imbalance:
            positives = int(np.sum(y_train == 1))
            negatives = int(np.sum(y_train == 0))
            if positives == 0 or negatives == 0:
                raise ValueError("Both classes must exist in the training data.")
            pos_weight = torch.tensor(
                [negatives / positives],
                dtype=torch.float32,
                device=device,
            )
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    weights = None
    if handle_class_imbalance:
        classes = np.unique(y_train)
        calculated = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        weights = torch.tensor(calculated, dtype=torch.float32, device=device)

    return nn.CrossEntropyLoss(weight=weights)


def build_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
):
    """Build a supported optimizer."""
    name = optimizer_name.lower()
    options = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    if name not in options:
        raise ValueError("Unsupported optimizer '{}'.".format(optimizer_name))
    return options[name](
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def _predictions_from_outputs(
    outputs: torch.Tensor,
    task_type: str,
    number_of_classes: Optional[int],
) -> np.ndarray:
    if task_type == "regression":
        return outputs.detach().cpu().numpy().reshape(-1)
    if number_of_classes == 2:
        return (
            (torch.sigmoid(outputs) >= 0.5)
            .long()
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
    return torch.argmax(outputs, dim=1).detach().cpu().numpy()


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
) -> Dict[str, float]:
    """Calculate the same core metrics used by machine_learning.py."""
    if task_type == "classification":
        return {
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    task_type: str,
    number_of_classes: Optional[int],
    device: torch.device,
):
    """Evaluate one loader without updating weights."""
    model.eval()
    total_loss = 0.0
    total_rows = 0
    targets_all = []
    predictions_all = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            loss = loss_function(outputs, targets)

            rows = features.shape[0]
            total_loss += float(loss.item()) * rows
            total_rows += rows

            targets_all.append(targets.detach().cpu().numpy().reshape(-1))
            predictions_all.append(
                _predictions_from_outputs(outputs, task_type, number_of_classes)
            )

    y_true = np.concatenate(targets_all)
    y_pred = np.concatenate(predictions_all)

    return {
        "loss": total_loss / total_rows,
        "metrics": calculate_metrics(y_true, y_pred, task_type),
        "targets": y_true.tolist(),
        "predictions": y_pred.tolist(),
    }


def _train_one_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    task_type: str,
    number_of_classes: Optional[int],
    output_size: int,
    hidden_layers: List[int],
    activations: Union[str, Sequence[str]],
    dropout: float,
    use_batch_norm: bool,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    handle_class_imbalance: bool,
    random_state: int,
    device: torch.device,
):
    """Train one network with validation-loss early stopping."""
    set_random_seed(random_state)

    model = FeedForwardNetwork(
        input_size=X_train.shape[1],
        output_size=output_size,
        hidden_layers=hidden_layers,
        activations=activations,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
    ).to(device)

    loss_function = build_loss_function(
        task_type,
        y_train,
        number_of_classes,
        handle_class_imbalance,
        device,
    )
    optimizer = build_optimizer(
        model,
        optimizer_name,
        learning_rate,
        weight_decay,
    )

    train_loader = _loader(
        X_train, y_train, task_type, number_of_classes, batch_size, True
    )
    validation_loader = _loader(
        X_validation, y_validation, task_type, number_of_classes, batch_size, False
    )

    best_loss = float("inf")
    best_epoch = 1
    best_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            rows = features.shape[0]
            total_loss += float(loss.item()) * rows
            total_rows += rows

        validation = _evaluate(
            model,
            validation_loader,
            loss_function,
            task_type,
            number_of_classes,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "training_loss": total_loss / total_rows,
                "validation_loss": validation["loss"],
            }
        )

        if validation["loss"] < best_loss - min_delta:
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)

    return model, loss_function, best_epoch, history


# ============================================================
# CV evaluation
# ============================================================

def _summarize_fold_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
    """Summarize every metric across folds."""
    result = {}
    for metric_name in fold_metrics[0]:
        scores = [float(row[metric_name]) for row in fold_metrics]
        result[metric_name] = {
            "mean": float(np.mean(scores)),
            "standard_deviation": float(np.std(scores, ddof=0)),
            "minimum": float(np.min(scores)),
            "maximum": float(np.max(scores)),
            "fold_scores": scores,
        }
    return result


def cross_validate_feed_forward_network(
    prepared: Dict[str, Any],
    requested_folds: int = 10,
    validation_fraction: float = 0.15,
    hidden_layers: Optional[List[int]] = None,
    activations: Union[str, Sequence[str]] = "relu",
    dropout: float = 0.0,
    use_batch_norm: bool = False,
    optimizer_name: str = "adam",
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    max_epochs: int = 200,
    patience: int = 15,
    min_delta: float = 0.0,
    handle_class_imbalance: bool = True,
    random_state: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Outer K-fold CV + internal validation split for early stopping.

    Preprocessing is fitted only on each fold's internal training
    data, so neither validation nor outer test data leaks into it.
    """
    hidden_layers = [] if hidden_layers is None else hidden_layers
    device = torch.device("cpu") if device is None else device

    cv, actual_folds, warning = create_neural_cross_validator(
        prepared["y"],
        prepared["task_type"],
        requested_folds,
        random_state,
    )

    X = prepared["X"]
    y = prepared["y"]

    fold_metrics = []
    fold_results = []
    best_epochs = []

    split_iterator = cv.split(X, y if prepared["task_type"] == "classification" else None)

    for fold_number, (outer_train_idx, outer_test_idx) in enumerate(
        split_iterator, start=1
    ):
        X_outer_train = X.iloc[outer_train_idx]
        X_outer_test = X.iloc[outer_test_idx]
        y_outer_train = y[outer_train_idx]
        y_outer_test = y[outer_test_idx]

        (
            X_internal_train,
            X_internal_validation,
            y_internal_train,
            y_internal_validation,
        ) = _split_internal_validation(
            X_outer_train,
            y_outer_train,
            prepared["task_type"],
            validation_fraction,
            random_state + fold_number,
        )

        preprocessor = build_neural_network_preprocessor(
            prepared["numeric_columns"],
            prepared["categorical_columns"],
        )

        X_train_t = np.asarray(
            preprocessor.fit_transform(X_internal_train), dtype=np.float32
        )
        X_validation_t = np.asarray(
            preprocessor.transform(X_internal_validation), dtype=np.float32
        )
        X_test_t = np.asarray(
            preprocessor.transform(X_outer_test), dtype=np.float32
        )

        model, loss_function, best_epoch, history = _train_one_network(
            X_train_t,
            y_internal_train,
            X_validation_t,
            y_internal_validation,
            prepared["task_type"],
            prepared["number_of_classes"],
            prepared["output_size"],
            hidden_layers,
            activations,
            dropout,
            use_batch_norm,
            optimizer_name,
            learning_rate,
            weight_decay,
            batch_size,
            max_epochs,
            patience,
            min_delta,
            handle_class_imbalance,
            random_state + fold_number,
            device,
        )

        test_loader = _loader(
            X_test_t,
            y_outer_test,
            prepared["task_type"],
            prepared["number_of_classes"],
            batch_size,
            False,
        )

        test_result = _evaluate(
            model,
            test_loader,
            loss_function,
            prepared["task_type"],
            prepared["number_of_classes"],
            device,
        )

        fold_metrics.append(test_result["metrics"])
        best_epochs.append(best_epoch)
        fold_results.append(
            {
                "fold": fold_number,
                "best_epoch": best_epoch,
                "test_metrics": test_result["metrics"],
                "training_history": history,
            }
        )

    return {
        "requested_folds": requested_folds,
        "actual_folds": actual_folds,
        "warning": warning,
        "metrics": _summarize_fold_metrics(fold_metrics),
        "folds": fold_results,
        "best_epochs": best_epochs,
        "median_best_epoch": max(1, int(np.median(best_epochs))),
    }


# ============================================================
# Final model
# ============================================================

def fit_final_feed_forward_network(
    prepared: Dict[str, Any],
    final_epochs: int,
    hidden_layers: Optional[List[int]] = None,
    activations: Union[str, Sequence[str]] = "relu",
    dropout: float = 0.0,
    use_batch_norm: bool = False,
    optimizer_name: str = "adam",
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    handle_class_imbalance: bool = True,
    random_state: int = 42,
    device: Optional[torch.device] = None,
):
    """
    Fit preprocessing and the final network on all valid data.

    The epoch count comes from the median best epoch observed
    during cross-validation, so no permanent holdout is needed.
    """
    hidden_layers = [] if hidden_layers is None else hidden_layers
    device = torch.device("cpu") if device is None else device
    set_random_seed(random_state)

    preprocessor = build_neural_network_preprocessor(
        prepared["numeric_columns"],
        prepared["categorical_columns"],
    )
    X_all = np.asarray(
        preprocessor.fit_transform(prepared["X"]),
        dtype=np.float32,
    )
    y_all = prepared["y"]

    model = FeedForwardNetwork(
        input_size=X_all.shape[1],
        output_size=prepared["output_size"],
        hidden_layers=hidden_layers,
        activations=activations,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
    ).to(device)

    loss_function = build_loss_function(
        prepared["task_type"],
        y_all,
        prepared["number_of_classes"],
        handle_class_imbalance,
        device,
    )
    optimizer = build_optimizer(
        model,
        optimizer_name,
        learning_rate,
        weight_decay,
    )
    loader = _loader(
        X_all,
        y_all,
        prepared["task_type"],
        prepared["number_of_classes"],
        batch_size,
        True,
    )

    history = []

    for epoch in range(1, final_epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0

        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            rows = features.shape[0]
            total_loss += float(loss.item()) * rows
            total_rows += rows

        history.append(
            {
                "epoch": epoch,
                "training_loss": total_loss / total_rows,
            }
        )

    feature_names = (
        preprocessor.get_feature_names_out().astype(str).tolist()
    )

    return model, preprocessor, feature_names, history


# ============================================================
# Weights and prediction
# ============================================================

def extract_network_weights(
    model: FeedForwardNetwork,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Return every Linear layer's weights.

    Direct feature weights are separately returned only when
    hidden_layers=[].
    """
    linear_layers = [
        layer for layer in model.network if isinstance(layer, nn.Linear)
    ]

    layers = []
    for index, layer in enumerate(linear_layers):
        layers.append(
            {
                "layer_index": index,
                "weights": layer.weight.detach().cpu().numpy().tolist(),
                "bias": layer.bias.detach().cpu().numpy().tolist(),
            }
        )

    is_direct = len(model.hidden_layers) == 0
    direct = None

    if is_direct:
        weights = linear_layers[-1].weight.detach().cpu().numpy()
        bias = linear_layers[-1].bias.detach().cpu().numpy()

        if weights.shape[0] == 1:
            ranked = [
                {
                    "feature": feature,
                    "weight": float(weight),
                    "absolute_weight": float(abs(weight)),
                }
                for feature, weight in zip(feature_names, weights[0])
            ]
            ranked.sort(key=lambda x: x["absolute_weight"], reverse=True)
            direct = {
                "output": (
                    class_names[1]
                    if class_names is not None and len(class_names) == 2
                    else "output"
                ),
                "weights": ranked,
                "bias": float(bias[0]),
            }
        else:
            direct = {}
            names = class_names or [str(i) for i in range(weights.shape[0])]
            for i, name in enumerate(names):
                ranked = [
                    {
                        "feature": feature,
                        "weight": float(weight),
                        "absolute_weight": float(abs(weight)),
                    }
                    for feature, weight in zip(feature_names, weights[i])
                ]
                ranked.sort(key=lambda x: x["absolute_weight"], reverse=True)
                direct[name] = {
                    "weights": ranked,
                    "bias": float(bias[i]),
                }

    return {
        "direct_input_output_weights": is_direct,
        "direct_weights": direct,
        "layers": layers,
    }


def predict_feed_forward_network(
    result: Dict[str, Any],
    new_data: pd.DataFrame,
) -> Dict[str, Any]:
    """Predict new rows using the final fitted preprocessor/model."""
    required = result["feature_columns"]
    missing = [c for c in required if c not in new_data.columns]
    if missing:
        raise ValueError("New data is missing columns: {}.".format(missing))

    X = np.asarray(
        result["preprocessor"].transform(new_data[required]),
        dtype=np.float32,
    )
    model = result["model"]
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X, dtype=torch.float32, device=device))

    if result["task_type"] == "regression":
        return {
            "predictions": outputs.cpu().numpy().reshape(-1).astype(float).tolist()
        }

    if result["number_of_classes"] == 2:
        probabilities = torch.sigmoid(outputs).cpu().numpy().reshape(-1)
        encoded = (probabilities >= 0.5).astype(int)
        labels = result["label_encoder"].inverse_transform(encoded)
        return {
            "predictions": labels.tolist(),
            "positive_class": result["class_names"][1],
            "positive_probabilities": probabilities.astype(float).tolist(),
        }

    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
    encoded = np.argmax(probabilities, axis=1)
    labels = result["label_encoder"].inverse_transform(encoded)
    return {
        "predictions": labels.tolist(),
        "probabilities": probabilities.astype(float).tolist(),
        "class_names": result["class_names"],
    }


# ============================================================
# Main orchestration
# ============================================================

def train_feed_forward_network(
    dataframe: pd.DataFrame,
    profile: Dict[str, Any],
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    task_type: Optional[str] = None,
    hidden_layers: Optional[List[int]] = None,
    activations: Union[str, Sequence[str]] = "relu",
    dropout: float = 0.0,
    use_batch_norm: bool = False,
    optimizer_name: str = "adam",
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    max_epochs: int = 200,
    patience: int = 15,
    min_delta: float = 0.0,
    cv_folds: int = 10,
    validation_fraction: float = 0.15,
    handle_class_imbalance: bool = True,
    random_state: int = 42,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full flow:

    data -> outer K-fold CV -> internal validation/early stopping
         -> summarize test-fold metrics
         -> median best epoch
         -> final fit on all data
         -> weights + fitted model/preprocessor
    """
    hidden_layers = [] if hidden_layers is None else hidden_layers

    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")

    prepared = prepare_neural_network_data(
        dataframe,
        profile,
        target_column,
        feature_columns,
        task_type,
    )

    cv_result = cross_validate_feed_forward_network(
        prepared=prepared,
        requested_folds=cv_folds,
        validation_fraction=validation_fraction,
        hidden_layers=hidden_layers,
        activations=activations,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        handle_class_imbalance=handle_class_imbalance,
        random_state=random_state,
        device=device,
    )

    final_epochs = cv_result["median_best_epoch"]

    model, preprocessor, feature_names, final_history = (
        fit_final_feed_forward_network(
            prepared=prepared,
            final_epochs=final_epochs,
            hidden_layers=hidden_layers,
            activations=activations,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            handle_class_imbalance=handle_class_imbalance,
            random_state=random_state,
            device=device,
        )
    )

    weights = extract_network_weights(
        model,
        feature_names,
        prepared["class_names"],
    )

    primary_metric = (
        "f1_macro"
        if prepared["task_type"] == "classification"
        else "r2"
    )

    return {
        "task_type": prepared["task_type"],
        "task_detection": prepared["task_detection"],
        "target_column": target_column,
        "feature_columns": prepared["feature_columns"],
        "numeric_columns": prepared["numeric_columns"],
        "categorical_columns": prepared["categorical_columns"],
        "feature_names": feature_names,
        "class_names": prepared["class_names"],
        "number_of_classes": prepared["number_of_classes"],
        "architecture": {
            "input_size": len(feature_names),
            "hidden_layers": list(hidden_layers),
            "activations": resolve_activations(hidden_layers, activations),
            "output_size": prepared["output_size"],
            "dropout": dropout,
            "use_batch_norm": use_batch_norm,
        },
        "requested_cv_folds": cv_result["requested_folds"],
        "actual_cv_folds": cv_result["actual_folds"],
        "cv_warning": cv_result["warning"],
        "cross_validation": cv_result,
        "final_epochs": final_epochs,
        "primary_metric": primary_metric,
        "primary_score": cv_result["metrics"][primary_metric]["mean"],
        "weights": weights,
        "model": model,
        "preprocessor": preprocessor,
        "label_encoder": prepared["label_encoder"],
        "final_training_history": final_history,
        "rows_used": prepared["rows_used"],
        "removed_target_rows": prepared["removed_target_rows"],
        "device": str(device),
    }
