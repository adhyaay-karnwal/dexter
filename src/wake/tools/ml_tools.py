from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from joblib import dump, load
from langchain.tools import tool
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

####################################
# Machine Learning Operations Tools
####################################


def _load_dataframe(file_path: str) -> pd.DataFrame:
    path = Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file format: {suffix}")


def _build_preprocessor(df: pd.DataFrame, feature_columns: List[str]) -> ColumnTransformer:
    numeric_cols = [col for col in feature_columns if np.issubdtype(df[col].dtype, np.number)]
    categorical_cols = [col for col in feature_columns if col not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse=False, min_frequency=0.01),
            ),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_transformer, categorical_cols))

    if not transformers:
        raise ValueError("No valid feature columns detected for preprocessing.")

    return ColumnTransformer(transformers)


def _initialise_estimator(model_type: str, algorithm: str, hyperparameters: Optional[dict]) -> BaseEstimator:
    algo = algorithm.lower()
    hyperparameters = hyperparameters or {}

    if model_type == "classification":
        mapping = {
            "random_forest": RandomForestClassifier,
            "logistic_regression": LogisticRegression,
            "svm": lambda **kwargs: SVC(probability=True, **kwargs),
            "decision_tree": DecisionTreeClassifier,
            "mlp": MLPClassifier,
        }
        if algo not in mapping:
            raise ValueError(f"Unsupported classification algorithm: {algorithm}")
        return mapping[algo](**hyperparameters)

    if model_type == "regression":
        mapping = {
            "linear_regression": LinearRegression,
            "ridge": Ridge,
            "lasso": Lasso,
            "random_forest": RandomForestRegressor,
            "svm": SVR,
            "gradient_boosting": GradientBoostingRegressor,
            "mlp": MLPRegressor,
        }
        if algo not in mapping:
            raise ValueError(f"Unsupported regression algorithm: {algorithm}")
        return mapping[algo](**hyperparameters)

    if model_type == "clustering":
        mapping = {
            "kmeans": KMeans,
            "dbscan": DBSCAN,
            "hierarchical": AgglomerativeClustering,
        }
        if algo not in mapping:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
        return mapping[algo](**hyperparameters)

    if model_type == "neural_network":
        return MLPClassifier(**hyperparameters)

    raise ValueError(f"Unsupported model type: {model_type}")


def _ensure_model_dir(path: Optional[str], algorithm: str) -> Path:
    if path:
        output = Path(path).expanduser()
    else:
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        output = models_dir / f"{algorithm.lower()}_model.joblib"
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, Any]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None and y_proba.shape[1] == 2:
        try:
            from sklearn.metrics import roc_auc_score

            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        except Exception:  # pragma: no cover - safety
            pass
    return metrics


def _regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mae),
    }


def _feature_importance(estimator: BaseEstimator, feature_names: List[str]) -> Optional[Dict[str, float]]:
    try:
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            importances = np.ravel(estimator.coef_)
        else:
            return None
        return {name: float(val) for name, val in zip(feature_names, importances)}
    except Exception:  # pragma: no cover - defensive
        return None


class TrainModelInput(BaseModel):
    model_type: Literal["classification", "regression", "clustering", "neural_network"] = Field(
        ..., description="Type of ML model to train"
    )
    algorithm: str = Field(
        ..., description="Specific algorithm to use (e.g., 'random_forest', 'logistic_regression', 'kmeans', 'mlp')"
    )
    data_path: str = Field(..., description="Path to the dataset (CSV/JSON/Excel/Parquet)")
    target_column: Optional[str] = Field(
        default=None, description="Target column (required for supervised tasks)"
    )
    test_size: float = Field(default=0.2, description="Proportion of data reserved for evaluation")
    hyperparameters: Optional[dict] = Field(
        default=None, description="Dictionary of hyperparameters for the underlying estimator"
    )
    output_model_path: Optional[str] = Field(
        default=None, description="Where to persist the trained pipeline (joblib)"
    )
    random_state: Optional[int] = Field(default=42, description="Random seed for reproducibility")


@tool(args_schema=TrainModelInput)
def train_ml_model(
    model_type: str,
    algorithm: str,
    data_path: str,
    target_column: Optional[str] = None,
    test_size: float = 0.2,
    hyperparameters: Optional[dict] = None,
    output_model_path: Optional[str] = None,
    random_state: Optional[int] = 42,
) -> dict:
    """Train an ML model end-to-end using scikit-learn pipelines."""
    try:
        df = _load_dataframe(data_path)

        if model_type in {"classification", "regression", "neural_network"}:
            if target_column is None:
                raise ValueError("target_column is required for supervised learning")
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")

            X = df.drop(columns=[target_column])
            y = df[target_column]
            feature_columns = X.columns.tolist()
        elif model_type == "clustering":
            X = df.copy()
            feature_columns = X.columns.tolist()
            y = None
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        preprocessor = _build_preprocessor(df, feature_columns)
        estimator = _initialise_estimator(model_type, algorithm, hyperparameters)
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])

        if model_type in {"classification", "regression", "neural_network"}:
            stratify = y if model_type == "classification" and y.nunique() > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            if model_type == "classification" or model_type == "neural_network":
                y_proba = None
                if hasattr(pipeline.named_steps["model"], "predict_proba"):
                    y_proba = pipeline.predict_proba(X_test)
                metrics = _classification_metrics(y_test, y_pred, y_proba)
            else:
                metrics = _regression_metrics(y_test, y_pred)

            feature_importance = None
            if hasattr(pipeline.named_steps["model"], "feature_importances_") or hasattr(
                pipeline.named_steps["model"], "coef_"
            ):
                # Extract feature names after preprocessing (OneHotEncoder expands)
                final_feature_names = feature_columns
                try:
                    transformer = pipeline.named_steps["preprocess"]
                    final_feature_names = transformer.get_feature_names_out().tolist()
                except Exception:  # pragma: no cover - not all preprocessors support it
                    pass
                feature_importance = _feature_importance(pipeline.named_steps["model"], final_feature_names)
        else:
            pipeline.fit(X)
            cluster_labels = pipeline.named_steps["model"].labels_
            metrics = {}
            try:
                transformed = pipeline.named_steps["preprocess"].transform(X)
                if len(set(cluster_labels)) > 1:
                    metrics["silhouette_score"] = float(silhouette_score(transformed, cluster_labels))
            except Exception:  # pragma: no cover
                pass
            feature_importance = None

        output_path = _ensure_model_dir(output_model_path, algorithm)
        dump(pipeline, output_path)

        result = {
            "status": "success",
            "model_type": model_type,
            "algorithm": algorithm,
            "data_path": str(Path(data_path).resolve()),
            "model_path": str(output_path.resolve()),
            "metrics": metrics,
        }
        if feature_importance:
            result["feature_importance"] = feature_importance
        if model_type in {"classification", "regression", "neural_network"}:
            result["test_size"] = float(test_size)
        return result

    except Exception as exc:
        return {"status": "error", "error": str(exc)}


class HyperparameterTuningInput(BaseModel):
    model_type: Literal["classification", "regression"] = Field(..., description="Model family")
    algorithm: str = Field(..., description="Estimator to tune")
    data_path: str = Field(..., description="Path to the dataset")
    target_column: str = Field(..., description="Target column name")
    param_grid: dict = Field(..., description="Hyperparameter search space")
    search_method: Literal["grid", "random"] = Field(
        default="grid", description="Search strategy (grid search or random search)"
    )
    cv_folds: int = Field(default=5, description="Number of cross-validation folds")
    n_iter: Optional[int] = Field(
        default=None, description="Number of iterations for random search (ignored for grid search)"
    )
    random_state: Optional[int] = Field(default=42, description="Random seed")
    output_model_path: Optional[str] = Field(default=None, description="Where to save the tuned model")


@tool(args_schema=HyperparameterTuningInput)
def tune_hyperparameters(
    model_type: str,
    algorithm: str,
    data_path: str,
    target_column: str,
    param_grid: dict,
    search_method: str = "grid",
    cv_folds: int = 5,
    n_iter: Optional[int] = None,
    random_state: Optional[int] = 42,
    output_model_path: Optional[str] = None,
) -> dict:
    """Perform hyperparameter optimisation using scikit-learn search utilities."""
    try:
        df = _load_dataframe(data_path)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        X = df.drop(columns=[target_column])
        y = df[target_column]
        feature_columns = X.columns.tolist()

        preprocessor = _build_preprocessor(df, feature_columns)
        estimator = _initialise_estimator(model_type, algorithm, None)
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])

        search_params = {f"model__{key}": value for key, value in param_grid.items()}

        if search_method == "grid":
            search = GridSearchCV(
                pipeline,
                search_params,
                cv=cv_folds,
                n_jobs=-1,
                scoring="accuracy" if model_type == "classification" else "r2",
            )
        elif search_method == "random":
            if n_iter is None:
                n_iter = 20
            search = RandomizedSearchCV(
                pipeline,
                search_params,
                n_iter=n_iter,
                cv=cv_folds,
                random_state=random_state,
                n_jobs=-1,
                scoring="accuracy" if model_type == "classification" else "r2",
            )
        else:
            raise ValueError(f"Unsupported search_method: {search_method}")

        search.fit(X, y)

        output_path = _ensure_model_dir(output_model_path, algorithm + "_tuned")
        dump(search.best_estimator_, output_path)

        return {
            "status": "success",
            "algorithm": algorithm,
            "search_method": search_method,
            "best_params": {key.replace("model__", ""): value for key, value in search.best_params_.items()},
            "best_score": float(search.best_score_),
            "model_path": str(output_path.resolve()),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


class ModelEvaluationInput(BaseModel):
    model_path: str = Field(..., description="Path to a trained model (joblib pipeline)")
    test_data_path: str = Field(..., description="Dataset to evaluate against")
    target_column: str = Field(..., description="Target column for supervised tasks")
    problem_type: Literal["classification", "regression"] = Field(..., description="Type of ML problem")


@tool(args_schema=ModelEvaluationInput)
def evaluate_model(
    model_path: str,
    test_data_path: str,
    target_column: str,
    problem_type: str,
) -> dict:
    """Evaluate a persisted pipeline on new data and compute metrics."""
    try:
        model = load(Path(model_path).expanduser())
        df = _load_dataframe(test_data_path)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not in dataset")

        X = df.drop(columns=[target_column])
        y = df[target_column]
        y_pred = model.predict(X)

        if problem_type == "classification":
            y_proba = None
            if hasattr(model.named_steps["model"], "predict_proba"):
                y_proba = model.predict_proba(X)
            metrics = _classification_metrics(y, y_pred, y_proba)
        elif problem_type == "regression":
            metrics = _regression_metrics(y, y_pred)
        else:
            raise ValueError("problem_type must be 'classification' or 'regression'")

        return {
            "status": "success",
            "model_path": str(Path(model_path).resolve()),
            "data_path": str(Path(test_data_path).resolve()),
            "metrics": metrics,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


class NeuralNetworkInput(BaseModel):
    architecture: Literal["mlp", "cnn", "rnn", "lstm", "transformer"] = Field(
        ..., description="Desired neural network architecture"
    )
    input_shape: List[int] = Field(..., description="Input tensor shape")
    output_size: int = Field(..., description="Number of output units")
    hidden_layers: List[int] = Field(default_factory=lambda: [128, 64], description="Hidden layer sizes")
    activation: str = Field(default="relu", description="Activation function")
    optimizer: str = Field(default="adam", description="Optimiser suggestion")
    learning_rate: float = Field(default=0.001, description="Suggested learning rate")


@tool(args_schema=NeuralNetworkInput)
def build_neural_network(
    architecture: str,
    input_shape: List[int],
    output_size: int,
    hidden_layers: List[int],
    activation: str = "relu",
    optimizer: str = "adam",
    learning_rate: float = 0.001,
) -> dict:
    """Generate a neural-network blueprint and training recommendations."""
    depth = len(hidden_layers)
    return {
        "architecture": architecture,
        "input_shape": input_shape,
        "output_size": output_size,
        "hidden_layers": hidden_layers,
        "activation": activation,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "recommendations": [
            "Use wake.tools.terminal_tools.execute_python_code to instantiate and train the network",
            "Set deterministic seeds for reproducibility",
            "Monitor loss/accuracy with observe_training_process or observe_process",
            "Enable checkpointing and early stopping where possible",
        ],
        "sample_code": f"""
import torch
import torch.nn as nn

class WakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_features = {input_shape[0] if input_shape else 0}
        for hidden in {hidden_layers}:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.{activation.capitalize()}())
            in_features = hidden
        layers.append(nn.Linear(in_features, {output_size}))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

model = WakeNet()
print(model)
optimizer = torch.optim.{optimizer.capitalize()}(model.parameters(), lr={learning_rate})
criterion = nn.CrossEntropyLoss()
        """.strip(),
    }


class TrainNeuralNetworkInput(BaseModel):
    model_description: str = Field(..., description="Narrative description of the architecture to train")
    train_data_path: str = Field(..., description="Training dataset path")
    val_data_path: Optional[str] = Field(default=None, description="Validation dataset path")
    epochs: int = Field(default=10, description="Number of training epochs")
    batch_size: int = Field(default=32, description="Batch size")
    framework: Literal["pytorch", "tensorflow"] = Field(
        default="pytorch", description="Preferred deep learning framework"
    )
    device: Literal["auto", "cpu", "cuda"] = Field(default="auto", description="Device selection policy")


@tool(args_schema=TrainNeuralNetworkInput)
def train_neural_network(
    model_description: str,
    train_data_path: str,
    val_data_path: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 32,
    framework: str = "pytorch",
    device: str = "auto",
) -> dict:
    """
    Provide a managed training plan for neural networks.

    Wake generates a reproducible training script that can be executed via the
    ``execute_python_code`` tool. This keeps the training logic flexible while
    enabling the agent to orchestrate datasets, monitoring, and optimisation.
    """
    return {
        "model_description": model_description,
        "framework": framework,
        "epochs": epochs,
        "batch_size": batch_size,
        "device_policy": device,
        "train_data": train_data_path,
        "val_data": val_data_path,
        "instructions": [
            "1. Load and preprocess the dataset (use pandas/torchvision/datasets).",
            "2. Instantiate the network as described in model_description.",
            "3. Use observe_process or observe_training_process to monitor long runs.",
            "4. Log metrics to CSV for plot_training_history to consume.",
            "5. Save checkpoints with timestamps for reproducibility.",
        ],
        "template_code": f"""
# This template assumes PyTorch and tabular CSV data. Adjust as needed.
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === Load data ===
raw = pd.read_csv('{train_data_path}')
X = raw.drop(columns=['target']).values  # update target column name
y = raw['target'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

train_ds = torch.utils.data.TensorDataset(X_train, y_train)
val_ds = torch.utils.data.TensorDataset(X_val, y_val)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size={batch_size}, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size={batch_size})

device = '{device}'
if device == 'auto':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = WakeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

history = []
for epoch in range(1, {epochs} + 1):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            val_loss += criterion(preds, yb).item() * xb.size(0)
            correct += (preds.argmax(dim=1) == yb).sum().item()
    val_loss /= len(val_loader.dataset)
    val_acc = correct / len(val_loader.dataset)

    history.append({'epoch': epoch, 'train_loss': epoch_loss, 'val_loss': val_loss, 'val_accuracy': val_acc})
    print(f"Epoch {epoch}: train_loss={epoch_loss:.4f} val_loss={val_loss:.4f} val_accuracy={val_acc:.4f}")

pd.DataFrame(history).to_csv('training_history.csv', index=False)
torch.save(model.state_dict(), f"wake_model_{{int(time.time())}}.pt")
        """.strip(),
    }


class FeatureEngineeringInput(BaseModel):
    data_path: str = Field(..., description="Dataset path")
    operations: List[Literal["polynomial", "interaction", "binning", "scaling", "pca", "feature_selection"]] = Field(
        ..., description="Feature engineering transformations to apply"
    )
    target_column: Optional[str] = Field(
        default=None, description="Target column name (required for feature_selection)"
    )
    output_path: Optional[str] = Field(
        default=None, description="Where to persist the transformed dataset"
    )


@tool(args_schema=FeatureEngineeringInput)
def engineer_features(
    data_path: str,
    operations: List[str],
    target_column: Optional[str] = None,
    output_path: Optional[str] = None,
) -> dict:
    """Perform configurable feature-engineering operations on a dataset."""
    try:
        df = _load_dataframe(data_path).copy()
        original_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        transformed_info: List[str] = []

        if "polynomial" in operations:
            if not numeric_cols:
                transformed_info.append("Polynomial features skipped (no numeric columns)")
            else:
                poly = PolynomialFeatures(degree=2, include_bias=False)
                poly_features = poly.fit_transform(df[numeric_cols])
                poly_names = poly.get_feature_names_out(numeric_cols)
                poly_df = pd.DataFrame(poly_features, columns=poly_names)
                df = pd.concat([df.drop(columns=numeric_cols), poly_df], axis=1)
                transformed_info.append(
                    f"Generated polynomial features (degree=2) for {len(numeric_cols)} numeric columns"
                )
                numeric_cols = poly_names.tolist()

        if "interaction" in operations and numeric_cols:
            # Interaction terms already covered by degree=2 polynomial features; note accordingly
            transformed_info.append("Interaction terms available via polynomial features (degree=2)")

        if "binning" in operations and numeric_cols:
            from sklearn.preprocessing import KBinsDiscretizer

            discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
            binned = discretizer.fit_transform(df[numeric_cols])
            bin_cols = [f"{col}_bin" for col in numeric_cols]
            for idx, col in enumerate(bin_cols):
                df[col] = binned[:, idx]
            transformed_info.append("Added quantile-based bin features for numeric columns")

        if "scaling" in operations and numeric_cols:
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            transformed_info.append("Applied Min-Max scaling to numeric features")

        if "pca" in operations and numeric_cols:
            n_components = min(3, len(numeric_cols))
            if n_components >= 1:
                pca = PCA(n_components=n_components)
                components = pca.fit_transform(df[numeric_cols])
                for idx in range(n_components):
                    df[f"pca_component_{idx+1}"] = components[:, idx]
                explained = pca.explained_variance_ratio_.cumsum().tolist()
                transformed_info.append(
                    f"Added {n_components} PCA components covering {explained[-1]*100:.2f}% variance"
                )

        if "feature_selection" in operations:
            if target_column is None:
                transformed_info.append("Feature selection skipped (target_column not provided)")
            elif target_column not in df.columns:
                transformed_info.append(
                    f"Feature selection skipped (target column '{target_column}' missing)"
                )
            else:
                if df[target_column].nunique() <= 20:
                    selector = SelectKBest(score_func=f_classif, k=min(20, len(numeric_cols)))
                else:
                    selector = SelectKBest(score_func=f_regression, k=min(20, len(numeric_cols)))
                selector.fit(df[numeric_cols], df[target_column])
                scores = selector.scores_
                selected = [numeric_cols[i] for i in np.argsort(scores)[::-1][: selector.k]]
                transformed_info.append(
                    "Top features by SelectKBest: "
                    + ", ".join(f"{name} ({scores[idx]:.2f})" for idx, name in zip(np.argsort(scores)[::-1][: selector.k], selected))
                )

        output = None
        if output_path:
            output = Path(output_path).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.suffix.lower() == ".parquet":
                df.to_parquet(output, index=False)
            else:
                df.to_csv(output, index=False)

        return {
            "status": "success",
            "original_columns": original_cols,
            "transformed_columns": df.columns.tolist(),
            "operations": transformed_info,
            "output_path": str(output.resolve()) if output else None,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
