from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

import matplotlib
matplotlib.use("Agg")  # Ensure headless environments can render figures
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from langchain.tools import tool
from pydantic import BaseModel, Field

####################################
# Data Visualization Tools
####################################

sns.set_theme(style="whitegrid")


def _load_dataframe(data_path: str) -> pd.DataFrame:
    path = Path(data_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file extension: {suffix}")


class CreateVisualizationInput(BaseModel):
    data_path: str = Field(..., description="Path to dataset (CSV/JSON/Excel/Parquet)")
    chart_type: Literal[
        "line",
        "scatter",
        "histogram",
        "bar",
        "box",
        "heatmap",
        "confusion_matrix",
        "roc_curve",
        "learning_curve",
    ] = Field(..., description="Type of visualization to create")
    x_column: Optional[str] = Field(default=None, description="X-axis column")
    y_column: Optional[str] = Field(default=None, description="Y-axis column / values")
    hue_column: Optional[str] = Field(default=None, description="Grouping column for categorical hue")
    output_path: str = Field(..., description="File path to save the generated plot (PNG/SVG/PDF)")


@tool(args_schema=CreateVisualizationInput)
def create_visualization(
    data_path: str,
    chart_type: str,
    output_path: str,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    hue_column: Optional[str] = None,
) -> dict:
    """Render a high-quality visualization and save it to disk."""
    try:
        df = _load_dataframe(data_path)
        output = Path(output_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "line":
            if not x_column or not y_column:
                raise ValueError("x_column and y_column are required for line charts")
            sns.lineplot(data=df, x=x_column, y=y_column, hue=hue_column, ax=ax)

        elif chart_type == "scatter":
            if not x_column or not y_column:
                raise ValueError("x_column and y_column are required for scatter plots")
            sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column, ax=ax)

        elif chart_type == "histogram":
            if not x_column:
                raise ValueError("x_column is required for histograms")
            sns.histplot(data=df, x=x_column, hue=hue_column, kde=True, ax=ax)

        elif chart_type == "bar":
            if not x_column or not y_column:
                raise ValueError("x_column and y_column are required for bar charts")
            sns.barplot(data=df, x=x_column, y=y_column, hue=hue_column, ax=ax)

        elif chart_type == "box":
            if not x_column or not y_column:
                raise ValueError("x_column and y_column are required for box plots")
            sns.boxplot(data=df, x=x_column, y=y_column, hue=hue_column, ax=ax)

        elif chart_type == "heatmap":
            corr = df.select_dtypes(include=["number"]).corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")

        elif chart_type == "confusion_matrix":
            if not x_column or not y_column:
                raise ValueError("x_column (true labels) and y_column (predicted labels) are required")
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

            cm = confusion_matrix(df[x_column], df[y_column])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, cmap="Blues")
            ax.set_title("Confusion Matrix")

        elif chart_type == "roc_curve":
            if not x_column or not y_column:
                raise ValueError("x_column (true labels) and y_column (probabilities/scores) are required")
            from sklearn.metrics import roc_curve, auc

            fpr, tpr, _ = roc_curve(df[x_column], df[y_column])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()

        elif chart_type == "learning_curve":
            if not all(col in df.columns for col in ["epoch", "train_metric", "val_metric"]):
                raise ValueError(
                    "Learning curve expects 'epoch', 'train_metric', and 'val_metric' columns"
                )
            ax.plot(df["epoch"], df["train_metric"], label="Training")
            ax.plot(df["epoch"], df["val_metric"], label="Validation")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Metric")
            ax.set_title("Learning Curve")
            ax.legend()

        else:  # pragma: no cover - defensive guard
            raise ValueError(f"Unsupported chart_type: {chart_type}")

        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return {
            "status": "success",
            "output_path": str(output.resolve()),
            "chart_type": chart_type,
            "rows_visualized": int(len(df)),
        }
    except Exception as exc:
        plt.close("all")
        return {"status": "error", "error": str(exc)}


class PlotTrainingHistoryInput(BaseModel):
    history_file: str = Field(..., description="CSV or JSON file containing training history")
    metrics: List[str] = Field(default_factory=lambda: ["loss", "accuracy"], description="Metrics to plot")
    output_path: str = Field(..., description="Path to save the plot")


@tool(args_schema=PlotTrainingHistoryInput)
def plot_training_history(
    history_file: str,
    metrics: List[str],
    output_path: str,
) -> dict:
    """Plot training/validation metrics over epochs."""
    try:
        path = Path(history_file).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"History file not found: {history_file}")

        if path.suffix.lower() == ".json":
            history = pd.read_json(path)
        else:
            history = pd.read_csv(path)

        if "epoch" not in history.columns:
            raise ValueError("History file must contain an 'epoch' column")

        output = Path(output_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            train_col = f"train_{metric}"
            val_col = f"val_{metric}"
            if train_col not in history.columns or val_col not in history.columns:
                raise ValueError(
                    f"History missing expected columns '{train_col}' and/or '{val_col}'"
                )
            ax.plot(history["epoch"], history[train_col], label=f"Train {metric}")
            ax.plot(history["epoch"], history[val_col], label=f"Validation {metric}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"{metric.capitalize()} vs Epoch")
            ax.legend()

        fig.tight_layout()
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return {
            "status": "success",
            "output_path": str(output.resolve()),
            "metrics": metrics,
            "epochs": int(history["epoch"].max()),
        }
    except Exception as exc:
        plt.close("all")
        return {"status": "error", "error": str(exc)}
