from langchain.tools import tool
from typing import Optional, Literal
from pydantic import BaseModel, Field

####################################
# Data Visualization Tools
####################################

class CreateVisualizationInput(BaseModel):
    data_description: str = Field(description="Description of the data to visualize")
    chart_type: Literal["line", "scatter", "histogram", "bar", "box", "heatmap", "confusion_matrix", "roc_curve", "learning_curve"] = Field(
        description="Type of visualization to create"
    )
    x_column: Optional[str] = Field(default=None, description="Column for x-axis")
    y_column: Optional[str] = Field(default=None, description="Column for y-axis")
    output_path: str = Field(description="Path to save the visualization (PNG, SVG, or PDF)")
    
@tool(args_schema=CreateVisualizationInput)
def create_visualization(
    data_description: str,
    chart_type: str,
    output_path: str,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None
) -> dict:
    """
    Creates professional visualizations for data analysis and model evaluation.
    
    Supported chart types:
    - line: Time series, training curves, metric evolution
    - scatter: Feature relationships, clustering results
    - histogram: Feature distributions
    - bar: Categorical comparisons, feature importance
    - box: Distribution comparisons, outlier detection
    - heatmap: Correlation matrices, confusion matrices
    - confusion_matrix: Classification performance
    - roc_curve: Binary classification performance
    - learning_curve: Model performance vs. training size
    
    Returns path to saved visualization.
    """
    return {
        "note": "Use execute_python_code() to create visualizations with matplotlib/seaborn:",
        "example_code": f"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data
df = pd.read_csv('your_data.csv')

# Create {chart_type} plot
plt.figure(figsize=(10, 6))
{_get_plot_code(chart_type, x_column, y_column)}

plt.savefig('{output_path}', dpi=300, bbox_inches='tight')
print(f"Visualization saved to {output_path}")
""",
        "data_description": data_description,
        "chart_type": chart_type,
        "output_path": output_path
    }


def _get_plot_code(chart_type: str, x_col: Optional[str], y_col: Optional[str]) -> str:
    """Helper to generate chart-specific code snippets."""
    if chart_type == "line":
        return f"plt.plot(df['{x_col}'], df['{y_col}'])\nplt.xlabel('{x_col}')\nplt.ylabel('{y_col}')"
    elif chart_type == "scatter":
        return f"plt.scatter(df['{x_col}'], df['{y_col}'], alpha=0.5)\nplt.xlabel('{x_col}')\nplt.ylabel('{y_col}')"
    elif chart_type == "histogram":
        return f"plt.hist(df['{x_col}'], bins=30, edgecolor='black')\nplt.xlabel('{x_col}')\nplt.ylabel('Frequency')"
    elif chart_type == "bar":
        return f"df.groupby('{x_col}')['{y_col}'].mean().plot(kind='bar')\nplt.xlabel('{x_col}')\nplt.ylabel('{y_col}')"
    elif chart_type == "box":
        return f"sns.boxplot(x='{x_col}', y='{y_col}', data=df)"
    elif chart_type == "heatmap":
        return "sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)"
    elif chart_type == "confusion_matrix":
        return """from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm).plot()"""
    elif chart_type == "roc_curve":
        return """from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')"""
    elif chart_type == "learning_curve":
        return """from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = learning_curve(model, X, y, cv=5)
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation score')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.legend()"""
    else:
        return f"# Create {chart_type} plot here"


class PlotTrainingHistoryInput(BaseModel):
    history_file: str = Field(description="Path to training history file (CSV or JSON)")
    metrics: list[str] = Field(default=["loss", "accuracy"], description="Metrics to plot")
    output_path: str = Field(description="Path to save the plot")
    
@tool(args_schema=PlotTrainingHistoryInput)
def plot_training_history(history_file: str, metrics: list[str], output_path: str) -> dict:
    """
    Creates plots of training history showing loss, accuracy, and other metrics over epochs.
    Displays both training and validation curves to identify overfitting.
    """
    return {
        "note": "Use execute_python_code() to plot training history:",
        "example_code": f"""
import pandas as pd
import matplotlib.pyplot as plt

# Load training history
history = pd.read_csv('{history_file}')

# Create subplots for each metric
fig, axes = plt.subplots(1, len({metrics}), figsize=(15, 5))
if len({metrics}) == 1:
    axes = [axes]

for idx, metric in enumerate({metrics}):
    ax = axes[idx]
    ax.plot(history['epoch'], history[f'train_{{metric}}'], label=f'Training {{metric}}')
    ax.plot(history['epoch'], history[f'val_{{metric}}'], label=f'Validation {{metric}}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{{metric.capitalize()}} vs. Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('{output_path}', dpi=300, bbox_inches='tight')
print(f"Training history plot saved to {output_path}")
""",
        "history_file": history_file,
        "metrics": metrics,
        "output_path": output_path
    }
