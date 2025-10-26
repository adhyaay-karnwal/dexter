from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from langchain.tools import tool
from pydantic import BaseModel, Field

####################################
# Data Loading and Analysis Tools
####################################


def _load_dataframe(file_path: str) -> pd.DataFrame:
    """Load a dataframe from common analytics file formats."""
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


def _safe_to_python(value):
    """Convert numpy / pandas types to native Python objects."""
    if isinstance(value, (np.generic, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, (pd.Series, np.ndarray)):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value


class LoadDataInput(BaseModel):
    file_path: str = Field(..., description="Path to the data file to load. Supports CSV, JSON, Excel, and Parquet formats.")
    file_type: Optional[Literal["csv", "json", "excel", "parquet", "auto"]] = Field(
        default="auto",
        description="Type of file to load. Use 'auto' to automatically detect based on file extension."
    )


@tool(args_schema=LoadDataInput)
def load_dataset(file_path: str, file_type: str = "auto") -> dict:
    """
    Load a dataset and return a high-level summary.

    The tool inspects the dataset, computes shape information, dtype breakdown,
    missing value counts, and previews the first few rows so the agent can
    quickly reason about data suitability for downstream ML tasks.
    """
    try:
        # Auto-detect file type if requested (validation handled in _load_dataframe)
        if file_type != "auto":
            suffix = Path(file_path).suffix.lower()
            mapping = {"csv": ".csv", "json": ".json", "excel": ".xlsx", "parquet": ".parquet"}
            expected_suffix = mapping.get(file_type)
            if expected_suffix and suffix != expected_suffix:
                return {"error": f"File extension {suffix} does not match requested type '{file_type}'."}

        df = _load_dataframe(file_path)
        summary = {
            "status": "success",
            "file_path": str(Path(file_path).resolve()),
            "shape": list(df.shape),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": {col: int(count) for col, count in df.isnull().sum().items()},
            "preview": df.head(5).to_dict(orient="records"),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 ** 2),
        }
        return summary
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


class StatisticsInput(BaseModel):
    file_path: str = Field(..., description="Path to the dataset to analyse")
    columns: Optional[List[str]] = Field(
        default=None,
        description="Specific columns to analyse. If omitted, all numeric columns are used.",
    )
    include_correlation: bool = Field(
        default=True, description="Include the correlation matrix for numeric columns."
    )
    include_distributions: bool = Field(
        default=True, description="Include distribution analysis such as skewness and kurtosis."
    )


@tool(args_schema=StatisticsInput)
def compute_statistics(
    file_path: str,
    columns: Optional[List[str]] = None,
    include_correlation: bool = True,
    include_distributions: bool = True,
) -> dict:
    """
    Compute descriptive statistics for a dataset.

    The tool returns summary statistics, missing-value diagnostics, optional
    correlation matrices, and distribution insights so downstream planning can
    make informed choices about preprocessing and model selection.
    """
    try:
        df = _load_dataframe(file_path)

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                return {"status": "error", "error": f"Columns not found: {missing_cols}"}

        if not columns:
            return {"status": "error", "error": "No numeric columns available for statistical analysis."}

        subset = df[columns]
        summary = subset.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).transpose()

        result = {
            "status": "success",
            "file_path": str(Path(file_path).resolve()),
            "row_count": int(df.shape[0]),
            "columns_analyzed": columns,
            "summary": {col: {stat: _safe_to_python(val) for stat, val in stats.items()} for col, stats in summary.iterrows()},
            "missing_values": {col: int(df[col].isnull().sum()) for col in columns},
        }

        if include_distributions:
            result["skewness"] = {col: _safe_to_python(subset[col].skew()) for col in columns}
            result["kurtosis"] = {col: _safe_to_python(subset[col].kurtosis()) for col in columns}

        if include_correlation and len(columns) > 1:
            corr = subset.corr().replace({np.nan: None})
            result["correlation_matrix"] = {row: {col: _safe_to_python(val) for col, val in corr_row.items()} for row, corr_row in corr.iterrows()}

        return result
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


class ClassifyDataInput(BaseModel):
    file_path: str = Field(..., description="Path to the dataset to classify")
    target_column: Optional[str] = Field(
        default=None,
        description="Name of the target column if one exists. Helps determine supervised task type.",
    )
    time_column: Optional[str] = Field(
        default=None,
        description="Column containing temporal information for potential time-series problems.",
    )


@tool(args_schema=ClassifyDataInput)
def classify_dataset_type(
    file_path: str,
    target_column: Optional[str] = None,
    time_column: Optional[str] = None,
) -> dict:
    """
    Analyse a dataset and recommend machine-learning problem framing.

    The tool inspects column types, unique value counts, and temporal
    characteristics to determine whether the dataset is best approached as a
    classification, regression, time-series, clustering, or anomaly-detection
    problem. It also surfaces algorithm and preprocessing recommendations.
    """
    try:
        df = _load_dataframe(file_path)
        problem_types: List[str] = []
        suggestions: List[str] = []
        notes: List[str] = []

        if target_column and target_column not in df.columns:
            return {"status": "error", "error": f"Target column '{target_column}' not found."}

        # Detect temporal problems
        time_candidate = time_column
        if time_candidate is None:
            datetime_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.datetime64)]
            if datetime_cols:
                time_candidate = datetime_cols[0]

        if time_candidate and time_candidate in df.columns:
            problem_types.append("time_series")
            suggestions.append("ARIMA, Prophet, LSTM, Temporal CNN, transformers for time series")
            notes.append(f"Column '{time_candidate}' contains temporal information. Consider sorting by this column and creating lag features.")

        if target_column:
            target = df[target_column]
            unique_values = target.nunique(dropna=True)
            if np.issubdtype(target.dtype, np.number):
                # Heuristic: treat as classification if limited unique values relative to dataset size
                if unique_values <= 20 or unique_values / max(len(target), 1) < 0.05:
                    problem_types.append("classification")
                    suggestions.append("Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, neural networks")
                else:
                    problem_types.append("regression")
                    suggestions.append("Linear Regression, Random Forest Regressor, Gradient Boosting, XGBoost, neural networks")
            else:
                problem_types.append("classification")
                if unique_values == 2:
                    suggestions.append("Logistic Regression, SVM, Random Forest, Gradient Boosting")
                else:
                    suggestions.append("Multiclass classifiers such as Random Forest, XGBoost, LightGBM, neural networks")

            notes.append(
                f"Target column '{target_column}' has {unique_values} unique values."
            )
        else:
            # No explicit target; recommend unsupervised approaches
            problem_types.append("unsupervised_learning")
            suggestions.append("KMeans, DBSCAN, PCA, Autoencoders, Isolation Forest")
            notes.append("No target column provided. Consider clustering or anomaly detection, or specify a target for supervised tasks.")

        # If dataset contains high-cardinality categorical features, flag encoding
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if categorical_cols:
            notes.append(
                "Detected categorical columns: " + ", ".join(categorical_cols) + ". Plan encoding (one-hot, target encoding)."
            )

        return {
            "status": "success",
            "file_path": str(Path(file_path).resolve()),
            "problem_types": problem_types,
            "recommended_algorithms": suggestions,
            "notes": notes,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


class DataCleaningInput(BaseModel):
    file_path: str = Field(..., description="Path to the dataset to clean")
    operations: List[Literal[
        "drop_nulls",
        "fill_nulls",
        "remove_duplicates",
        "handle_outliers",
        "normalize",
        "encode_categorical",
    ]] = Field(
        ..., description="Cleaning operations to perform"
    )
    output_path: Optional[str] = Field(
        default=None, description="Path to save the cleaned dataset. If omitted, data is not persisted."
    )


@tool(args_schema=DataCleaningInput)
def clean_dataset(
    file_path: str,
    operations: List[str],
    output_path: Optional[str] = None,
) -> dict:
    """
    Execute automated data-cleaning operations.

    Supports null handling, duplicate removal, outlier filtering, normalization,
    and categorical encoding for rapid dataset preparation. When an
    ``output_path`` is provided the transformed dataset is persisted to disk.
    """
    try:
        df = _load_dataframe(file_path).copy()
        original_shape = df.shape
        operations_performed: List[str] = []

        for operation in operations:
            if operation == "drop_nulls":
                before = len(df)
                df = df.dropna()
                operations_performed.append(f"Dropped {before - len(df)} rows with missing values")

            elif operation == "fill_nulls":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(include=["object", "category"]).columns

                for col in numeric_cols:
                    if df[col].isnull().any():
                        df[col] = df[col].fillna(df[col].median())
                for col in categorical_cols:
                    if df[col].isnull().any():
                        df[col] = df[col].fillna(df[col].mode().iat[0])
                operations_performed.append("Filled null values using median/mode strategies")

            elif operation == "remove_duplicates":
                before = len(df)
                df = df.drop_duplicates()
                operations_performed.append(f"Removed {before - len(df)} duplicate rows")

            elif operation == "handle_outliers":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                removed = 0
                for col in numeric_cols:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    mask = df[col].between(lower, upper) | df[col].isnull()
                    removed += int((~mask).sum())
                    df = df[mask]
                operations_performed.append(f"Removed {removed} outlier rows using IQR bounds")

            elif operation == "normalize":
                from sklearn.preprocessing import MinMaxScaler

                numeric_cols = df.select_dtypes(include=[np.number]).columns
                scaler = MinMaxScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                operations_performed.append(f"Normalized {len(numeric_cols)} numeric columns with Min-Max scaling")

            elif operation == "encode_categorical":
                categorical_cols = df.select_dtypes(include=["object", "category"]).columns
                if len(categorical_cols) > 0:
                    df = pd.get_dummies(df, columns=list(categorical_cols), drop_first=True)
                    operations_performed.append(
                        f"One-hot encoded {len(categorical_cols)} categorical columns (drop_first=True)"
                    )
                else:
                    operations_performed.append("No categorical columns found to encode")

        output_saved_to = None
        if output_path:
            output_path = str(Path(output_path).expanduser())
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            suffix = Path(output_path).suffix.lower()
            if suffix == ".csv" or suffix == "":
                df.to_csv(output_path, index=False)
            elif suffix in {".xlsx", ".xls"}:
                df.to_excel(output_path, index=False)
            elif suffix == ".parquet":
                df.to_parquet(output_path, index=False)
            else:
                df.to_csv(output_path, index=False)
            output_saved_to = output_path

        return {
            "status": "success",
            "original_shape": list(original_shape),
            "cleaned_shape": list(df.shape),
            "operations": operations_performed,
            "rows_remaining": int(df.shape[0]),
            "columns_remaining": int(df.shape[1]),
            "output_saved_to": output_saved_to or "not_saved",
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
