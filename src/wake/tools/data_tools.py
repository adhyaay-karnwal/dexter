from langchain.tools import tool
from typing import Optional, Literal, Any
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from pathlib import Path
import json

####################################
# Data Loading and Analysis Tools
####################################

class LoadDataInput(BaseModel):
    file_path: str = Field(description="Path to the data file to load. Supports CSV, JSON, Excel, and Parquet formats.")
    file_type: Optional[Literal["csv", "json", "excel", "parquet", "auto"]] = Field(
        default="auto",
        description="Type of file to load. Use 'auto' to automatically detect based on file extension."
    )
    
@tool(args_schema=LoadDataInput)
def load_dataset(file_path: str, file_type: str = "auto") -> dict:
    """
    Loads a dataset from a file into memory for analysis.
    Automatically handles CSV, JSON, Excel (.xlsx, .xls), and Parquet files.
    Returns a summary of the loaded dataset including shape, columns, dtypes, and a preview of the first few rows.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
        
        # Auto-detect file type
        if file_type == "auto":
            ext = path.suffix.lower()
            if ext == ".csv":
                file_type = "csv"
            elif ext == ".json":
                file_type = "json"
            elif ext in [".xlsx", ".xls"]:
                file_type = "excel"
            elif ext == ".parquet":
                file_type = "parquet"
            else:
                return {"error": f"Unsupported file type: {ext}"}
        
        # Load based on file type
        if file_type == "csv":
            df = pd.read_csv(file_path)
        elif file_type == "json":
            df = pd.read_json(file_path)
        elif file_type == "excel":
            df = pd.read_excel(file_path)
        elif file_type == "parquet":
            df = pd.read_parquet(file_path)
        else:
            return {"error": f"Unsupported file type: {file_type}"}
        
        # Generate summary
        summary = {
            "status": "success",
            "file_path": str(file_path),
            "shape": df.shape,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "preview": df.head(5).to_dict(orient='records'),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2
        }
        
        return summary
    except Exception as e:
        return {"error": str(e)}


class StatisticsInput(BaseModel):
    data_description: str = Field(description="Description of the dataset or data source to analyze")
    columns: Optional[list[str]] = Field(default=None, description="Specific columns to analyze. If None, analyzes all numeric columns.")
    
@tool(args_schema=StatisticsInput)
def compute_statistics(data_description: str, columns: Optional[list[str]] = None) -> dict:
    """
    Computes comprehensive statistical summaries for a dataset.
    Provides mean, median, std, min, max, quartiles, skewness, kurtosis for numeric columns.
    Also provides correlation matrix for relationships between variables.
    Note: This tool requires data to be loaded first via load_dataset.
    """
    return {
        "note": "This is a planning tool. To compute actual statistics, you need to:",
        "steps": [
            "1. Load the dataset using load_dataset()",
            "2. Use execute_python_code() to compute statistics with pandas.describe(), correlation matrices, etc.",
            "3. Visualize distributions if needed using create_visualization()"
        ],
        "data_description": data_description,
        "columns": columns
    }


class ClassifyDataInput(BaseModel):
    data_description: str = Field(description="Description of the dataset to classify")
    features: list[str] = Field(description="List of feature column names to consider")
    
@tool(args_schema=ClassifyDataInput)
def classify_dataset_type(data_description: str, features: list[str]) -> dict:
    """
    Analyzes a dataset and classifies it into ML problem types.
    Determines if the data is suitable for:
    - Classification (binary/multiclass)
    - Regression
    - Time series analysis
    - Clustering
    - Anomaly detection
    
    Also suggests appropriate ML algorithms and preprocessing steps.
    """
    problem_types = []
    suggestions = []
    
    # Basic heuristics for classification
    feature_keywords = " ".join(features).lower()
    
    if any(keyword in feature_keywords for keyword in ["target", "label", "class", "category"]):
        problem_types.append("classification")
        suggestions.append("Consider: Logistic Regression, Random Forest, XGBoost, Neural Networks")
    
    if any(keyword in feature_keywords for keyword in ["price", "value", "amount", "score", "rate"]):
        problem_types.append("regression")
        suggestions.append("Consider: Linear Regression, Ridge/Lasso, Gradient Boosting, Neural Networks")
    
    if any(keyword in feature_keywords for keyword in ["time", "date", "timestamp", "year", "month"]):
        problem_types.append("time_series")
        suggestions.append("Consider: ARIMA, Prophet, LSTM, Temporal CNN")
    
    return {
        "data_description": data_description,
        "features_analyzed": features,
        "identified_problem_types": problem_types if problem_types else ["unsupervised_learning"],
        "ml_suggestions": suggestions if suggestions else ["Consider: K-Means, DBSCAN, PCA, Autoencoders"],
        "preprocessing_recommendations": [
            "Check for missing values and outliers",
            "Normalize/standardize features",
            "Encode categorical variables",
            "Split into train/test sets",
            "Consider feature engineering"
        ]
    }


class DataCleaningInput(BaseModel):
    file_path: str = Field(description="Path to the dataset to clean")
    operations: list[str] = Field(
        description="List of cleaning operations to perform: 'drop_nulls', 'fill_nulls', 'remove_duplicates', 'handle_outliers', 'normalize', 'encode_categorical'"
    )
    output_path: Optional[str] = Field(default=None, description="Path to save the cleaned dataset. If None, returns summary only.")
    
@tool(args_schema=DataCleaningInput)
def clean_dataset(file_path: str, operations: list[str], output_path: Optional[str] = None) -> dict:
    """
    Performs automated data cleaning operations on a dataset.
    
    Available operations:
    - drop_nulls: Remove rows with missing values
    - fill_nulls: Fill missing values with mean/median/mode
    - remove_duplicates: Remove duplicate rows
    - handle_outliers: Detect and handle outliers using IQR method
    - normalize: Scale numeric features to 0-1 range
    - encode_categorical: Convert categorical variables to numeric
    
    Returns a summary of operations performed and their effects.
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        original_shape = df.shape
        operations_performed = []
        
        for operation in operations:
            if operation == "drop_nulls":
                before = len(df)
                df = df.dropna()
                operations_performed.append(f"Dropped {before - len(df)} rows with null values")
            
            elif operation == "fill_nulls":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if df[col].isnull().any():
                        df[col].fillna(df[col].median(), inplace=True)
                operations_performed.append(f"Filled null values in {len(numeric_cols)} numeric columns with median")
            
            elif operation == "remove_duplicates":
                before = len(df)
                df = df.drop_duplicates()
                operations_performed.append(f"Removed {before - len(df)} duplicate rows")
            
            elif operation == "handle_outliers":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                outliers_removed = 0
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    before = len(df)
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    outliers_removed += (before - len(df))
                operations_performed.append(f"Removed {outliers_removed} outliers using IQR method")
            
            elif operation == "normalize":
                from sklearn.preprocessing import MinMaxScaler
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                scaler = MinMaxScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                operations_performed.append(f"Normalized {len(numeric_cols)} numeric columns to 0-1 range")
            
            elif operation == "encode_categorical":
                categorical_cols = df.select_dtypes(include=['object']).columns
                df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
                operations_performed.append(f"One-hot encoded {len(categorical_cols)} categorical columns")
        
        # Save if output path provided
        if output_path:
            if output_path.endswith('.csv'):
                df.to_csv(output_path, index=False)
            elif output_path.endswith('.parquet'):
                df.to_parquet(output_path, index=False)
            else:
                df.to_csv(output_path, index=False)
        
        return {
            "status": "success",
            "original_shape": original_shape,
            "cleaned_shape": df.shape,
            "operations_performed": operations_performed,
            "rows_remaining": len(df),
            "columns_remaining": len(df.columns),
            "output_saved_to": output_path if output_path else "Not saved"
        }
    except Exception as e:
        return {"error": str(e)}
