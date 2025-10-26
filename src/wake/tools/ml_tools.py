from langchain.tools import tool
from typing import Optional, Literal, Any
from pydantic import BaseModel, Field

####################################
# Machine Learning Operations Tools
####################################

class TrainModelInput(BaseModel):
    model_type: Literal["classification", "regression", "clustering", "neural_network"] = Field(
        description="Type of ML model to train"
    )
    algorithm: str = Field(
        description="Specific algorithm to use (e.g., 'random_forest', 'xgboost', 'logistic_regression', 'linear_regression', 'kmeans', 'mlp')"
    )
    data_path: str = Field(description="Path to the training dataset")
    target_column: Optional[str] = Field(default=None, description="Name of the target/label column (not needed for unsupervised learning)")
    test_size: float = Field(default=0.2, description="Proportion of data to use for testing (0.0 to 1.0)")
    hyperparameters: Optional[dict] = Field(default=None, description="Dictionary of hyperparameters for the model")
    
@tool(args_schema=TrainModelInput)
def train_ml_model(
    model_type: str,
    algorithm: str,
    data_path: str,
    target_column: Optional[str] = None,
    test_size: float = 0.2,
    hyperparameters: Optional[dict] = None
) -> dict:
    """
    Trains a machine learning model using the specified algorithm and data.
    Handles data splitting, model training, and returns performance metrics.
    
    Supported algorithms:
    - Classification: random_forest, xgboost, logistic_regression, svm, decision_tree
    - Regression: linear_regression, ridge, lasso, xgboost, random_forest, svr
    - Clustering: kmeans, dbscan, hierarchical
    - Neural Networks: mlp, cnn, rnn, lstm
    
    Returns training metrics, validation metrics, and model performance summary.
    """
    return {
        "note": "This tool requires Python execution. Use execute_python_code() to:",
        "steps": [
            f"1. Load data from {data_path}",
            f"2. Split into train/test sets (test_size={test_size})",
            f"3. Initialize {algorithm} model with hyperparameters",
            "4. Train the model on training data",
            "5. Evaluate on test data",
            "6. Return metrics (accuracy, precision, recall, F1, MSE, R², etc.)",
            "7. Save model to disk"
        ],
        "model_type": model_type,
        "algorithm": algorithm,
        "hyperparameters": hyperparameters or "default"
    }


class HyperparameterTuningInput(BaseModel):
    model_type: str = Field(description="Type of model (classification/regression)")
    algorithm: str = Field(description="ML algorithm to tune")
    data_path: str = Field(description="Path to training data")
    target_column: str = Field(description="Target variable column name")
    param_grid: dict = Field(description="Dictionary defining hyperparameter search space")
    search_method: Literal["grid", "random", "bayesian"] = Field(default="grid", description="Search strategy")
    cv_folds: int = Field(default=5, description="Number of cross-validation folds")
    
@tool(args_schema=HyperparameterTuningInput)
def tune_hyperparameters(
    model_type: str,
    algorithm: str,
    data_path: str,
    target_column: str,
    param_grid: dict,
    search_method: str = "grid",
    cv_folds: int = 5
) -> dict:
    """
    Performs automated hyperparameter tuning using grid search, random search, or Bayesian optimization.
    Tests multiple hyperparameter combinations to find the best performing model configuration.
    
    Returns the best hyperparameters found, cross-validation scores, and performance comparison.
    """
    return {
        "note": "Hyperparameter tuning requires Python execution",
        "approach": f"{search_method} search with {cv_folds}-fold cross-validation",
        "param_grid": param_grid,
        "steps": [
            "1. Load and preprocess data",
            f"2. Set up {search_method} search with param_grid",
            f"3. Perform {cv_folds}-fold cross-validation",
            "4. Identify best hyperparameters",
            "5. Retrain model with best parameters",
            "6. Return tuning results and best model"
        ]
    }


class ModelEvaluationInput(BaseModel):
    model_path: str = Field(description="Path to the saved model file")
    test_data_path: str = Field(description="Path to test/validation dataset")
    metrics: list[str] = Field(
        description="Metrics to compute (e.g., 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mse', 'rmse', 'r2', 'mae')"
    )
    
@tool(args_schema=ModelEvaluationInput)
def evaluate_model(model_path: str, test_data_path: str, metrics: list[str]) -> dict:
    """
    Evaluates a trained ML model on test data and computes comprehensive performance metrics.
    
    Generates:
    - Classification metrics: accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix
    - Regression metrics: MSE, RMSE, MAE, R², adjusted R²
    - Visualizations: confusion matrix, ROC curves, prediction plots
    
    Returns detailed performance analysis and identifies areas for improvement.
    """
    return {
        "model_path": model_path,
        "test_data": test_data_path,
        "metrics_to_compute": metrics,
        "outputs": [
            "Performance metrics dictionary",
            "Confusion matrix (for classification)",
            "Feature importance (if available)",
            "Prediction errors analysis",
            "Recommendations for improvement"
        ]
    }


class NeuralNetworkInput(BaseModel):
    architecture: str = Field(
        description="Neural network architecture description (e.g., 'mlp', 'cnn', 'rnn', 'lstm', 'transformer')"
    )
    input_shape: list[int] = Field(description="Shape of input data (e.g., [784] for MNIST, [224, 224, 3] for images)")
    output_size: int = Field(description="Number of output neurons (classes for classification, 1 for regression)")
    hidden_layers: list[int] = Field(description="List of hidden layer sizes (e.g., [128, 64, 32])")
    activation: str = Field(default="relu", description="Activation function (relu, tanh, sigmoid, etc.)")
    optimizer: str = Field(default="adam", description="Optimizer (adam, sgd, rmsprop, etc.)")
    learning_rate: float = Field(default=0.001, description="Learning rate for training")
    
@tool(args_schema=NeuralNetworkInput)
def build_neural_network(
    architecture: str,
    input_shape: list[int],
    output_size: int,
    hidden_layers: list[int],
    activation: str = "relu",
    optimizer: str = "adam",
    learning_rate: float = 0.001
) -> dict:
    """
    Builds a neural network architecture based on specifications.
    Supports various architectures: MLP, CNN, RNN, LSTM, Transformer.
    
    Returns the model architecture summary and training configuration.
    Can be used with PyTorch or TensorFlow/Keras backends.
    """
    return {
        "architecture_type": architecture,
        "configuration": {
            "input_shape": input_shape,
            "output_size": output_size,
            "hidden_layers": hidden_layers,
            "activation": activation,
            "optimizer": optimizer,
            "learning_rate": learning_rate
        },
        "next_steps": [
            "1. Use execute_python_code() to instantiate the model",
            "2. Load and preprocess training data",
            "3. Use train_neural_network() to begin training",
            "4. Monitor training with observe_training() for real-time metrics"
        ]
    }


class TrainNeuralNetworkInput(BaseModel):
    model_description: str = Field(description="Description of the neural network to train")
    train_data_path: str = Field(description="Path to training data")
    val_data_path: Optional[str] = Field(default=None, description="Path to validation data")
    epochs: int = Field(default=10, description="Number of training epochs")
    batch_size: int = Field(default=32, description="Batch size for training")
    early_stopping: bool = Field(default=True, description="Enable early stopping based on validation loss")
    save_checkpoints: bool = Field(default=True, description="Save model checkpoints during training")
    
@tool(args_schema=TrainNeuralNetworkInput)
def train_neural_network(
    model_description: str,
    train_data_path: str,
    val_data_path: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 32,
    early_stopping: bool = True,
    save_checkpoints: bool = True
) -> dict:
    """
    Trains a neural network with real-time monitoring and automatic optimization.
    
    Features:
    - Real-time loss and accuracy tracking
    - Automatic learning rate scheduling
    - Early stopping to prevent overfitting
    - Checkpoint saving for best models
    - GPU acceleration if available
    
    Returns training history, best epoch, and final model performance.
    """
    return {
        "model": model_description,
        "training_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "early_stopping": early_stopping,
            "checkpointing": save_checkpoints
        },
        "monitoring": [
            "Training loss per epoch",
            "Validation loss per epoch",
            "Training accuracy/metrics",
            "Validation accuracy/metrics",
            "Learning rate schedule",
            "GPU memory usage"
        ],
        "note": "Use observe_training() to monitor training in real-time"
    }


class FeatureEngineeringInput(BaseModel):
    data_path: str = Field(description="Path to dataset for feature engineering")
    operations: list[str] = Field(
        description="Feature engineering operations: 'polynomial', 'interaction', 'binning', 'scaling', 'pca', 'feature_selection'"
    )
    target_column: Optional[str] = Field(default=None, description="Target column name for supervised feature selection")
    
@tool(args_schema=FeatureEngineeringInput)
def engineer_features(data_path: str, operations: list[str], target_column: Optional[str] = None) -> dict:
    """
    Performs automated feature engineering to create new features and improve model performance.
    
    Available operations:
    - polynomial: Create polynomial features (x², x³, etc.)
    - interaction: Create interaction terms between features
    - binning: Discretize continuous variables into bins
    - scaling: Standardize or normalize features
    - pca: Apply PCA for dimensionality reduction
    - feature_selection: Select most important features using statistical tests or model-based methods
    
    Returns summary of new features created and their importance scores.
    """
    return {
        "data_path": data_path,
        "operations": operations,
        "target_column": target_column,
        "expected_outputs": [
            "New features created",
            "Feature importance rankings",
            "Correlation with target variable",
            "Recommended features to use",
            "Transformed dataset path"
        ]
    }
