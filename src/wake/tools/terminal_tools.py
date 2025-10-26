from langchain.tools import tool
from typing import List, Optional
from pydantic import BaseModel, Field
import subprocess
import os
import sys

####################################
# Terminal Execution and Observation Tools
####################################

class ExecutePythonInput(BaseModel):
    code: str = Field(description="Python code to execute")
    working_directory: Optional[str] = Field(default=None, description="Directory to run the code in")
    capture_output: bool = Field(default=True, description="Whether to capture and return output")
    timeout: int = Field(default=300, description="Timeout in seconds (default: 5 minutes)")
    
@tool(args_schema=ExecutePythonInput)
def execute_python_code(
    code: str,
    working_directory: Optional[str] = None,
    capture_output: bool = True,
    timeout: int = 300
) -> dict:
    """
    Executes Python code in a sandboxed environment and returns the output.
    
    Use this tool for:
    - Running ML training scripts
    - Data preprocessing and analysis
    - Computing statistics and metrics
    - Testing model predictions
    - Generating visualizations
    
    The code runs with access to common ML libraries: numpy, pandas, scikit-learn, matplotlib.
    Returns stdout, stderr, and execution status.
    """
    try:
        # Change to working directory if specified
        original_dir = os.getcwd()
        if working_directory:
            os.chdir(working_directory)
        
        # Execute the code
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=capture_output,
            text=True,
            timeout=timeout
        )
        
        # Return to original directory
        if working_directory:
            os.chdir(original_dir)
        
        return {
            "status": "success" if result.returncode == 0 else "error",
            "return_code": result.returncode,
            "stdout": result.stdout if capture_output else "Output not captured",
            "stderr": result.stderr if capture_output else "Errors not captured",
            "execution_time": "completed within timeout"
        }
    except subprocess.TimeoutExpired:
        if working_directory:
            os.chdir(original_dir)
        return {
            "status": "timeout",
            "error": f"Execution exceeded {timeout} seconds timeout",
            "suggestion": "Consider breaking the task into smaller parts or increasing timeout"
        }
    except Exception as e:
        if working_directory:
            os.chdir(original_dir)
        return {
            "status": "error",
            "error": str(e)
        }


class ExecuteCommandInput(BaseModel):
    command: str = Field(description="Shell command to execute")
    working_directory: Optional[str] = Field(default=None, description="Directory to run the command in")
    timeout: int = Field(default=60, description="Timeout in seconds")
    
@tool(args_schema=ExecuteCommandInput)
def execute_shell_command(command: str, working_directory: Optional[str] = None, timeout: int = 60) -> dict:
    """
    Executes a shell command and returns the output.
    
    Useful for:
    - Installing packages (pip install, conda install)
    - Running training scripts (.py files)
    - Managing files and directories
    - Checking system resources
    - Starting/stopping processes
    
    Returns command output, errors, and exit code.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_directory
        )
        
        return {
            "status": "success" if result.returncode == 0 else "error",
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": command
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "error": f"Command exceeded {timeout} seconds timeout",
            "command": command
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "command": command
        }


class MonitorProcessInput(BaseModel):
    process_description: str = Field(description="Description of the process to monitor")
    log_file: Optional[str] = Field(default=None, description="Path to log file to parse for metrics")
    metrics_to_track: list[str] = Field(
        default=["loss", "accuracy", "val_loss", "val_accuracy", "epoch"],
        description="Metrics to track during training"
    )
    history_window: int = Field(
        default=5,
        description="Number of recent entries to use when detecting trends/stagnation"
    )


@tool(args_schema=MonitorProcessInput)
def observe_training_process(
    process_description: str,
    log_file: Optional[str] = None,
    metrics_to_track: list[str] = ["loss", "accuracy", "val_loss", "val_accuracy", "epoch"],
    history_window: int = 5,
) -> dict:
    """Parse training logs and surface current status, trends, and warnings."""
    import json
    import re
    from collections import defaultdict, deque

    result = {
        "process": process_description,
        "log_file": log_file,
        "metrics": {},
        "trend_analysis": {},
        "warnings": [],
    }

    if not log_file:
        result["note"] = "Provide a log_file path to enable parsing. Redirect your trainer output to a file."
        return result

    path = os.path.expanduser(log_file)
    if not os.path.exists(path):
        result["status"] = "error"
        result["error"] = f"Log file not found: {log_file}"
        return result

    try:
        # Read the tail of the file to keep memory usage low
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            handle.seek(0, os.SEEK_END)
            file_size = handle.tell()
            handle.seek(max(file_size - 10_000, 0))
            tail = handle.read()
    except Exception as exc:  # pragma: no cover - defensive
        result["status"] = "error"
        result["error"] = str(exc)
        return result

    metric_pattern = re.compile(r"(?P<metric>\b[a-zA-Z_]+\b)\s*[=:]\s*(?P<value>-?\d+\.\d+|-?\d+)")
    history: dict[str, deque] = defaultdict(lambda: deque(maxlen=history_window))
    extracted_lines: List[str] = []

    for line in tail.splitlines():
        matches = metric_pattern.findall(line)
        if not matches:
            continue
        line_metrics = {}
        for metric, value in matches:
            if metric in metrics_to_track:
                numeric_value = float(value)
                history[metric].append(numeric_value)
                line_metrics[metric] = numeric_value
        if line_metrics:
            extracted_lines.append(json.dumps(line_metrics))

    result["metrics"] = {metric: list(values) for metric, values in history.items()}
    result["recent_entries"] = extracted_lines[-history_window:]

    # Trend analysis
    for metric, values in history.items():
        if len(values) < 2:
            continue
        delta = values[-1] - values[0]
        if metric.lower().startswith("val") or metric.lower().endswith("loss"):
            if abs(delta) < 1e-3:
                result["trend_analysis"][metric] = "Plateau detected"
            elif delta > 0:
                result["trend_analysis"][metric] = "Trending upward (possible degradation)"
            else:
                result["trend_analysis"][metric] = "Improving"
        else:
            if abs(delta) < 1e-3:
                result["trend_analysis"][metric] = "Flat trend"
            elif delta > 0:
                result["trend_analysis"][metric] = "Trending upward"
            else:
                result["trend_analysis"][metric] = "Trending downward"

    # Simple warnings
    if "loss" in history and "val_loss" in history:
        if history["val_loss"] and history["loss"] and history["val_loss"][-1] > history["loss"][-1] * 1.2:
            result["warnings"].append("Validation loss notably higher than training loss (possible overfitting)")
    if "accuracy" in history and "val_accuracy" in history:
        if history["val_accuracy"] and history["accuracy"] and history["val_accuracy"][-1] < history["accuracy"][-1] * 0.8:
            result["warnings"].append("Validation accuracy significantly below training accuracy")

    result["status"] = "success"
    return result


class CheckSystemResourcesInput(BaseModel):
    detailed: bool = Field(default=False, description="Return detailed resource information")
    
@tool(args_schema=CheckSystemResourcesInput)
def check_system_resources(detailed: bool = False) -> dict:
    """
    Checks available system resources including CPU, memory, GPU, and disk space.
    
    Returns:
    - CPU usage and core count
    - Available RAM and total memory
    - GPU availability and memory (if CUDA available)
    - Disk space
    - Python environment information
    
    Useful for determining if system can handle training workload.
    """
    import psutil
    
    try:
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_total_gb = memory.total / (1024 ** 3)
        memory_available_gb = memory.available / (1024 ** 3)
        memory_percent = memory.percent
        
        # Disk info
        disk = psutil.disk_usage('/')
        disk_total_gb = disk.total / (1024 ** 3)
        disk_free_gb = disk.free / (1024 ** 3)
        disk_percent = disk.percent
        
        result = {
            "cpu": {
                "cores": cpu_count,
                "usage_percent": cpu_percent,
                "status": "OK" if cpu_percent < 80 else "High Load"
            },
            "memory": {
                "total_gb": round(memory_total_gb, 2),
                "available_gb": round(memory_available_gb, 2),
                "usage_percent": memory_percent,
                "status": "OK" if memory_percent < 80 else "High Usage"
            },
            "disk": {
                "total_gb": round(disk_total_gb, 2),
                "free_gb": round(disk_free_gb, 2),
                "usage_percent": disk_percent,
                "status": "OK" if disk_percent < 90 else "Low Space"
            },
            "python_version": sys.version,
            "platform": sys.platform
        }
        
        # Try to get GPU info
        try:
            import torch
            result["gpu"] = {
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            }
        except ImportError:
            result["gpu"] = {"status": "PyTorch not installed, GPU info unavailable"}
        
        return result
    except Exception as e:
        return {"error": str(e)}


class InstallPackageInput(BaseModel):
    package_name: str = Field(description="Name of the Python package to install")
    version: Optional[str] = Field(default=None, description="Specific version to install (e.g., '1.0.0')")
    upgrade: bool = Field(default=False, description="Whether to upgrade if already installed")
    
@tool(args_schema=InstallPackageInput)
def install_python_package(package_name: str, version: Optional[str] = None, upgrade: bool = False) -> dict:
    """
    Installs a Python package using pip.
    
    Useful for installing ML libraries on-the-fly:
    - torch, tensorflow, keras
    - xgboost, lightgbm, catboost
    - transformers, datasets
    - optuna, ray[tune]
    - Any other PyPI package
    
    Returns installation status and any errors.
    """
    try:
        package_spec = package_name
        if version:
            package_spec = f"{package_name}=={version}"
        
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(package_spec)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        return {
            "status": "success" if result.returncode == 0 else "error",
            "package": package_spec,
            "stdout": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
            "stderr": result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
        }
    except Exception as e:
        return {
            "status": "error",
            "package": package_name,
            "error": str(e)
        }
