from langchain.tools import tool
from typing import Optional
from pydantic import BaseModel, Field
import subprocess
import os

####################################
# Observe Integration (Real-time Terminal Observation)
####################################

class ObserveProcessInput(BaseModel):
    command: str = Field(description="Command to run under observation")
    working_directory: Optional[str] = Field(default=None, description="Directory to execute the command in")
    log_path: Optional[str] = Field(default=None, description="Optional log file to stream")
    
@tool(args_schema=ObserveProcessInput)
def observe_process(
    command: str,
    working_directory: Optional[str] = None,
    log_path: Optional[str] = None
) -> dict:
    """
    Integrates with the observe CLI (https://github.com/rajansagarwal/observe) to stream real-time terminal output.
    
    Usage:
    1. Ensure `observe` CLI is installed (`pip install observe`)
    2. Use this tool to run a long-running command with live streaming
    3. Optionally provide log_path to tail and display in real time
    
    Returns the command output and observation status.
    """
    try:
        observe_command = f"observe run '{command}'"
        if log_path:
            observe_command = f"observe run '{command}' --tail {log_path}"
        
        result = subprocess.run(
            observe_command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=working_directory
        )
        
        return {
            "status": "success" if result.returncode == 0 else "error",
            "command": command,
            "observe_command": observe_command,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "note": "Install observe CLI with `pip install observe`. Requires observe server running for web dashboard."
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "command": command,
            "note": "Ensure observe is installed and accessible in PATH."
        }
