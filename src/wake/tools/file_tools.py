from langchain.tools import tool
from typing import Optional, Literal
from pydantic import BaseModel, Field
from pathlib import Path
import json
import yaml

####################################
# File Input/Output Tools
####################################

class ReadFileInput(BaseModel):
    file_path: str = Field(description="Path to the file to read")
    file_type: Optional[Literal["text", "json", "yaml", "markdown", "python", "auto"]] = Field(
        default="auto",
        description="Type of file. If 'auto', detects based on file extension"
    )
    
@tool(args_schema=ReadFileInput)
def read_file(file_path: str, file_type: str = "auto") -> dict:
    """
    Reads a file from disk and returns its contents.
    Supports text, JSON, YAML, Markdown, Python, and arbitrary files.
    Useful for inspecting datasets, scripts, logs, and configuration files.
    """
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}
    
    if file_type == "auto":
        suffix = path.suffix.lower()
        if suffix in [".json"]:
            file_type = "json"
        elif suffix in [".yaml", ".yml"]:
            file_type = "yaml"
        elif suffix in [".md"]:
            file_type = "markdown"
        elif suffix in [".py"]:
            file_type = "python"
        else:
            file_type = "text"
    
    content = path.read_text(encoding="utf-8")
    
    if file_type == "json":
        try:
            return {"type": "json", "content": json.loads(content)}
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse JSON: {str(e)}", "raw": content}
    elif file_type == "yaml":
        try:
            return {"type": "yaml", "content": yaml.safe_load(content)}
        except yaml.YAMLError as e:
            return {"error": f"Failed to parse YAML: {str(e)}", "raw": content}
    else:
        return {
            "type": file_type,
            "content": content,
            "metadata": {
                "size_bytes": path.stat().st_size,
                "last_modified": path.stat().st_mtime
            }
        }


class WriteFileInput(BaseModel):
    file_path: str = Field(description="Path to write to")
    content: str = Field(description="Content to write")
    overwrite: bool = Field(default=True, description="Whether to overwrite existing file")
    
@tool(args_schema=WriteFileInput)
def write_file(file_path: str, content: str, overwrite: bool = True) -> dict:
    """
    Writes content to a file. Creates directories if they don't exist.
    Useful for saving datasets, model reports, training logs, configuration files, etc.
    """
    path = Path(file_path)
    if path.exists() and not overwrite:
        return {"error": f"File already exists: {file_path}", "status": "not_overwritten"}
    
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    
    return {
        "status": "success",
        "file_path": str(path),
        "bytes_written": len(content)
    }


class AppendFileInput(BaseModel):
    file_path: str = Field(description="Path of the file to append to")
    content: str = Field(description="Content to append")
    
@tool(args_schema=AppendFileInput)
def append_to_file(file_path: str, content: str) -> dict:
    """
    Appends content to an existing file (creates it if missing).
    Useful for logging, progress updates, and incremental report generation.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")
    
    return {
        "status": "success",
        "file_path": str(path),
        "content_appended": content[:200] + ("..." if len(content) > 200 else "")
    }


class ListDirectoryInput(BaseModel):
    directory: str = Field(description="Directory to list")
    include_hidden: bool = Field(default=False, description="Show hidden files")
    
@tool(args_schema=ListDirectoryInput)
def list_directory(directory: str, include_hidden: bool = False) -> dict:
    """
    Lists files and subdirectories in the specified path.
    Includes metadata such as size and modification time.
    """
    path = Path(directory)
    if not path.exists():
        return {"error": f"Directory not found: {directory}"}
    
    items = []
    for item in sorted(path.iterdir()):
        if not include_hidden and item.name.startswith('.'):
            continue
        items.append({
            "name": item.name,
            "type": "directory" if item.is_dir() else "file",
            "size": item.stat().st_size,
            "last_modified": item.stat().st_mtime
        })
    
    return {
        "directory": str(path.resolve()),
        "items": items,
        "count": len(items)
    }
