from langchain.tools import tool
from typing import Optional, Literal
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup
import json

####################################
# Research and Web Scraping Tools
####################################

class WebSearchInput(BaseModel):
    query: str = Field(description="Search query for ML research, papers, datasets, or libraries")
    search_type: Literal["general", "academic", "github", "datasets"] = Field(
        default="general",
        description="Type of search: general web, academic papers, GitHub repos, or datasets"
    )
    
@tool(args_schema=WebSearchInput)
def search_web(query: str, search_type: str = "general") -> dict:
    """
    Searches the web for ML-related information: papers, datasets, tutorials, libraries, documentation.
    
    Search types:
    - general: Web search for tutorials, blog posts, documentation
    - academic: Academic papers and research (arXiv, Google Scholar)
    - github: GitHub repositories and code examples
    - datasets: ML datasets (Kaggle, UCI, HuggingFace datasets)
    
    Returns a list of relevant URLs, titles, and snippets.
    """
    # For production, integrate with a real search API (Bing, Serper, etc.)
    # This is a placeholder implementation
    
    search_urls = {
        "general": f"https://www.google.com/search?q={query}+machine+learning",
        "academic": f"https://arxiv.org/search/?query={query}&searchtype=all",
        "github": f"https://github.com/search?q={query}+machine+learning",
        "datasets": f"https://www.kaggle.com/search?q={query}"
    }
    
    return {
        "query": query,
        "search_type": search_type,
        "note": "Web search tool - requires API integration for live results",
        "suggested_sources": {
            "academic": [
                f"arXiv.org: https://arxiv.org/search/?query={query}",
                f"Papers With Code: https://paperswithcode.com/search?q={query}",
                f"Google Scholar: https://scholar.google.com/scholar?q={query}"
            ],
            "datasets": [
                f"Kaggle: https://www.kaggle.com/search?q={query}",
                f"HuggingFace: https://huggingface.co/datasets?search={query}",
                f"UCI ML Repo: https://archive.ics.uci.edu/ml/datasets.php"
            ],
            "github": [
                f"GitHub: https://github.com/search?q={query}",
                f"Awesome lists: https://github.com/search?q=awesome+{query}"
            ]
        }
    }


class FetchURLInput(BaseModel):
    url: str = Field(description="URL to fetch content from")
    extract_type: Literal["text", "html", "links", "metadata"] = Field(
        default="text",
        description="What to extract: plain text, HTML, links, or metadata"
    )
    
@tool(args_schema=FetchURLInput)
def fetch_url_content(url: str, extract_type: str = "text") -> dict:
    """
    Fetches content from a URL and extracts relevant information.
    Useful for downloading documentation, reading papers (HTML), or scraping data sources.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Wake ML Agent/1.0)"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        if extract_type == "text":
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return {
                "url": url,
                "type": "text",
                "content": text[:5000] + ("..." if len(text) > 5000 else ""),
                "full_length": len(text)
            }
        
        elif extract_type == "html":
            return {
                "url": url,
                "type": "html",
                "content": response.text[:5000],
                "full_length": len(response.text)
            }
        
        elif extract_type == "links":
            links = []
            for link in soup.find_all('a', href=True):
                links.append({
                    "text": link.get_text(strip=True),
                    "href": link['href']
                })
            return {
                "url": url,
                "type": "links",
                "links": links[:100],
                "total_links": len(links)
            }
        
        elif extract_type == "metadata":
            title = soup.find('title')
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            
            return {
                "url": url,
                "type": "metadata",
                "title": title.string if title else "No title",
                "description": meta_desc['content'] if meta_desc and 'content' in meta_desc.attrs else "No description",
                "status_code": response.status_code
            }
        
    except Exception as e:
        return {
            "error": str(e),
            "url": url
        }


class DownloadDatasetInput(BaseModel):
    dataset_name: str = Field(description="Name of the dataset to download")
    source: Literal["kaggle", "huggingface", "sklearn", "tensorflow", "uci"] = Field(
        description="Source platform for the dataset"
    )
    save_path: str = Field(description="Directory to save the downloaded dataset")
    
@tool(args_schema=DownloadDatasetInput)
def download_dataset(dataset_name: str, source: str, save_path: str) -> dict:
    """
    Downloads ML datasets from popular sources.
    
    Supported sources:
    - kaggle: Kaggle datasets (requires API key)
    - huggingface: HuggingFace datasets
    - sklearn: Built-in sklearn datasets
    - tensorflow: TensorFlow datasets
    - uci: UCI Machine Learning Repository
    
    Returns download status and dataset information.
    """
    from pathlib import Path
    
    save_path_obj = Path(save_path)
    save_path_obj.mkdir(parents=True, exist_ok=True)
    
    if source == "sklearn":
        return {
            "note": "Use execute_python_code() to load sklearn datasets:",
            "example": f"""
from sklearn.datasets import load_{dataset_name}
data = load_{dataset_name}()
# data.data, data.target, data.feature_names available
""",
            "available_datasets": [
                "iris", "digits", "wine", "breast_cancer", 
                "diabetes", "boston", "california_housing"
            ]
        }
    
    elif source == "huggingface":
        return {
            "note": "Use execute_python_code() with datasets library:",
            "example": f"""
from datasets import load_dataset
dataset = load_dataset("{dataset_name}")
dataset.save_to_disk("{save_path}")
""",
            "dataset": dataset_name,
            "save_path": str(save_path)
        }
    
    elif source == "tensorflow":
        return {
            "note": "Use execute_python_code() with TensorFlow datasets:",
            "example": f"""
import tensorflow_datasets as tfds
dataset = tfds.load("{dataset_name}", data_dir="{save_path}")
""",
            "dataset": dataset_name
        }
    
    elif source == "kaggle":
        return {
            "note": "Requires Kaggle API credentials in ~/.kaggle/kaggle.json",
            "command": f"kaggle datasets download -d {dataset_name} -p {save_path}",
            "setup": "pip install kaggle && kaggle datasets list"
        }
    
    else:
        return {
            "error": f"Unsupported source: {source}",
            "supported": ["kaggle", "huggingface", "sklearn", "tensorflow", "uci"]
        }


class GetMLLibraryInfoInput(BaseModel):
    library_name: str = Field(description="Name of the ML library (e.g., 'scikit-learn', 'pytorch', 'tensorflow')")
    info_type: Literal["overview", "installation", "examples", "documentation"] = Field(
        default="overview",
        description="Type of information to retrieve"
    )
    
@tool(args_schema=GetMLLibraryInfoInput)
def get_ml_library_info(library_name: str, info_type: str = "overview") -> dict:
    """
    Provides information about popular ML libraries: usage, installation, examples, documentation links.
    
    Covers: PyTorch, TensorFlow, Scikit-learn, XGBoost, LightGBM, Keras, HuggingFace, and more.
    """
    library_database = {
        "pytorch": {
            "overview": "Deep learning framework with dynamic computation graphs",
            "installation": "pip install torch torchvision torchaudio",
            "documentation": "https://pytorch.org/docs/",
            "examples": "https://pytorch.org/tutorials/"
        },
        "tensorflow": {
            "overview": "End-to-end ML platform with Keras integration",
            "installation": "pip install tensorflow",
            "documentation": "https://www.tensorflow.org/api_docs",
            "examples": "https://www.tensorflow.org/tutorials"
        },
        "scikit-learn": {
            "overview": "Classical ML algorithms library",
            "installation": "pip install scikit-learn",
            "documentation": "https://scikit-learn.org/stable/",
            "examples": "https://scikit-learn.org/stable/auto_examples/"
        },
        "xgboost": {
            "overview": "Gradient boosting framework",
            "installation": "pip install xgboost",
            "documentation": "https://xgboost.readthedocs.io/",
            "examples": "https://github.com/dmlc/xgboost/tree/master/demo"
        },
        "transformers": {
            "overview": "State-of-the-art NLP models (HuggingFace)",
            "installation": "pip install transformers",
            "documentation": "https://huggingface.co/docs/transformers/",
            "examples": "https://huggingface.co/course"
        }
    }
    
    if library_name.lower() not in library_database:
        return {
            "library": library_name,
            "status": "not_in_database",
            "suggestion": f"Search for '{library_name}' on PyPI or GitHub for documentation"
        }
    
    info = library_database[library_name.lower()]
    
    if info_type == "overview":
        return {
            "library": library_name,
            "overview": info["overview"],
            "documentation": info["documentation"]
        }
    elif info_type == "installation":
        return {
            "library": library_name,
            "installation_command": info["installation"],
            "note": "Run using execute_shell_command() or install_python_package()"
        }
    elif info_type == "examples":
        return {
            "library": library_name,
            "examples_url": info["examples"],
            "documentation": info["documentation"]
        }
    else:
        return info
