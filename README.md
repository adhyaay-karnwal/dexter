# Wake 🤖

Wake is an autonomous machine learning research and engineering agent that plans, executes, and optimizes end-to-end ML workflows. Built with a modular architecture, Wake combines task planning, self-validation, and comprehensive ML tooling to act as your autonomous ML engineer.

## Overview

Wake takes complex machine learning questions and challenges, breaking them down into clear, actionable task plans. It executes those tasks using a rich toolkit for data handling, model training, evaluation, and optimization—checking its own work and iterating until the job is done.

It's not just another chatbot. It's an agent that plans ahead, validates progress, and keeps iterating with ML best practices.

**Key Capabilities:**
- **Intelligent Task Planning**: Automatically decomposes ML workflows into structured, sequential steps
- **Autonomous Execution**: Selects and uses the right tools for data loading, training, evaluation, and more
- **Self-Validation**: Checks its own work and iterates until objectives are met
- **Comprehensive ML Toolkit**: Data processing, model training, neural networks, visualization, research tools
- **Real-Time Monitoring**: Observe training processes, detect issues, and optimize on-the-fly
- **Safety Features**: Loop detection, step limits, and resource management to prevent runaway execution

[![Twitter Follow](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

## Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key (get [here](https://platform.openai.com/api-keys))

## Installation

1. Clone the repository:
```bash
git clone https://github.com/virattt/wake.git
cd wake
```

2. Install dependencies with uv:
```bash
uv sync
```

3. Set up your environment variables:
```bash
# Copy the example environment file
cp env.example .env

# Edit .env and add your API key
# OPENAI_API_KEY=your-openai-api-key
```

## Usage

Run Wake in interactive mode:
```bash
uv run wake-agent
```

## Example Queries

Try asking Wake questions like:
- "Load the iris dataset and train a random forest classifier"
- "Analyze the MNIST dataset and build a CNN for digit classification"
- "Compare XGBoost and Random Forest performance on the wine quality dataset"
- "Tune hyperparameters for a neural network on my custom dataset.csv"
- "Visualize the correlation matrix for the Boston housing dataset"
- "Search for papers on transformer architectures for time series"

Wake will automatically:
1. Break down your request into ML workflow tasks
2. Load and analyze datasets
3. Preprocess and clean data
4. Train and evaluate models
5. Generate visualizations and reports
6. Provide comprehensive, metric-driven answers

## Architecture

Wake uses a multi-agent architecture with specialized components:

- **Planning Agent**: Analyzes ML queries and creates structured task sequences
- **Action Agent**: Selects appropriate tools and executes ML operations
- **Validation Agent**: Verifies task completion and ensures objectives are met
- **Answer Agent**: Synthesizes findings into comprehensive, actionable responses

## Tool Categories

Wake has access to a comprehensive ML toolkit organized into categories:

### Data Operations
- Load datasets (CSV, JSON, Excel, Parquet)
- Compute statistics and analyze distributions
- Classify dataset types (classification, regression, etc.)
- Clean data (handle missing values, outliers, normalization)

### ML Operations
- Train models (classification, regression, clustering, neural networks)
- Hyperparameter tuning (grid search, random search, Bayesian optimization)
- Model evaluation (metrics, confusion matrices, performance analysis)
- Build and train neural networks (MLP, CNN, RNN, LSTM)
- Feature engineering (polynomial features, PCA, feature selection)

### Execution & System
- Execute Python code for custom analysis
- Run shell commands
- Monitor training processes in real-time
- Check system resources (CPU, GPU, memory)
- Install Python packages on-the-fly

### File Operations
- Read files (text, JSON, YAML, scripts)
- Write results and reports
- Append to logs
- List and explore directories

### Research
- Web search for papers, datasets, tutorials
- Fetch documentation from URLs
- Download datasets (Kaggle, HuggingFace, sklearn, TensorFlow)
- Get ML library information and examples

### Visualization
- Create plots (line, scatter, histogram, heatmap, confusion matrix, ROC curves)
- Plot training history
- Generate reports with visualizations

## Project Structure

```
wake/
├── src/
│   └── wake/
│       ├── agent.py          # Main agent orchestration logic
│       ├── model.py           # LLM interface
│       ├── prompts.py         # System prompts (planning, action, validation, answer)
│       ├── schemas.py         # Pydantic models
│       ├── cli.py             # CLI entry point
│       ├── tools/             # ML toolkit
│       │   ├── data_tools.py
│       │   ├── ml_tools.py
│       │   ├── file_tools.py
│       │   ├── terminal_tools.py
│       │   ├── research_tools.py
│       │   ├── visualization_tools.py
│       │   └── observe_integration.py
│       └── utils/             # Utility functions
│           ├── logger.py
│           ├── ui.py
│           └── intro.py
├── pyproject.toml
└── uv.lock
```

## Configuration

Wake supports configuration via the `Agent` class initialization:

```python
from wake.agent import Agent

agent = Agent(
    max_steps=20,              # Global safety limit
    max_steps_per_task=5       # Per-task iteration limit
)
```

## Use Cases

Wake is designed for:

1. **ML Experimentation**: Quickly test different algorithms on datasets
2. **Model Training**: Autonomous training with hyperparameter optimization
3. **Data Analysis**: Exploratory data analysis with automatic insights
4. **Research**: Find papers, datasets, and best practices
5. **Model Evaluation**: Comprehensive performance analysis with visualizations
6. **Learning**: Understand ML concepts through hands-on experimentation
7. **Prototyping**: Rapidly prototype ML solutions

## Advanced Features

### Real-Time Training Observation
Wake can monitor training processes in real-time, detecting:
- Training stagnation (plateau in metrics)
- Overfitting (validation loss diverging from training loss)
- Optimal stopping points
- Resource issues (OOM errors, GPU problems)

### Autonomous Optimization
Wake can autonomously:
- Adjust hyperparameters based on performance
- Suggest architecture modifications
- Recommend preprocessing steps
- Identify and fix common issues

### Multi-Step Workflows
Wake chains operations intelligently:
```
Query → Load Data → Analyze → Clean → Train → Evaluate → Report
```

## How to Contribute

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

**Important**: Please keep your pull requests small and focused. This will make it easier to review and merge.

## Roadmap

- [ ] Integration with Weights & Biases for experiment tracking
- [ ] Support for LLM fine-tuning workflows
- [ ] Automated feature engineering pipelines
- [ ] Multi-modal learning support (vision + text)
- [ ] Distributed training capabilities
- [ ] Model deployment tools
- [ ] Custom plugin system for domain-specific tools

## License

This project is licensed under the MIT License.

---

**Wake** — Your Autonomous ML Research Engineer
