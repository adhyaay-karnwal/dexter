from typing import Callable

from wake.tools.data_tools import (
    load_dataset,
    compute_statistics,
    classify_dataset_type,
    clean_dataset,
)
from wake.tools.ml_tools import (
    train_ml_model,
    tune_hyperparameters,
    evaluate_model,
    build_neural_network,
    train_neural_network,
    engineer_features,
)
from wake.tools.file_tools import (
    read_file,
    write_file,
    append_to_file,
    list_directory,
)
from wake.tools.terminal_tools import (
    execute_python_code,
    execute_shell_command,
    observe_training_process,
    check_system_resources,
    install_python_package,
)
from wake.tools.research_tools import (
    search_web,
    fetch_url_content,
    download_dataset,
    get_ml_library_info,
)
from wake.tools.visualization_tools import (
    create_visualization,
    plot_training_history,
)
from wake.tools.observe_integration import observe_process

TOOLS: list[Callable[..., object]] = [
    # Data operations
    load_dataset,
    compute_statistics,
    classify_dataset_type,
    clean_dataset,
    # File I/O
    read_file,
    write_file,
    append_to_file,
    list_directory,
    # Research & data acquisition
    search_web,
    fetch_url_content,
    download_dataset,
    get_ml_library_info,
    # ML operations
    train_ml_model,
    tune_hyperparameters,
    evaluate_model,
    build_neural_network,
    train_neural_network,
    engineer_features,
    # Execution & system tools
    execute_python_code,
    execute_shell_command,
    observe_training_process,
    observe_process,
    check_system_resources,
    install_python_package,
    # Visualization
    create_visualization,
    plot_training_history,
]
