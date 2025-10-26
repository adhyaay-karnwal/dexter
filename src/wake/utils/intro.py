def print_intro():
    """Display the Wake welcome screen with ASCII art."""
    # ANSI color codes
    WAKE_AQUA = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Clear screen effect with some spacing
    print("\n" * 2)
    
    # Welcome box with aqua border
    box_width = 60
    welcome_text = "Wake — Autonomous ML Research Agent"
    padding = (box_width - len(welcome_text) - 2) // 2
    
    print(f"{WAKE_AQUA}{'═' * box_width}{RESET}")
    print(
        f"{WAKE_AQUA}║{' ' * padding}{BOLD}{welcome_text}{RESET}{WAKE_AQUA}{' ' * (box_width - len(welcome_text) - padding - 2)}║{RESET}"
    )
    print(f"{WAKE_AQUA}{'═' * box_width}{RESET}")
    print()
    
    # ASCII art for WAKE in a clean, technical style
    wake_art = f"""{BOLD}{WAKE_AQUA}
██╗    ██╗ █████╗ ██╗  ██╗███████╗
██║    ██║██╔══██╗██║ ██╔╝██╔════╝
██║ █╗ ██║███████║█████╔╝ █████╗  
██║███╗██║██╔══██║██╔═██╗ ██╔══╝  
╚███╔███╔╝██║  ██║██║  ██╗███████╗
 ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
{RESET}"""
    
    print(wake_art)
    print()
    print("Your autonomous engineer for machine learning research and model operations.")
    print("Type 'exit' or 'quit' to end the session.")
    print()

