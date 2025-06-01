# filepath: c:\Users\Bachn\OneDrive\Desktop\my_projects\hdbscan\logging_setup.py
import logging
import sys
from datetime import datetime

def setup_detailed_logging(log_level=logging.INFO):
    """
    Setup detailed logging for the clustering pipeline
    
    Args:
        log_level: Logging level (logging.INFO, logging.DEBUG, etc.)
    """
    # Create formatter with detailed information
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler for real-time monitoring
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
      # File handler for detailed logs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"clustering_pipeline_{timestamp}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # Always debug level for file
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=[console_handler, file_handler],
        force=True
    )
    
    logging.info("Logging setup complete. Detailed logs saved to: %s", log_filename)
    return log_filename

# Example usage:
if __name__ == "__main__":
    setup_detailed_logging(logging.INFO)
    
    # Test the logging
    logging.info("This is an info message")
    logging.warning("This is a warning message")
    logging.error("This is an error message")
    logging.debug("This is a debug message (only in file)")