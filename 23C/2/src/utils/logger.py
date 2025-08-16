"""
Logging utilities
"""

import logging
import os
from datetime import datetime
from config.config import Config

def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """Setup logger with specified name and optional log file"""
    
    # Create logs directory if it doesn't exist
    os.makedirs(Config.LOGS_DIR, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, Config.LOG_LEVEL))
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(Config.LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'{name}_{timestamp}.log'
    
    file_handler = logging.FileHandler(Config.get_log_path(log_file))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger