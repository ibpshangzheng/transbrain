# config/logger.py
import os
import logging
from typing import Optional

_global_logger: Optional[logging.Logger] = None

def init_logger(log_dir: str = "logs",log_file: str = "train_dnn.log",level: int = logging.INFO,
    file_mode: str = 'a') -> logging.Logger:

    global _global_logger
    
    if _global_logger is not None:
        return _global_logger 
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("GlobalLogger")
    logger.setLevel(level)

    file_handler = logging.FileHandler(
        filename=os.path.join(log_dir, log_file),
        mode=file_mode,
        encoding='utf-8'
    )
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('[%(levelname)s] %(message)s')
    )
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    _global_logger = logger
    return logger

def get_logger() -> logging.Logger:

    if _global_logger is None:
        raise RuntimeError("Logger not initialized. Call init_logger() first.")
    return _global_logger