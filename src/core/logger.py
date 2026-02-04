"""
Logging Module
==============

Professional logging system with file and console handlers.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Colored console formatter."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


class PipelineLogger:
    """
    Professional logging system for ML pipeline.
    
    Features:
    - File and console handlers
    - Colored console output
    - Configurable log levels
    - Automatic log file rotation
    """
    
    _instance: Optional['PipelineLogger'] = None
    _logger: Optional[logging.Logger] = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        name: str = "CreditApprovalML",
        log_dir: str = "ml_pipeline_output/logs",
        level: int = logging.INFO,
        file_enabled: bool = True
    ):
        if self._logger is not None:
            return
        
        self.name = name
        self.log_dir = Path(log_dir)
        self.level = level
        self.file_enabled = file_enabled
        
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with handlers."""
        self._logger = logging.getLogger(self.name)
        self._logger.setLevel(self.level)
        self._logger.handlers.clear()
        self._logger.propagate = False
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_format = '%(asctime)s | %(levelname)s | %(message)s'
        console_handler.setFormatter(ColoredFormatter(console_format, datefmt='%H:%M:%S'))
        self._logger.addHandler(console_handler)
        
        # File handler
        if self.file_enabled:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                log_file = self.log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(logging.DEBUG)
                file_format = '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
                file_handler.setFormatter(logging.Formatter(file_format))
                self._logger.addHandler(file_handler)
            except Exception as e:
                self._logger.warning(f"Could not setup file logging: {e}")
    
    @property
    def logger(self) -> logging.Logger:
        """Get configured logger."""
        return self._logger
    
    def info(self, message: str):
        self._logger.info(message)
    
    def debug(self, message: str):
        self._logger.debug(message)
    
    def warning(self, message: str):
        self._logger.warning(message)
    
    def error(self, message: str):
        self._logger.error(message)
    
    def critical(self, message: str):
        self._logger.critical(message)


# Global logger instance
_logger: Optional[PipelineLogger] = None


def setup_logger(
    log_dir: str = "ml_pipeline_output/logs",
    level: int = logging.INFO
) -> PipelineLogger:
    """Setup and return global logger."""
    global _logger
    _logger = PipelineLogger(log_dir=log_dir, level=level)
    return _logger


def get_logger() -> logging.Logger:
    """Get global logger instance."""
    global _logger
    if _logger is None:
        _logger = PipelineLogger()
    return _logger.logger
