"""
Utilities Module
================

Contains logging, dependency management, decorators, and helper functions
for the Credit Approval ML Pipeline.
"""

import gc
import sys
import os
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from functools import wraps

import numpy as np

from .config import ModelConfig, PLOT_STYLE


# Suppress warnings
warnings.filterwarnings('ignore')
os.environ.update({
    'PYTHONWARNINGS': 'ignore',
    'TF_CPP_MIN_LOG_LEVEL': '3',
    'NUMBA_DISABLE_JIT': '1'
})


# =============================================================================
# ERROR HANDLING DECORATOR
# =============================================================================

def handle_errors(func: Callable) -> Callable:
    """
    Error handling decorator for pipeline functions.
    
    Catches exceptions and logs them appropriately before re-raising.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Try to get logger from instance
            if args and hasattr(args[0], 'logger'):
                args[0].logger.error(f"Error in {func.__name__}: {str(e)}")
            else:
                print(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper


# =============================================================================
# LOGGING SYSTEM
# =============================================================================

class MLPipelineLogger:
    """
    Professional logging system for ML pipeline.
    
    Creates both file and console handlers with formatted output.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup comprehensive logging system."""
        try:
            # Create logs directory
            logs_dir = Path(self.config.output_dir) / self.config.logs_dir
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Create logger
            logger = logging.getLogger('CreditApprovalML')
            logger.setLevel(logging.INFO)
            
            # Clear existing handlers
            logger.handlers.clear()
            
            # File handler
            log_file = logs_dir / f"ml_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            return logger
            
        except Exception as e:
            # Fallback to basic logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger('CreditApprovalML')
            logger.warning(f"Could not setup advanced logging: {e}")
            return logger
    
    def get_logger(self) -> logging.Logger:
        """Get configured logger."""
        return self.logger


# =============================================================================
# DEPENDENCY MANAGER
# =============================================================================

class DependencyManager:
    """
    Manages optional dependencies with graceful fallbacks.
    
    Checks for availability of: scipy, optuna, lightgbm, xgboost, catboost, GPU
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.available_packages: Dict[str, Any] = {}
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check availability of optional dependencies."""
        
        # Check SciPy for statistical tests
        try:
            from scipy import stats
            from scipy.stats import friedmanchisquare
            self.available_packages['scipy'] = stats
            self.logger.info("✅ SciPy available for statistical testing")
        except ImportError:
            self.logger.warning("⚠️ SciPy not available - statistical tests disabled")
            self.available_packages['scipy'] = None
        
        # Check Optuna
        try:
            import optuna
            self.available_packages['optuna'] = optuna
            self.logger.info("✅ Optuna available for hyperparameter optimization")
        except ImportError:
            self.logger.warning("⚠️ Optuna not available - using default parameters")
            self.available_packages['optuna'] = None
        
        # Check LightGBM
        try:
            import lightgbm as lgb
            self.available_packages['lightgbm'] = lgb
            self.logger.info("✅ LightGBM available")
        except ImportError:
            self.logger.warning("⚠️ LightGBM not available")
            self.available_packages['lightgbm'] = None
        
        # Check XGBoost
        try:
            import xgboost as xgb
            self.available_packages['xgboost'] = xgb
            self.logger.info("✅ XGBoost available")
        except ImportError:
            self.logger.warning("⚠️ XGBoost not available")
            self.available_packages['xgboost'] = None
        
        # Check CatBoost
        try:
            import catboost as cb
            self.available_packages['catboost'] = cb
            self.logger.info("✅ CatBoost available")
        except ImportError:
            self.logger.warning("⚠️ CatBoost not available")
            self.available_packages['catboost'] = None
        
        # Check GPU availability
        self._check_gpu()
    
    def _check_gpu(self):
        """Check GPU availability."""
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            self.available_packages['gpu'] = gpu_available
            if gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
                self.logger.info(f"✅ GPU available: {gpu_name}")
            else:
                self.logger.info("ℹ️ GPU not available, using CPU")
        except ImportError:
            # Try nvidia-smi as fallback
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    self.available_packages['gpu'] = True
                    self.logger.info("✅ GPU detected via nvidia-smi")
                else:
                    self.available_packages['gpu'] = False
                    self.logger.info("ℹ️ GPU not available, using CPU")
            except:
                self.available_packages['gpu'] = False
                self.logger.info("ℹ️ Cannot check GPU status, assuming CPU")
    
    def is_available(self, package_name: str) -> bool:
        """Check if a package is available."""
        pkg = self.available_packages.get(package_name)
        if package_name == 'gpu':
            return pkg is True
        return pkg is not None
    
    def get_package(self, package_name: str) -> Any:
        """Get package if available."""
        return self.available_packages.get(package_name)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_output_directories(config: ModelConfig) -> None:
    """
    Setup output directory structure.
    
    Creates all necessary directories for models, plots, results, logs, and final_model.
    """
    directories = [
        config.output_dir,
        f"{config.output_dir}/{config.models_dir}",
        f"{config.output_dir}/{config.plots_dir}",
        f"{config.output_dir}/{config.results_dir}",
        f"{config.output_dir}/{config.logs_dir}",
        f"{config.output_dir}/{config.final_model_dir}"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def memory_cleanup() -> None:
    """
    Clean up memory after heavy operations.
    
    Forces garbage collection and optionally reports memory usage.
    """
    gc.collect()
    
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"🧹 Memory cleanup - Current usage: {memory_info.rss / 1024 / 1024:.1f} MB")
    except ImportError:
        print("🧹 Memory cleanup completed")


def check_environment() -> Dict[str, str]:
    """
    Check environment capabilities and return info dict.
    
    Returns:
        Dict with environment information
    """
    import pandas as pd
    import sklearn
    
    env_info = {
        'python_version': sys.version.split()[0],
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'sklearn_version': sklearn.__version__,
        'environment': 'Unknown'
    }
    
    # Detect environment
    if os.path.exists('/content'):
        env_info['environment'] = 'Google Colab'
    elif os.path.exists('/kaggle'):
        env_info['environment'] = 'Kaggle Notebook'
    else:
        env_info['environment'] = 'Local/Other'
    
    print("\n🔍 ENVIRONMENT CHECK:")
    print(f"   • Python version: {env_info['python_version']}")
    print(f"   • NumPy version: {env_info['numpy_version']}")
    print(f"   • Pandas version: {env_info['pandas_version']}")
    print(f"   • Scikit-learn version: {env_info['sklearn_version']}")
    print(f"   • Environment: {env_info['environment']}")
    
    return env_info


def setup_plotting():
    """Configure matplotlib with professional settings."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(PLOT_STYLE)


# =============================================================================
# PIPELINE INITIALIZATION
# =============================================================================

def initialize_pipeline(config: Optional[ModelConfig] = None) -> tuple:
    """
    Initialize the complete ML pipeline.
    
    Args:
        config: Optional ModelConfig instance. If None, creates default.
    
    Returns:
        Tuple of (config, logger, dependency_manager)
    """
    print("🚀 INITIALIZING ENTERPRISE ML PIPELINE")
    print("=" * 50)
    
    # Check environment
    check_environment()
    
    # Load configuration
    if config is None:
        config = ModelConfig()
    
    # Setup directories
    setup_output_directories(config)
    
    # Setup logging
    logger_manager = MLPipelineLogger(config)
    logger = logger_manager.get_logger()
    
    # Check dependencies
    dependency_manager = DependencyManager(logger)
    
    # Setup matplotlib
    setup_plotting()
    
    # Set random seed for reproducibility
    np.random.seed(config.random_state)
    
    logger.info("✅ Pipeline initialization completed")
    
    return config, logger, dependency_manager
