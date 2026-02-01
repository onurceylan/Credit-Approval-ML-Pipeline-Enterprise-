"""
Data Loader
===========

Robust data loading with multi-environment support.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging

import pandas as pd

from ..core.config import PipelineConfig
from ..core.exceptions import DataValidationError


class DataLoader:
    """
    Robust data loading with validation.
    
    Supports Kaggle, Colab, and Local environments.
    """
    
    # Environment-specific path mappings
    ENV_PATHS = {
        'colab_drive': {
            'application': '/content/drive/MyDrive/credit-approval/data/raw/application_record.csv',
            'credit': '/content/drive/MyDrive/credit-approval/data/raw/credit_record.csv'
        },
        'colab_local': {
            'application': '/content/application_record.csv',
            'credit': '/content/credit_record.csv'
        },
        'kaggle': {
            'application': '/kaggle/input/credit-card-approval-prediction/application_record.csv',
            'credit': '/kaggle/input/credit-card-approval-prediction/credit_record.csv'
        },
        'local': {
            'application': 'data/raw/application_record.csv',
            'credit': 'data/raw/credit_record.csv'
        }
    }
    
    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._data_paths = self._detect_data_paths()
    
    def _detect_data_paths(self) -> Dict[str, str]:
        """Auto-detect data paths based on environment."""
        for env_name, paths in self.ENV_PATHS.items():
            if all(Path(p).exists() for p in paths.values()):
                self.logger.info(f"✅ Data found in: {env_name}")
                return paths
        
        # Fallback to config paths
        return {
            'application': f"{self.config.raw_data_dir}/{self.config.application_file}",
            'credit': f"{self.config.raw_data_dir}/{self.config.credit_file}"
        }
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load application and credit data.
        
        Returns:
            Tuple of (application_df, credit_df)
        """
        self.logger.info("📥 Loading data...")
        
        # Load application data
        app_path = self._data_paths['application']
        if not Path(app_path).exists():
            raise DataValidationError(f"Application data not found: {app_path}")
        
        app_data = pd.read_csv(app_path)
        self.logger.info(f"   📊 Application data: {app_data.shape}")
        
        # Load credit data
        credit_path = self._data_paths['credit']
        if not Path(credit_path).exists():
            raise DataValidationError(f"Credit data not found: {credit_path}")
        
        credit_data = pd.read_csv(credit_path)
        self.logger.info(f"   📊 Credit data: {credit_data.shape}")
        
        return app_data, credit_data
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about loaded data."""
        return {
            'paths': self._data_paths,
            'environment': self._detect_environment()
        }
    
    def _detect_environment(self) -> str:
        """Detect current environment."""
        if Path('/content').exists():
            return 'colab'
        elif Path('/kaggle').exists():
            return 'kaggle'
        return 'local'
