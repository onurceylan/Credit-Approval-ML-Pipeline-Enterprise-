"""
Data Validator
==============

Comprehensive data validation and quality checks.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

from ..core.config import PipelineConfig
from ..core.exceptions import DataValidationError


class DataValidator:
    """
    Comprehensive data validation.
    
    Performs:
    - Column validation
    - Type checking
    - Missing value analysis
    - ID overlap validation
    - Data integrity checks
    """
    
    REQUIRED_APP_COLS = ['ID']
    REQUIRED_CREDIT_COLS = ['ID', 'MONTHS_BALANCE', 'STATUS']
    
    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.validation_report: Dict[str, Any] = {}
    
    def validate(
        self,
        app_data: pd.DataFrame,
        credit_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Run all validation checks.
        
        Returns:
            Validation report dictionary
        """
        self.logger.info("🔍 Validating data...")
        
        self._check_required_columns(app_data, credit_data)
        self._check_data_types(app_data, credit_data)
        self._check_empty_data(app_data, credit_data)
        id_overlap = self._check_id_overlap(app_data, credit_data)
        
        self.validation_report = {
            'timestamp': datetime.now().isoformat(),
            'application': self._get_df_stats(app_data),
            'credit': self._get_df_stats(credit_data),
            'id_overlap': id_overlap,
            'valid': True
        }
        
        self._save_report()
        self.logger.info("✅ Data validation passed")
        
        return self.validation_report
    
    def _check_required_columns(
        self,
        app_data: pd.DataFrame,
        credit_data: pd.DataFrame
    ):
        """Check required columns exist."""
        missing_app = [c for c in self.REQUIRED_APP_COLS if c not in app_data.columns]
        missing_credit = [c for c in self.REQUIRED_CREDIT_COLS if c not in credit_data.columns]
        
        if missing_app:
            raise DataValidationError(f"Missing application columns: {missing_app}")
        if missing_credit:
            raise DataValidationError(f"Missing credit columns: {missing_credit}")
    
    def _check_data_types(self, app_data: pd.DataFrame, credit_data: pd.DataFrame):
        """Check data types."""
        if not pd.api.types.is_numeric_dtype(app_data['ID']):
            raise DataValidationError("Application ID must be numeric")
        if not pd.api.types.is_numeric_dtype(credit_data['ID']):
            raise DataValidationError("Credit ID must be numeric")
    
    def _check_empty_data(self, app_data: pd.DataFrame, credit_data: pd.DataFrame):
        """Check for empty dataframes."""
        if app_data.empty:
            raise DataValidationError("Application data is empty")
        if credit_data.empty:
            raise DataValidationError("Credit data is empty")
    
    def _check_id_overlap(
        self,
        app_data: pd.DataFrame,
        credit_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Check ID overlap between datasets."""
        app_ids = set(app_data['ID'].unique())
        credit_ids = set(credit_data['ID'].unique())
        common_ids = app_ids & credit_ids
        
        if not common_ids:
            raise DataValidationError("No common IDs between datasets")
        
        overlap_pct = len(common_ids) / len(app_ids) * 100
        self.logger.info(f"   📊 ID overlap: {len(common_ids):,} ({overlap_pct:.1f}%)")
        
        return {
            'common_ids': len(common_ids),
            'app_only': len(app_ids - credit_ids),
            'credit_only': len(credit_ids - app_ids),
            'overlap_pct': overlap_pct
        }
    
    def _get_df_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get dataframe statistics."""
        return {
            'shape': list(df.shape),
            'columns': list(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'missing_total': int(df.isnull().sum().sum()),
            'dtypes': {str(k): str(v) for k, v in df.dtypes.items()}
        }
    
    def _save_report(self):
        """Save validation report."""
        try:
            report_path = Path(self.config.output_dir) / self.config.results_dir / "data_validation_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(self.validation_report, f, indent=2, default=str)
            self.logger.info(f"   💾 Validation report saved")
        except Exception as e:
            self.logger.warning(f"Could not save validation report: {e}")
