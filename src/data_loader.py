"""
Data Loader Module
==================

Contains classes for robust data loading, validation, and temporal splitting
for the Credit Approval ML Pipeline.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any

import pandas as pd

from .config import ModelConfig, DataValidationError, BAD_CREDIT_STATUSES, TEMPORAL_CUTOFF_MONTHS, OUTPUT_FILES
from .utils import handle_errors


# =============================================================================
# ROBUST DATA LOADER
# =============================================================================

class RobustDataLoader:
    """
    Robust data loading with validation and error handling.
    
    Handles multiple data sources, validates data quality,
    and generates comprehensive validation reports.
    """
    
    def __init__(self, config: ModelConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.validation_report: Dict[str, Any] = {}
    
    @handle_errors
    def load_and_validate_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and validate input data.
        
        Returns:
            Tuple of (application_data, credit_data) DataFrames
        """
        self.logger.info("📥 Loading and validating data...")
        
        # Load application data
        app_data = self._load_application_data()
        
        # Load credit data
        credit_data = self._load_credit_data()
        
        # Validate data
        self._validate_data_quality(app_data, credit_data)
        
        # Generate validation report
        self._generate_validation_report(app_data, credit_data)
        
        return app_data, credit_data
    
    def _load_application_data(self) -> pd.DataFrame:
        """Load application data with error handling."""
        app_path = self.config.data_paths['application']
        
        if not Path(app_path).exists():
            raise DataValidationError(f"Application data not found: {app_path}")
        
        try:
            app_data = pd.read_csv(app_path)
            self.logger.info(f"   📊 Application data loaded: {app_data.shape}")
            return app_data
        except Exception as e:
            raise DataValidationError(f"Failed to load application data: {str(e)}")
    
    def _load_credit_data(self) -> pd.DataFrame:
        """Load credit data with error handling."""
        credit_path = self.config.data_paths['credit']
        
        if not Path(credit_path).exists():
            raise DataValidationError(f"Credit data not found: {credit_path}")
        
        try:
            credit_data = pd.read_csv(credit_path)
            self.logger.info(f"   📊 Credit data loaded: {credit_data.shape}")
            return credit_data
        except Exception as e:
            raise DataValidationError(f"Failed to load credit data: {str(e)}")
    
    def _validate_data_quality(self, app_data: pd.DataFrame, credit_data: pd.DataFrame):
        """Comprehensive data quality validation."""
        self.logger.info("🔍 Performing data quality validation...")
        
        # Check required columns
        self._check_required_columns(app_data, credit_data)
        
        # Check data types
        self._check_data_types(app_data, credit_data)
        
        # Check for empty data
        self._check_empty_data(app_data, credit_data)
        
        # Check ID overlap
        self._check_id_overlap(app_data, credit_data)
        
        # Check for data integrity issues
        self._check_data_integrity(app_data, credit_data)
        
        self.logger.info("✅ Data quality validation completed")
    
    def _check_required_columns(self, app_data: pd.DataFrame, credit_data: pd.DataFrame):
        """Check for required columns."""
        required_app_cols = ['ID']
        required_credit_cols = ['ID', 'MONTHS_BALANCE', 'STATUS']
        
        missing_app_cols = [col for col in required_app_cols if col not in app_data.columns]
        missing_credit_cols = [col for col in required_credit_cols if col not in credit_data.columns]
        
        if missing_app_cols:
            raise DataValidationError(f"Missing application columns: {missing_app_cols}")
        
        if missing_credit_cols:
            raise DataValidationError(f"Missing credit columns: {missing_credit_cols}")
    
    def _check_data_types(self, app_data: pd.DataFrame, credit_data: pd.DataFrame):
        """Check data types."""
        if not pd.api.types.is_numeric_dtype(app_data['ID']):
            raise DataValidationError("Application ID must be numeric")
        
        if not pd.api.types.is_numeric_dtype(credit_data['ID']):
            raise DataValidationError("Credit ID must be numeric")
    
    def _check_empty_data(self, app_data: pd.DataFrame, credit_data: pd.DataFrame):
        """Check for empty data."""
        if app_data.empty:
            raise DataValidationError("Application data is empty")
        
        if credit_data.empty:
            raise DataValidationError("Credit data is empty")
    
    def _check_id_overlap(self, app_data: pd.DataFrame, credit_data: pd.DataFrame):
        """Check for ID overlap between datasets."""
        app_ids = set(app_data['ID'].unique())
        credit_ids = set(credit_data['ID'].unique())
        common_ids = app_ids & credit_ids
        
        if not common_ids:
            raise DataValidationError("No common IDs between application and credit data")
        
        overlap_pct = len(common_ids) / len(app_ids) * 100
        self.logger.info(f"   📊 ID overlap: {len(common_ids):,} IDs ({overlap_pct:.1f}%)")
    
    def _check_data_integrity(self, app_data: pd.DataFrame, credit_data: pd.DataFrame):
        """Check data integrity issues."""
        # Check for duplicate IDs in application data
        app_duplicates = app_data['ID'].duplicated().sum()
        if app_duplicates > 0:
            self.logger.warning(f"⚠️ Found {app_duplicates} duplicate application IDs")
        
        # Check for missing values
        app_missing = app_data.isnull().sum().sum()
        credit_missing = credit_data.isnull().sum().sum()
        
        if app_missing > 0:
            self.logger.info(f"   📊 Application missing values: {app_missing:,}")
        
        if credit_missing > 0:
            self.logger.info(f"   📊 Credit missing values: {credit_missing:,}")
    
    def _generate_validation_report(self, app_data: pd.DataFrame, credit_data: pd.DataFrame):
        """Generate comprehensive validation report."""
        self.validation_report = {
            'timestamp': datetime.now().isoformat(),
            'application_data': {
                'shape': list(app_data.shape),
                'columns': list(app_data.columns),
                'memory_usage_mb': app_data.memory_usage(deep=True).sum() / 1024**2,
                'missing_values': app_data.isnull().sum().to_dict(),
                'dtypes': app_data.dtypes.astype(str).to_dict()
            },
            'credit_data': {
                'shape': list(credit_data.shape),
                'columns': list(credit_data.columns),
                'memory_usage_mb': credit_data.memory_usage(deep=True).sum() / 1024**2,
                'missing_values': credit_data.isnull().sum().to_dict(),
                'dtypes': credit_data.dtypes.astype(str).to_dict()
            },
            'data_quality_checks': {
                'id_overlap': len(set(app_data['ID']) & set(credit_data['ID'])),
                'app_duplicates': int(app_data['ID'].duplicated().sum()),
                'credit_records_per_id': credit_data.groupby('ID').size().describe().to_dict()
            }
        }
        
        # Save validation report
        self._save_report()
    
    def _save_report(self):
        """Save validation report to JSON."""
        try:
            report_path = Path(self.config.output_dir) / self.config.results_dir / OUTPUT_FILES['validation_report']
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(self.validation_report, f, indent=2, default=str)
            self.logger.info(f"   💾 Validation report saved: {report_path}")
        except Exception as e:
            self.logger.warning(f"   ⚠️ Could not save validation report: {e}")


# =============================================================================
# TEMPORAL DATA SPLITTER
# =============================================================================

class TemporalDataSplitter:
    """
    Handles temporal data splitting to prevent data leakage.
    
    Uses historical credit data to create target variable,
    ensuring no future information leaks into training.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    @handle_errors
    def create_target_variable(
        self, 
        app_data: pd.DataFrame, 
        credit_data: pd.DataFrame,
        cutoff_months: int = TEMPORAL_CUTOFF_MONTHS
    ) -> pd.DataFrame:
        """
        Create target variable using temporal split to prevent leakage.
        
        Args:
            app_data: Application data DataFrame
            credit_data: Credit history DataFrame
            cutoff_months: Months before current to use as cutoff (negative)
        
        Returns:
            Merged DataFrame with target variable
        """
        self.logger.info("🎯 Creating target variable with temporal split...")
        self.logger.info(f"   📅 Using credit history before month {cutoff_months}")
        
        # Use only historical credit data
        historical_credit = credit_data[credit_data['MONTHS_BALANCE'] <= cutoff_months].copy()
        
        self.logger.info(f"   📊 Historical records: {len(historical_credit):,}")
        self.logger.info(f"   📊 Future records excluded: {len(credit_data) - len(historical_credit):,}")
        
        # Create binary target for bad credit
        historical_credit['is_bad_credit'] = historical_credit['STATUS'].apply(
            lambda x: 1 if str(x) in BAD_CREDIT_STATUSES else 0
        )
        
        # Aggregate by ID
        target_data = historical_credit.groupby('ID').agg({
            'is_bad_credit': 'max',  # 1 if ever had bad credit
            'MONTHS_BALANCE': 'count'  # Number of historical records
        }).reset_index()
        
        target_data.columns = ['ID', 'target', 'credit_history_length']
        
        # Merge with application data
        merged_data = app_data.merge(target_data, on='ID', how='left')
        
        # Handle missing credit history
        # 0 = Good Credit, 1 = Bad Credit, 2 = No History
        merged_data['target'] = merged_data['target'].fillna(2).astype(int)
        merged_data['credit_history_length'] = merged_data['credit_history_length'].fillna(0)
        
        # Remove ID column for modeling
        if 'ID' in merged_data.columns:
            merged_data = merged_data.drop('ID', axis=1)
        
        # Log target distribution
        target_dist = merged_data['target'].value_counts().sort_index()
        self.logger.info("   📊 Target distribution:")
        self.logger.info(f"      • Good Credit (0): {target_dist.get(0, 0):,}")
        self.logger.info(f"      • Bad Credit (1): {target_dist.get(1, 0):,}")
        self.logger.info(f"      • No History (2): {target_dist.get(2, 0):,}")
        
        return merged_data
