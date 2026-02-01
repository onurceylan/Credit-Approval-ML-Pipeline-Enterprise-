"""
Data Preprocessor
=================

Target creation, data splitting, and preprocessing pipeline.
"""

import logging
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

from ..core.config import PipelineConfig
from ..core.exceptions import DataValidationError


class TargetCreator:
    """
    Creates target variable using temporal split.
    
    Prevents data leakage by using time-based cutoff.
    """
    
    BAD_STATUSES = ['2', '3', '4', '5']
    TEMPORAL_CUTOFF = -6
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def create_target(
        self,
        app_data: pd.DataFrame,
        credit_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create target variable from credit history.
        
        Uses temporal split to prevent data leakage.
        """
        self.logger.info("🎯 Creating target variable...")
        
        # Split credit data temporally
        observed = credit_data[credit_data['MONTHS_BALANCE'] < self.TEMPORAL_CUTOFF]
        future = credit_data[credit_data['MONTHS_BALANCE'] >= self.TEMPORAL_CUTOFF]
        
        self.logger.info(f"   📊 Observed records: {len(observed):,}")
        self.logger.info(f"   📊 Future records: {len(future):,}")
        
        # Identify customers with bad credit in future
        bad_customers = future[
            future['STATUS'].astype(str).isin(self.BAD_STATUSES)
        ]['ID'].unique()
        
        # Filter to customers with both observed and future data
        valid_ids = set(observed['ID'].unique()) & set(future['ID'].unique())
        self.logger.info(f"   📊 Valid customers: {len(valid_ids):,}")
        
        # Create target
        app_data = app_data[app_data['ID'].isin(valid_ids)].copy()
        app_data['target'] = app_data['ID'].isin(bad_customers).astype(int)
        
        # Log target distribution
        target_dist = app_data['target'].value_counts()
        self.logger.info(f"   📊 Target distribution: 0={target_dist.get(0, 0):,}, 1={target_dist.get(1, 0):,}")
        
        return app_data


class DataSplitter:
    """
    Safe data splitting to prevent leakage.
    
    Performs stratified splits for train/val/test.
    """
    
    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
    
    def split(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Split data into train/val/test sets.
        
        Returns:
            Dictionary with X_train, X_val, X_test, y_train, y_val, y_test
        """
        self.logger.info("✂️ Splitting data...")
        
        if 'target' not in data.columns:
            raise DataValidationError("Target column not found")
        
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Determine stratification
        stratify = y if y.value_counts().min() >= 2 else None
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=stratify
        )
        
        # Second split: train vs val
        val_size_adjusted = self.config.val_size / (1 - self.config.test_size)
        temp_stratify = y_temp if y_temp.value_counts().min() >= 2 else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.random_state,
            stratify=temp_stratify
        )
        
        # Log split sizes
        total = len(data)
        self.logger.info(f"   📊 Train: {len(X_train):,} ({len(X_train)/total*100:.1f}%)")
        self.logger.info(f"   📊 Val: {len(X_val):,} ({len(X_val)/total*100:.1f}%)")
        self.logger.info(f"   📊 Test: {len(X_test):,} ({len(X_test)/total*100:.1f}%)")
        
        # Create CV folds
        cv_folds = self._create_cv_folds(X_train, y_train)
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'cv_folds': cv_folds
        }
    
    def _create_cv_folds(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create cross-validation folds."""
        try:
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
            return list(cv.split(X_train, y_train))
        except ValueError:
            from sklearn.model_selection import KFold
            cv = KFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
            return list(cv.split(X_train))


class DataPreprocessor:
    """
    Complete preprocessing pipeline.
    
    Orchestrates target creation, splitting, and feature engineering.
    """
    
    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.target_creator = TargetCreator(logger)
        self.data_splitter = DataSplitter(config, logger)
    
    def preprocess(
        self,
        app_data: pd.DataFrame,
        credit_data: pd.DataFrame,
        feature_engineer: 'FeatureEngineer'
    ) -> Dict[str, Any]:
        """
        Run complete preprocessing pipeline.
        
        Returns:
            Processed splits with features
        """
        self.logger.info("\n🔄 Running preprocessing pipeline...")
        
        # Create target
        data_with_target = self.target_creator.create_target(app_data, credit_data)
        
        # Split data
        splits = self.data_splitter.split(data_with_target)
        
        # Fit feature engineer on training data only
        feature_engineer.fit(splits['X_train'], splits['y_train'])
        
        # Transform all splits
        splits['X_train'] = feature_engineer.transform(splits['X_train'])
        splits['X_val'] = feature_engineer.transform(splits['X_val'])
        splits['X_test'] = feature_engineer.transform(splits['X_test'])
        
        self.logger.info(f"   ✅ Final features: {len(feature_engineer.final_features)}")
        
        return splits
