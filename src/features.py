"""
Features Module
===============

Contains classes for safe data splitting, feature engineering, 
preprocessing pipeline, and data quality analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .config import ModelConfig, DataValidationError, OUTPUT_FILES
from .utils import handle_errors


# =============================================================================
# SAFE DATA SPLITTER
# =============================================================================

class SafeDataSplitter:
    """
    Safe data splitting to prevent any data leakage.
    
    Performs splitting BEFORE any preprocessing to ensure
    no information from validation/test leaks into training.
    """
    
    def __init__(self, config: ModelConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    @handle_errors
    def split_data_safely(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Split data safely before any preprocessing.
        
        Args:
            data: Full dataset with 'target' column
        
        Returns:
            Dictionary with train/val/test splits and CV folds
        """
        self.logger.info("⚡ Performing safe data splitting...")
        self.logger.info("🛡️ This prevents ALL data leakage!")
        
        # Separate features and target
        if 'target' not in data.columns:
            raise DataValidationError("Target column not found in data")
        
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Validate target distribution for stratification
        target_counts = y.value_counts()
        min_class_count = target_counts.min()
        
        if min_class_count < 2:
            self.logger.warning(f"   ⚠️ Minimum class has only {min_class_count} samples")
            stratify = None
        else:
            stratify = y
        
        self.logger.info(f"   📊 Total samples: {len(data):,}")
        self.logger.info(f"   📊 Features: {len(X.columns)}")
        
        # First split: train+val vs test
        try:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify
            )
        except ValueError as e:
            self.logger.warning(f"   ⚠️ Stratified split failed: {e}")
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=None
            )
        
        # Second split: train vs val
        val_size_adjusted = self.config.val_size / (1 - self.config.test_size)
        
        # Check if stratification is possible for validation split
        temp_target_counts = y_temp.value_counts()
        temp_min_class = temp_target_counts.min()
        temp_stratify = y_temp if temp_min_class >= 2 else None
        
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=self.config.random_state,
                stratify=temp_stratify
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=self.config.random_state,
                stratify=None
            )
        
        # Create splits dictionary
        splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        # Log split information
        total_size = len(data)
        train_pct = len(X_train) / total_size * 100
        val_pct = len(X_val) / total_size * 100
        test_pct = len(X_test) / total_size * 100
        
        self.logger.info(f"   📏 Split sizes:")
        self.logger.info(f"      • Train: {len(X_train):,} ({train_pct:.1f}%)")
        self.logger.info(f"      • Val:   {len(X_val):,} ({val_pct:.1f}%)")
        self.logger.info(f"      • Test:  {len(X_test):,} ({test_pct:.1f}%)")
        
        # Create cross-validation folds
        splits['cv_folds'] = self._create_cv_folds(X_train, y_train)
        
        return splits
    
    def _create_cv_folds(self, X_train: pd.DataFrame, y_train: pd.Series) -> List:
        """Create cross-validation fold indices."""
        try:
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
            return list(cv.split(X_train, y_train))
        except ValueError:
            self.logger.warning("   ⚠️ Using regular KFold instead of StratifiedKFold")
            kf = KFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
            return list(kf.split(X_train))


# =============================================================================
# ROBUST FEATURE ENGINEER
# =============================================================================

class RobustFeatureEngineer:
    """
    Robust feature engineering with proper fit-transform pattern.
    
    Creates new features from existing ones while preventing data leakage
    by fitting only on training data.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.fitted = False
        self.feature_stats: Dict[str, Any] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        self.original_features: List[str] = []
        self.missing_strategies: Dict[str, Any] = {}
        self.outlier_bounds: Dict[str, tuple] = {}
        self.scaler: Optional[StandardScaler] = None
    
    @handle_errors
    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit feature engineering on training data and transform.
        
        Args:
            X_train: Training features DataFrame
        
        Returns:
            Transformed training DataFrame
        """
        self.logger.info("🔧 Fitting feature engineering on training data...")
        
        X_processed = X_train.copy()
        
        # Store original feature names
        self.original_features = list(X_train.columns)
        
        # 1. Handle missing values
        X_processed = self._handle_missing_values(X_processed, fit=True)
        
        # 2. Create new features BEFORE encoding
        X_processed = self._create_new_features(X_processed, fit=True)
        
        # 3. Encode categorical variables
        X_processed = self._encode_categorical_variables(X_processed, fit=True)
        
        # 4. Handle outliers
        X_processed = self._handle_outliers(X_processed, fit=True)
        
        # 5. Scale numerical features
        X_processed = self._scale_numerical_features(X_processed, fit=True)
        
        self.fitted = True
        self.feature_names = list(X_processed.columns)
        
        self.logger.info(f"   ✅ Feature engineering completed!")
        self.logger.info(f"   📊 Features: {len(self.original_features)} → {len(self.feature_names)}")
        
        return X_processed
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted parameters.
        
        Args:
            X: Features DataFrame to transform
        
        Returns:
            Transformed DataFrame
        """
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        X_processed = X.copy()
        
        # Apply same transformations in same order
        X_processed = self._handle_missing_values(X_processed, fit=False)
        X_processed = self._create_new_features(X_processed, fit=False)
        X_processed = self._encode_categorical_variables(X_processed, fit=False)
        X_processed = self._handle_outliers(X_processed, fit=False)
        X_processed = self._scale_numerical_features(X_processed, fit=False)
        
        return X_processed
    
    def _handle_missing_values(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Handle missing values appropriately."""
        if fit:
            self.missing_strategies = {}
            
            for col in X.columns:
                if X[col].dtype.name == 'category':
                    mode_val = X[col].mode()
                    fill_value = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                    
                    if fill_value == 'Unknown' and 'Unknown' not in X[col].cat.categories:
                        X[col] = X[col].cat.add_categories(['Unknown'])
                    
                    self.missing_strategies[col] = fill_value
                    
                elif X[col].dtype == 'object':
                    mode_val = X[col].mode()
                    self.missing_strategies[col] = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                    
                else:
                    self.missing_strategies[col] = X[col].median()
        
        # Apply missing value strategies
        for col, fill_value in self.missing_strategies.items():
            if col in X.columns:
                if X[col].dtype.name == 'category':
                    if fill_value not in X[col].cat.categories:
                        X[col] = X[col].cat.add_categories([fill_value])
                    X[col] = X[col].fillna(fill_value)
                else:
                    X[col] = X[col].fillna(fill_value)
        
        return X
    
    def _create_new_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Create new features from existing ones."""
        X_new = X.copy()
        
        # 1. Age-related features
        if 'DAYS_BIRTH' in X.columns:
            X_new['AGE_YEARS'] = (-X['DAYS_BIRTH']) / 365.25
            X_new['AGE_YEARS'] = X_new['AGE_YEARS'].clip(lower=18, upper=100)
            
            X_new['AGE_GROUP'] = pd.cut(
                X_new['AGE_YEARS'],
                bins=[0, 25, 35, 45, 55, 65, 100],
                labels=['Young', 'Adult', 'Middle_Early', 'Middle_Late', 'Senior', 'Elder']
            )
            
            age_group_mapping = {
                'Young': 0, 'Adult': 1, 'Middle_Early': 2,
                'Middle_Late': 3, 'Senior': 4, 'Elder': 5
            }
            X_new['AGE_GROUP'] = X_new['AGE_GROUP'].map(age_group_mapping).fillna(1)
        
        # 2. Employment-related features
        if 'DAYS_EMPLOYED' in X.columns:
            unemployed_threshold = 100000
            is_unemployed = X['DAYS_EMPLOYED'] > unemployed_threshold
            
            employment_days = X['DAYS_EMPLOYED'].copy()
            employment_days[is_unemployed] = 0
            employment_days[~is_unemployed] = -employment_days[~is_unemployed]
            
            X_new['EMPLOYMENT_YEARS'] = employment_days / 365.25
            X_new['EMPLOYMENT_YEARS'] = X_new['EMPLOYMENT_YEARS'].clip(lower=0, upper=50)
            
            X_new['IS_EMPLOYED'] = (~is_unemployed).astype(int)
            
            X_new['EMPLOYMENT_CATEGORY'] = pd.cut(
                X_new['EMPLOYMENT_YEARS'],
                bins=[0, 1, 3, 5, 10, 50],
                labels=['New', 'Short', 'Medium', 'Long', 'Very_Long']
            )
            
            emp_cat_mapping = {'New': 0, 'Short': 1, 'Medium': 2, 'Long': 3, 'Very_Long': 4}
            X_new['EMPLOYMENT_CATEGORY'] = X_new['EMPLOYMENT_CATEGORY'].map(emp_cat_mapping).fillna(0)
        
        # 3. Income-related features
        if 'AMT_INCOME_TOTAL' in X.columns and 'CNT_FAM_MEMBERS' in X.columns:
            family_size = X['CNT_FAM_MEMBERS'].fillna(1).clip(lower=1)
            X_new['INCOME_PER_FAMILY_MEMBER'] = X['AMT_INCOME_TOTAL'] / family_size
            X_new['LOG_INCOME'] = np.log1p(X['AMT_INCOME_TOTAL'])
        
        # 4. Binary features for asset ownership
        binary_cols = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_MOBIL',
                       'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL']
        
        for col in binary_cols:
            if col in X_new.columns:
                if X_new[col].dtype == 'object':
                    X_new[col] = X_new[col].map({'Y': 1, 'N': 0}).fillna(0)
                else:
                    X_new[col] = X_new[col].fillna(0)
                X_new[col] = X_new[col].astype(int)
        
        # 5. Composite features
        asset_cols = [col for col in binary_cols if col in X_new.columns]
        if asset_cols:
            X_new['TOTAL_ASSETS'] = X_new[asset_cols].sum(axis=1)
            X_new['ASSET_DIVERSITY'] = (X_new[asset_cols] > 0).sum(axis=1)
        
        # 6. Family-related features
        if 'CNT_CHILDREN' in X.columns and 'CNT_FAM_MEMBERS' in X.columns:
            X_new['CNT_CHILDREN'] = X_new['CNT_CHILDREN'].fillna(0).clip(lower=0, upper=10)
            
            adults_in_family = X_new['CNT_FAM_MEMBERS'] - X_new['CNT_CHILDREN']
            X_new['CNT_ADULTS'] = adults_in_family.clip(lower=1)
            
            X_new['HAS_CHILDREN'] = (X_new['CNT_CHILDREN'] > 0).astype(int)
            X_new['LARGE_FAMILY'] = (X_new['CNT_FAM_MEMBERS'] > 4).astype(int)
        
        # 7. Age-income interaction
        if 'AGE_YEARS' in X_new.columns and 'AMT_INCOME_TOTAL' in X.columns:
            X_new['INCOME_PER_AGE'] = X_new['AMT_INCOME_TOTAL'] / X_new['AGE_YEARS'].clip(lower=18)
        
        # 8. Disposable income estimate
        if 'AMT_INCOME_TOTAL' in X.columns and 'CNT_FAM_MEMBERS' in X.columns:
            estimated_monthly_expense_per_person = 500
            estimated_family_expenses = X_new['CNT_FAM_MEMBERS'] * estimated_monthly_expense_per_person * 12
            
            X_new['DISPOSABLE_INCOME_ESTIMATE'] = (X_new['AMT_INCOME_TOTAL'] - estimated_family_expenses).clip(lower=0)
            X_new['DISPOSABLE_INCOME_RATIO'] = X_new['DISPOSABLE_INCOME_ESTIMATE'] / X_new['AMT_INCOME_TOTAL'].clip(lower=1)
        
        return X_new
    
    def _encode_categorical_variables(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical variables."""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if fit:
                if X[col].dtype.name == 'category':
                    X[col] = X[col].astype(str)
                
                le = LabelEncoder()
                X[col] = X[col].fillna('Unknown')
                
                try:
                    le.fit(X[col])
                    self.encoders[col] = le
                except Exception as e:
                    self.logger.warning(f"   ⚠️ Could not encode column {col}: {e}")
                    X = X.drop(col, axis=1)
                    continue
            
            if col in self.encoders and col in X.columns:
                if X[col].dtype.name == 'category':
                    X[col] = X[col].astype(str)
                
                X[col] = X[col].fillna('Unknown')
                
                encoder = self.encoders[col]
                X[col] = X[col].apply(
                    lambda x: x if x in encoder.classes_ else 'Unknown'
                )
                
                try:
                    X[col] = encoder.transform(X[col])
                except Exception as e:
                    self.logger.warning(f"   ⚠️ Could not transform column {col}: {e}")
                    X[col] = encoder.transform(['Unknown'])[0]
        
        return X
    
    def _handle_outliers(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Handle outliers using IQR method."""
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if fit:
            self.outlier_bounds = {}
            
            for col in numerical_cols:
                try:
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    if 'INCOME' in col.upper() or 'AMT_' in col.upper():
                        multiplier = 3.0
                    else:
                        multiplier = 1.5
                    
                    lower_bound = Q1 - multiplier * IQR
                    upper_bound = Q3 + multiplier * IQR
                    
                    self.outlier_bounds[col] = (lower_bound, upper_bound)
                    
                except Exception as e:
                    self.logger.warning(f"   ⚠️ Could not calculate outlier bounds for {col}: {e}")
                    self.outlier_bounds[col] = (X[col].min(), X[col].max())
        
        for col, (lower, upper) in self.outlier_bounds.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=upper)
        
        return X
    
    def _scale_numerical_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scale numerical features using StandardScaler."""
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return X
        
        if fit:
            self.scaler = StandardScaler()
            try:
                X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
            except Exception as e:
                self.logger.warning(f"   ⚠️ Scaling failed: {e}")
                self.scaler = None
        else:
            if self.scaler is not None:
                try:
                    X[numerical_cols] = self.scaler.transform(X[numerical_cols])
                except Exception as e:
                    self.logger.warning(f"   ⚠️ Transform scaling failed: {e}")
        
        return X
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get comprehensive feature information."""
        return {
            'original_features': self.original_features,
            'final_features': self.feature_names,
            'feature_count_change': len(self.feature_names) - len(self.original_features),
            'encoding_info': {col: list(encoder.classes_) for col, encoder in self.encoders.items()},
            'missing_strategies': self.missing_strategies,
            'outlier_bounds': {k: list(v) for k, v in self.outlier_bounds.items()},
            'new_features_created': [f for f in self.feature_names if f not in self.original_features]
        }


# =============================================================================
# DATA PREPROCESSING PIPELINE
# =============================================================================

class DataPreprocessingPipeline:
    """
    Complete data preprocessing pipeline.
    
    Orchestrates feature engineering and data transformation
    while maintaining proper fit-transform separation.
    """
    
    def __init__(self, config: ModelConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.feature_engineer = RobustFeatureEngineer(logger)
        self.preprocessing_info: Dict[str, Any] = {}
    
    @handle_errors
    def preprocess_data(self, splits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply complete preprocessing pipeline.
        
        Args:
            splits: Dictionary with train/val/test splits
        
        Returns:
            Processed splits dictionary
        """
        self.logger.info("🔄 Applying complete preprocessing pipeline...")
        
        X_train, X_val, X_test = splits['X_train'], splits['X_val'], splits['X_test']
        y_train, y_val, y_test = splits['y_train'], splits['y_val'], splits['y_test']
        
        self.logger.info(f"   📊 Input shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Fit and transform training data
        X_train_processed = self.feature_engineer.fit_transform(X_train)
        
        # Transform validation and test data
        X_val_processed = self.feature_engineer.transform(X_val)
        X_test_processed = self.feature_engineer.transform(X_test)
        
        self.logger.info(f"   📊 Output shapes - Train: {X_train_processed.shape}, Val: {X_val_processed.shape}, Test: {X_test_processed.shape}")
        
        # Store preprocessing info
        self.preprocessing_info = self.feature_engineer.get_feature_info()
        
        # Create processed splits
        processed_splits = {
            'X_train': X_train_processed,
            'X_val': X_val_processed,
            'X_test': X_test_processed,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'cv_folds': splits['cv_folds'],
            'feature_engineer': self.feature_engineer,
            'preprocessing_info': self.preprocessing_info
        }
        
        # Log results
        self.logger.info(f"   ✅ Preprocessing completed!")
        self.logger.info(f"   🔧 Feature engineering: {len(self.preprocessing_info['original_features'])} → {len(self.preprocessing_info['final_features'])} features")
        
        new_features = self.preprocessing_info['new_features_created']
        if new_features:
            self.logger.info(f"   ✨ New features created: {len(new_features)}")
            for feature in new_features[:5]:
                self.logger.info(f"      • {feature}")
            if len(new_features) > 5:
                self.logger.info(f"      • ... and {len(new_features) - 5} more")
        
        # Save preprocessing info
        self._save_preprocessing_info()
        
        return processed_splits
    
    def _save_preprocessing_info(self):
        """Save preprocessing information for reproducibility."""
        try:
            info_path = Path(self.config.output_dir) / self.config.results_dir / OUTPUT_FILES['preprocessing_info']
            info_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(info_path, 'w') as f:
                json.dump(self.preprocessing_info, f, indent=2, default=str)
            self.logger.info(f"   💾 Preprocessing info saved: {info_path}")
        except Exception as e:
            self.logger.warning(f"   ⚠️ Could not save preprocessing info: {e}")


# =============================================================================
# DATA QUALITY ANALYZER
# =============================================================================

class DataQualityAnalyzer:
    """
    Analyzes data quality throughout the preprocessing pipeline.
    
    Checks for data leakage, missing values, infinite values,
    and class distribution balance.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def analyze_data_quality(
        self, 
        original_data: pd.DataFrame, 
        processed_splits: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive data quality analysis.
        
        Args:
            original_data: Original unprocessed data
            processed_splits: Processed data splits
        
        Returns:
            Quality report dictionary
        """
        self.logger.info("📊 Analyzing data quality...")
        
        quality_report = {
            'original_shape': list(original_data.shape),
            'processed_train_shape': list(processed_splits['X_train'].shape),
            'processed_val_shape': list(processed_splits['X_val'].shape),
            'processed_test_shape': list(processed_splits['X_test'].shape),
            'missing_values': self._check_missing_values(processed_splits),
            'infinite_values': self._check_infinite_values(processed_splits),
            'memory_usage': self._calculate_memory_usage(processed_splits),
            'class_distribution': self._check_class_distribution(processed_splits),
            'data_leakage_check': self._check_data_leakage(processed_splits),
            'feature_info': processed_splits.get('preprocessing_info', {})
        }
        
        # Log findings
        self._log_quality_findings(quality_report)
        
        return quality_report
    
    def _check_missing_values(self, splits: Dict[str, Any]) -> int:
        """Check for remaining missing values."""
        total_missing = 0
        for key in ['X_train', 'X_val', 'X_test']:
            if key in splits:
                total_missing += splits[key].isnull().sum().sum()
        return total_missing
    
    def _check_infinite_values(self, splits: Dict[str, Any]) -> int:
        """Check for infinite values."""
        total_inf = 0
        for key in ['X_train', 'X_val', 'X_test']:
            if key in splits:
                df = splits[key]
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    total_inf += np.isinf(df[col]).sum()
        return total_inf
    
    def _calculate_memory_usage(self, splits: Dict[str, Any]) -> float:
        """Calculate total memory usage in MB."""
        total_memory = 0
        for key in ['X_train', 'X_val', 'X_test']:
            if key in splits:
                total_memory += splits[key].memory_usage(deep=True).sum()
        return total_memory / 1024 / 1024
    
    def _check_class_distribution(self, splits: Dict[str, Any]) -> Dict[str, Dict]:
        """Check class distribution across splits."""
        distributions = {}
        for key in ['y_train', 'y_val', 'y_test']:
            if key in splits:
                dist = splits[key].value_counts(normalize=True).to_dict()
                distributions[key] = {str(k): v for k, v in dist.items()}
        return distributions
    
    def _check_data_leakage(self, splits: Dict[str, Any]) -> Dict[str, bool]:
        """Check for potential data leakage."""
        leakage_check = {
            'train_val_overlap': False,
            'train_test_overlap': False,
            'val_test_overlap': False
        }
        
        # Check index overlap
        if 'X_train' in splits and 'X_val' in splits:
            train_idx = set(splits['X_train'].index)
            val_idx = set(splits['X_val'].index)
            leakage_check['train_val_overlap'] = bool(train_idx & val_idx)
        
        if 'X_train' in splits and 'X_test' in splits:
            train_idx = set(splits['X_train'].index)
            test_idx = set(splits['X_test'].index)
            leakage_check['train_test_overlap'] = bool(train_idx & test_idx)
        
        if 'X_val' in splits and 'X_test' in splits:
            val_idx = set(splits['X_val'].index)
            test_idx = set(splits['X_test'].index)
            leakage_check['val_test_overlap'] = bool(val_idx & test_idx)
        
        return leakage_check
    
    def _log_quality_findings(self, report: Dict[str, Any]):
        """Log quality analysis findings."""
        feature_info = report.get('feature_info', {})
        
        self.logger.info(f"   📊 Data Quality Summary:")
        self.logger.info(f"      • Missing values remaining: {report['missing_values']}")
        self.logger.info(f"      • Infinite values: {report['infinite_values']}")
        self.logger.info(f"      • Memory usage: {report['memory_usage']:.2f} MB")
        self.logger.info(f"      • Features: {len(feature_info.get('original_features', []))} → {len(feature_info.get('final_features', []))}")
        self.logger.info(f"      • New features created: {len(feature_info.get('new_features_created', []))}")
        
        # Check for leakage
        leakage = report['data_leakage_check']
        if any(leakage.values()):
            self.logger.warning("   ⚠️ Potential data leakage detected!")
        else:
            self.logger.info("   ✅ No data leakage detected")
        
        # Check class balance
        class_dist = report['class_distribution']
        if class_dist:
            train_dist = class_dist.get('y_train', {})
            val_dist = class_dist.get('y_val', {})
            
            if train_dist and val_dist:
                max_diff = max(abs(train_dist.get(k, 0) - val_dist.get(k, 0)) 
                              for k in set(train_dist) | set(val_dist))
                if max_diff > 0.1:
                    self.logger.warning(f"   ⚠️ Class distribution imbalance: {max_diff:.2%}")
                else:
                    self.logger.info("   ✅ Class distributions are well balanced across splits")
