"""
Feature Engineer
=================

Robust feature engineering with fit-transform pattern.
"""

import logging
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from ..core.config import PipelineConfig
from ..core.exceptions import FeatureEngineeringError


class FeatureEngineer:
    """
    Feature engineering with fit-transform pattern.
    
    Creates derived features and ensures no data leakage.
    """
    
    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.is_fitted = False
        self.feature_stats: Dict[str, Any] = {}
        self.final_features: List[str] = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineer':
        """
        Fit feature engineering on training data.
        
        Args:
            X: Training features
            y: Training target (optional)
        """
        self.logger.info("🔧 Fitting feature engineer...")
        
        X_transformed = self._create_features(X.copy())
        
        # Identify column types
        numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_transformed.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Fit scalers
        for col in numeric_cols:
            scaler = StandardScaler()
            valid_data = X_transformed[col].dropna()
            if len(valid_data) > 0:
                scaler.fit(valid_data.values.reshape(-1, 1))
                self.scalers[col] = scaler
                self.feature_stats[col] = {
                    'mean': float(valid_data.mean()),
                    'std': float(valid_data.std()),
                    'min': float(valid_data.min()),
                    'max': float(valid_data.max())
                }
        
        # Fit encoders
        for col in categorical_cols:
            encoder = LabelEncoder()
            valid_data = X_transformed[col].dropna().astype(str)
            if len(valid_data) > 0:
                encoder.fit(valid_data)
                self.encoders[col] = encoder
        
        self.final_features = numeric_cols + categorical_cols
        self.is_fitted = True
        self.logger.info(f"   ✅ Fitted on {len(self.final_features)} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted parameters.
        
        Args:
            X: Features to transform
        
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise FeatureEngineeringError("FeatureEngineer not fitted. Call fit() first.")
        
        X_transformed = self._create_features(X.copy())
        
        # Scale numerical columns
        for col, scaler in self.scalers.items():
            if col in X_transformed.columns:
                valid_idx = X_transformed[col].notna()
                if valid_idx.any():
                    values = X_transformed.loc[valid_idx, col].values.reshape(-1, 1)
                    X_transformed.loc[valid_idx, col] = scaler.transform(values).flatten()
        
        # Encode categorical columns
        for col, encoder in self.encoders.items():
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].fillna('Unknown').astype(str)
                try:
                    X_transformed[col] = encoder.transform(X_transformed[col])
                except ValueError:
                    # Handle unseen categories
                    X_transformed[col] = X_transformed[col].apply(
                        lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                    )
        
        # Fill remaining NaN
        X_transformed = X_transformed.fillna(0)
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features."""
        
        # Age features
        if 'DAYS_BIRTH' in df.columns:
            df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365).astype(int)
            df['AGE_GROUP'] = pd.cut(
                df['AGE_YEARS'],
                bins=[0, 25, 35, 45, 55, 65, 100],
                labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
            )
        
        # Employment features
        if 'DAYS_EMPLOYED' in df.columns:
            df['EMPLOYED_YEARS'] = df['DAYS_EMPLOYED'].apply(
                lambda x: 0 if x > 0 else int(-x / 365)
            )
            df['IS_EMPLOYED'] = (df['DAYS_EMPLOYED'] < 0).astype(int)
        
        # Income features
        if 'AMT_INCOME_TOTAL' in df.columns:
            income_median = df['AMT_INCOME_TOTAL'].median()
            df['INCOME_LOG'] = np.log1p(df['AMT_INCOME_TOTAL'])
            df['INCOME_CATEGORY'] = pd.cut(
                df['AMT_INCOME_TOTAL'],
                bins=[0, 50000, 100000, 200000, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
        
        # Family features
        if 'CNT_CHILDREN' in df.columns and 'CNT_FAM_MEMBERS' in df.columns:
            df['HAS_CHILDREN'] = (df['CNT_CHILDREN'] > 0).astype(int)
            df['FAMILY_SIZE'] = df['CNT_FAM_MEMBERS'].fillna(1)
        
        # Composite features
        if 'AMT_INCOME_TOTAL' in df.columns and 'CNT_FAM_MEMBERS' in df.columns:
            family_members = df['CNT_FAM_MEMBERS'].replace(0, 1)
            df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / family_members
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of final feature names."""
        return self.final_features
    
    def get_feature_stats(self) -> Dict[str, Any]:
        """Get feature statistics."""
        return self.feature_stats
