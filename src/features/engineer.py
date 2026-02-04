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
        """Fit feature engineering on training data."""
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
        """Transform features using fitted parameters."""
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
                X_transformed[col] = X_transformed[col].astype(object).fillna('Unknown').astype(str)
                try:
                    X_transformed[col] = encoder.transform(X_transformed[col])
                except ValueError:
                    X_transformed[col] = X_transformed[col].apply(
                        lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                    )
        
        X_transformed = X_transformed.fillna(0)
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features matching V3.5 Enterprise logic."""
        
        # 1. Temporal & Age Features
        if 'DAYS_BIRTH' in df.columns:
            df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365.25).astype(float)
            df['AGE_GROUP'] = pd.cut(
                df['AGE_YEARS'],
                bins=[0, 25, 35, 45, 55, 65, 120],
                labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
            ).astype(str)
        
        # 2. Employment & Stability
        if 'DAYS_EMPLOYED' in df.columns:
            # Handle anomalous 365243 value (retired/unemployed)
            df['DAYS_EMPLOYED_CLEAN'] = df['DAYS_EMPLOYED'].replace(365243, 0)
            df['EMPLOYED_YEARS'] = (-df['DAYS_EMPLOYED_CLEAN'] / 365.25).astype(float)
            df['IS_EMPLOYED'] = (df['DAYS_EMPLOYED'] < 0).astype(int)
            
            # Stability Score: Ratio of life spent employed
            if 'AGE_YEARS' in df.columns:
                df['STABILITY_INDEX'] = df['EMPLOYED_YEARS'] / (df['AGE_YEARS'] - 18).clip(lower=1)
        
        # 3. Financial & Income Features
        if 'AMT_INCOME_TOTAL' in df.columns:
            df['INCOME_LOG'] = np.log1p(df['AMT_INCOME_TOTAL'])
            
            # Income Categories-like branding
            df['INCOME_BRACKET'] = pd.cut(
                df['AMT_INCOME_TOTAL'],
                bins=[0, 50000, 100000, 200000, 500000, float('inf')],
                labels=['Bronze', 'Silver', 'Gold', 'Platinum', 'Elite']
            ).astype(str)
            
            # Income-to-Employment Ratio
            if 'EMPLOYED_YEARS' in df.columns:
                # clip to 1 to avoid infinity
                df['INCOME_PER_YEAR_EMP'] = df['AMT_INCOME_TOTAL'] / df['EMPLOYED_YEARS'].clip(lower=1)
        
        # 4. Family & Demographic Features
        if 'CNT_FAM_MEMBERS' in df.columns:
            df['FAMILY_SIZE_LOG'] = np.log1p(df['CNT_FAM_MEMBERS'].fillna(1))
            
            if 'AMT_INCOME_TOTAL' in df.columns:
                df['INCOME_PER_MEMBER'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'].fillna(1).clip(lower=1)
        
        # 5. Composite Luxury/Risk Scores
        cols_needed = ['AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
        if all(c in df.columns for c in cols_needed):
            high_income = (df['AMT_INCOME_TOTAL'] > 200000).astype(int)
            own_car = (df['FLAG_OWN_CAR'].isin(['Y', 1, '1'])).astype(int)
            own_realty = (df['FLAG_OWN_REALTY'].isin(['Y', 1, '1'])).astype(int)
            small_family = (df['CNT_FAM_MEMBERS'] <= 2).astype(int)
            df['LUXURY_INDEX'] = high_income + own_car + own_realty + small_family
            
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of final feature names."""
        return self.final_features
    
    def get_feature_stats(self) -> Dict[str, Any]:
        """Get feature statistics."""
        return self.feature_stats
