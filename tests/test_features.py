"""
Features Module Tests
=====================
"""

import pytest
import pandas as pd
import numpy as np


class TestFeatureEngineer:
    """Tests for FeatureEngineer."""
    
    def test_fit_transform(self):
        """Test fit_transform creates features."""
        from src.features.engineer import FeatureEngineer
        from src.core.config import get_config
        
        config = get_config()
        engineer = FeatureEngineer(config)
        
        # Sample data
        df = pd.DataFrame({
            'DAYS_BIRTH': [-10000, -15000, -20000],
            'DAYS_EMPLOYED': [-500, -1000, 365],
            'AMT_INCOME_TOTAL': [50000, 100000, 150000],
            'CNT_CHILDREN': [0, 1, 2],
            'CNT_FAM_MEMBERS': [1, 2, 4]
        })
        
        result = engineer.fit_transform(df)
        
        assert engineer.is_fitted
        assert 'AGE_YEARS' in result.columns
        assert 'EMPLOYED_YEARS' in result.columns
    
    def test_transform_requires_fit(self):
        """Test transform raises if not fitted."""
        from src.features.engineer import FeatureEngineer
        from src.core.config import get_config
        from src.core.exceptions import FeatureEngineeringError
        
        config = get_config()
        engineer = FeatureEngineer(config)
        
        df = pd.DataFrame({'test': [1, 2, 3]})
        
        with pytest.raises(FeatureEngineeringError):
            engineer.transform(df)


class TestTargetCreator:
    """Tests for TargetCreator."""
    
    def test_create_target_binary(self):
        """Test target is binary."""
        from src.features.preprocessor import TargetCreator
        
        creator = TargetCreator()
        
        app_df = pd.DataFrame({'ID': [1, 2, 3, 4, 5]})
        credit_df = pd.DataFrame({
            'ID': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            'MONTHS_BALANCE': [-10, -1, -10, -1, -10, -1, -10, -1, -10, -1],
            'STATUS': ['0', '0', '0', '2', '1', '0', '0', '3', '0', '0']
        })
        
        result = creator.create_target(app_df, credit_df)
        
        assert 'target' in result.columns
        assert set(result['target'].unique()).issubset({0, 1})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
