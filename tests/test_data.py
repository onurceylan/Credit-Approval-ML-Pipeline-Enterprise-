"""
Data Module Tests
=================
"""

import pytest
import pandas as pd
import numpy as np


class TestDataLoader:
    """Tests for DataLoader."""
    
    def test_environment_detection(self):
        """Test environment detection."""
        from src.core.config import get_config
        config = get_config()
        assert config.raw_data_dir is not None
    
    def test_data_paths_exist(self):
        """Test that config has data paths."""
        from src.core.config import get_config
        config = get_config()
        assert hasattr(config, 'raw_data_dir')
        assert hasattr(config, 'application_file')


class TestDataValidator:
    """Tests for DataValidator."""
    
    def test_validate_required_columns(self):
        """Test required columns validation."""
        from src.data.validator import DataValidator
        from src.core.config import get_config
        
        config = get_config()
        validator = DataValidator(config)
        
        # Valid data
        app_df = pd.DataFrame({'ID': [1, 2, 3]})
        credit_df = pd.DataFrame({
            'ID': [1, 2, 3],
            'MONTHS_BALANCE': [-1, -2, -3],
            'STATUS': ['0', '1', '2']
        })
        
        # Should not raise
        validator._check_required_columns(app_df, credit_df)
    
    def test_validate_empty_data_raises(self):
        """Test empty data raises error."""
        from src.data.validator import DataValidator
        from src.core.config import get_config
        from src.core.exceptions import DataValidationError
        
        config = get_config()
        validator = DataValidator(config)
        
        empty_df = pd.DataFrame()
        
        with pytest.raises(DataValidationError):
            validator._check_empty_data(empty_df, pd.DataFrame({'ID': [1]}))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
