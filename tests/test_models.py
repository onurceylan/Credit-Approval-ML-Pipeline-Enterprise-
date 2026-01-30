"""
Models Module Tests
===================
"""

import pytest
import pandas as pd
import numpy as np


class TestModelFactory:
    """Tests for ModelFactory."""
    
    def test_get_available_models(self):
        """Test available models list."""
        from src.models.factory import ModelFactory
        from src.core.config import get_config
        
        config = get_config()
        factory = ModelFactory(config)
        
        models = factory.get_available_models()
        
        # Should always have sklearn models
        assert 'random_forest' in models
        assert 'logistic_regression' in models
    
    def test_create_random_forest(self):
        """Test creating RandomForest model."""
        from src.models.factory import ModelFactory
        from src.core.config import get_config
        from sklearn.ensemble import RandomForestClassifier
        
        config = get_config()
        factory = ModelFactory(config)
        
        model = factory.create_model('random_forest', {'n_estimators': 10})
        
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 10
    
    def test_create_logistic_regression(self):
        """Test creating LogisticRegression model."""
        from src.models.factory import ModelFactory
        from src.core.config import get_config
        from sklearn.linear_model import LogisticRegression
        
        config = get_config()
        factory = ModelFactory(config)
        
        model = factory.create_model('logistic_regression')
        
        assert isinstance(model, LogisticRegression)
    
    def test_get_param_space(self):
        """Test parameter spaces."""
        from src.models.factory import ModelFactory
        from src.core.config import get_config
        
        config = get_config()
        factory = ModelFactory(config)
        
        space = factory.get_param_space('random_forest')
        
        assert 'n_estimators' in space
        assert 'max_depth' in space


class TestModelRegistry:
    """Tests for ModelRegistry."""
    
    def test_register_and_load(self, tmp_path):
        """Test model registration and loading."""
        from src.models.registry import ModelRegistry
        from src.core.config import PipelineConfig
        from sklearn.linear_model import LogisticRegression
        
        config = PipelineConfig(output_dir=str(tmp_path))
        registry = ModelRegistry(config)
        
        model = LogisticRegression()
        model.fit([[1], [2], [3]], [0, 1, 0])
        
        model_id = registry.register_model(
            model, 'test_model',
            {'accuracy': 0.9}
        )
        
        loaded = registry.load_model(model_id)
        
        assert loaded is not None
        assert hasattr(loaded, 'predict')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
