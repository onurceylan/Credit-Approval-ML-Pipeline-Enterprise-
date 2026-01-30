"""
Model Factory
=============

Creates ML models with GPU/CPU auto-detection.
"""

import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from ..core.config import PipelineConfig


class BaseModelWrapper(ABC):
    """Base class for model wrappers."""
    
    @abstractmethod
    def get_model(self, params: Dict[str, Any]) -> Any:
        pass
    
    @abstractmethod
    def get_param_space(self) -> Dict[str, Any]:
        pass


class ModelFactory:
    """
    Factory for creating ML models.
    
    Supports:
    - XGBoost (GPU/CPU)
    - LightGBM (GPU/CPU)
    - CatBoost (GPU/CPU)
    - RandomForest
    - GradientBoosting
    - LogisticRegression
    """
    
    SUPPORTED_MODELS = [
        'xgboost', 'lightgbm', 'catboost',
        'random_forest', 'gradient_boosting', 'logistic_regression'
    ]
    
    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._check_available_packages()
    
    def _check_available_packages(self):
        """Check which packages are available."""
        self.available = {}
        
        try:
            import xgboost
            self.available['xgboost'] = True
        except ImportError:
            self.available['xgboost'] = False
        
        try:
            import lightgbm
            self.available['lightgbm'] = True
        except ImportError:
            self.available['lightgbm'] = False
        
        try:
            import catboost
            self.available['catboost'] = True
        except ImportError:
            self.available['catboost'] = False
        
        self.logger.info(f"📦 Available packages: {[k for k, v in self.available.items() if v]}")
    
    def create_model(
        self,
        model_name: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create a model instance.
        
        Args:
            model_name: Name of the model
            params: Optional hyperparameters
        """
        params = params or {}
        
        if model_name == 'xgboost':
            return self._create_xgboost(params)
        elif model_name == 'lightgbm':
            return self._create_lightgbm(params)
        elif model_name == 'catboost':
            return self._create_catboost(params)
        elif model_name == 'random_forest':
            return self._create_random_forest(params)
        elif model_name == 'gradient_boosting':
            return self._create_gradient_boosting(params)
        elif model_name == 'logistic_regression':
            return self._create_logistic_regression(params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _create_xgboost(self, params: Dict) -> Any:
        if not self.available.get('xgboost'):
            raise ImportError("XGBoost not installed")
        
        import xgboost as xgb
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': self.config.random_state,
            'n_jobs': self.config.n_jobs,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        
        if self.config.use_gpu:
            default_params['tree_method'] = 'gpu_hist'
            default_params['device'] = f'cuda:{self.config.gpu_device_id}'
        
        default_params.update(params)
        return xgb.XGBClassifier(**default_params)
    
    def _create_lightgbm(self, params: Dict) -> Any:
        if not self.available.get('lightgbm'):
            raise ImportError("LightGBM not installed")
        
        import lightgbm as lgb
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': self.config.random_state,
            'n_jobs': self.config.n_jobs,
            'verbose': -1
        }
        
        if self.config.use_gpu:
            default_params['device'] = 'gpu'
        
        default_params.update(params)
        return lgb.LGBMClassifier(**default_params)
    
    def _create_catboost(self, params: Dict) -> Any:
        if not self.available.get('catboost'):
            raise ImportError("CatBoost not installed")
        
        from catboost import CatBoostClassifier
        
        default_params = {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'random_seed': self.config.random_state,
            'verbose': False
        }
        
        if self.config.use_gpu:
            default_params['task_type'] = 'GPU'
            default_params['devices'] = str(self.config.gpu_device_id)
        
        default_params.update(params)
        return CatBoostClassifier(**default_params)
    
    def _create_random_forest(self, params: Dict) -> RandomForestClassifier:
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': self.config.random_state,
            'n_jobs': self.config.n_jobs
        }
        default_params.update(params)
        return RandomForestClassifier(**default_params)
    
    def _create_gradient_boosting(self, params: Dict) -> GradientBoostingClassifier:
        default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': self.config.random_state
        }
        default_params.update(params)
        return GradientBoostingClassifier(**default_params)
    
    def _create_logistic_regression(self, params: Dict) -> LogisticRegression:
        default_params = {
            'max_iter': 1000,
            'random_state': self.config.random_state,
            'n_jobs': self.config.n_jobs
        }
        default_params.update(params)
        return LogisticRegression(**default_params)
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        available = ['random_forest', 'gradient_boosting', 'logistic_regression']
        
        if self.available.get('xgboost'):
            available.append('xgboost')
        if self.available.get('lightgbm'):
            available.append('lightgbm')
        if self.available.get('catboost'):
            available.append('catboost')
        
        return available
    
    def get_param_space(self, model_name: str) -> Dict[str, Any]:
        """Get hyperparameter search space for a model."""
        spaces = {
            'xgboost': {
                'n_estimators': (50, 200),
                'max_depth': (3, 8),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.8, 1.0),
                'colsample_bytree': (0.8, 1.0)
            },
            'lightgbm': {
                'n_estimators': (50, 200),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'num_leaves': (20, 100)
            },
            'catboost': {
                'iterations': (100, 300),
                'depth': (4, 8),
                'learning_rate': (0.01, 0.3)
            },
            'random_forest': {
                'n_estimators': (50, 200),
                'max_depth': (5, 25),
                'min_samples_split': (2, 20)
            },
            'gradient_boosting': {
                'n_estimators': (50, 200),
                'max_depth': (3, 10),
                'learning_rate': (0.05, 0.3)
            },
            'logistic_regression': {
                'C': (0.1, 10.0)
            }
        }
        return spaces.get(model_name, {})
