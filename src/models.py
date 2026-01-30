"""
Models Module
=============

Contains model factory and configurations for different ML algorithms
including GPU-accelerated boosting models.
"""

import logging
from typing import Dict, List, Any, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from .utils import DependencyManager


# =============================================================================
# MODEL FACTORY
# =============================================================================

class ModelFactory:
    """
    Factory class for creating different model types.
    
    Supports: XGBoost, LightGBM, CatBoost, RandomForest, 
    GradientBoosting, LogisticRegression with GPU/CPU options.
    """
    
    def __init__(self, dependency_manager: DependencyManager, logger: logging.Logger):
        self.dependency_manager = dependency_manager
        self.logger = logger
        self.available_models = self._get_available_models()
    
    def _get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get available model configurations.
        
        Returns:
            Dictionary of model configurations with params and param spaces
        """
        models = {}
        gpu_available = self.dependency_manager.is_available('gpu')
        
        # XGBoost - GPU/CPU
        if self.dependency_manager.is_available('xgboost'):
            from xgboost import XGBClassifier
            
            if gpu_available:
                models['XGBoost'] = {
                    'class': XGBClassifier,
                    'params': {
                        'tree_method': 'gpu_hist',
                        'device': 'cuda:0',
                        'eval_metric': 'mlogloss',
                        'random_state': 42,
                        'verbosity': 0
                    },
                    'param_space': {
                        'n_estimators': (50, 200),
                        'max_depth': (3, 8),
                        'learning_rate': (0.01, 0.3),
                        'subsample': (0.8, 1.0)
                    },
                    'type': 'xgboost'
                }
            else:
                models['XGBoost'] = {
                    'class': XGBClassifier,
                    'params': {
                        'tree_method': 'hist',
                        'eval_metric': 'mlogloss',
                        'random_state': 42,
                        'verbosity': 0,
                        'n_jobs': -1
                    },
                    'param_space': {
                        'n_estimators': (50, 200),
                        'max_depth': (3, 8),
                        'learning_rate': (0.01, 0.3),
                        'subsample': (0.8, 1.0)
                    },
                    'type': 'xgboost'
                }
        
        # LightGBM - GPU/CPU
        if self.dependency_manager.is_available('lightgbm'):
            from lightgbm import LGBMClassifier
            
            if gpu_available:
                models['LightGBM'] = {
                    'class': LGBMClassifier,
                    'params': {
                        'device': 'gpu',
                        'gpu_platform_id': 0,
                        'gpu_device_id': 0,
                        'objective': 'multiclass',
                        'random_state': 42,
                        'verbose': -1,
                        'force_row_wise': True
                    },
                    'param_space': {
                        'n_estimators': (50, 200),
                        'max_depth': (3, 10),
                        'learning_rate': (0.01, 0.3),
                        'num_leaves': (20, 100)
                    },
                    'type': 'lightgbm'
                }
            else:
                models['LightGBM'] = {
                    'class': LGBMClassifier,
                    'params': {
                        'objective': 'multiclass',
                        'random_state': 42,
                        'verbose': -1,
                        'n_jobs': -1
                    },
                    'param_space': {
                        'n_estimators': (50, 200),
                        'max_depth': (3, 10),
                        'learning_rate': (0.01, 0.3),
                        'num_leaves': (20, 100)
                    },
                    'type': 'lightgbm'
                }
        
        # CatBoost - GPU/CPU
        if self.dependency_manager.is_available('catboost'):
            from catboost import CatBoostClassifier
            
            if gpu_available:
                models['CatBoost'] = {
                    'class': CatBoostClassifier,
                    'params': {
                        'task_type': 'GPU',
                        'devices': '0',
                        'verbose': False,
                        'random_state': 42,
                        'loss_function': 'MultiClass',
                        'eval_metric': 'MultiClass'
                    },
                    'param_space': {
                        'iterations': (100, 300),
                        'depth': (4, 8),
                        'learning_rate': (0.01, 0.3),
                        'l2_leaf_reg': (1, 10)
                    },
                    'type': 'catboost'
                }
            else:
                models['CatBoost'] = {
                    'class': CatBoostClassifier,
                    'params': {
                        'task_type': 'CPU',
                        'verbose': False,
                        'random_state': 42,
                        'loss_function': 'MultiClass',
                        'eval_metric': 'MultiClass',
                        'thread_count': -1
                    },
                    'param_space': {
                        'iterations': (100, 300),
                        'depth': (4, 8),
                        'learning_rate': (0.01, 0.3),
                        'l2_leaf_reg': (1, 10)
                    },
                    'type': 'catboost'
                }
        
        # RandomForest - Always available (sklearn)
        models['RandomForest'] = {
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            },
            'param_space': {
                'n_estimators': (50, 200),
                'max_depth': (5, 25),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': ['sqrt', 'log2', None]
            },
            'type': 'sklearn'
        }
        
        # GradientBoosting - Always available (sklearn)
        models['GradientBoosting'] = {
            'class': GradientBoostingClassifier,
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'random_state': 42
            },
            'param_space': {
                'n_estimators': (50, 200),
                'learning_rate': (0.05, 0.3),
                'max_depth': (3, 10),
                'min_samples_split': (10, 50),
                'min_samples_leaf': (5, 20)
            },
            'type': 'sklearn'
        }
        
        # LogisticRegression - Always available (sklearn)
        models['LogisticRegression'] = {
            'class': LogisticRegression,
            'params': {
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1,
                'multi_class': 'multinomial',
                'solver': 'lbfgs'
            },
            'param_space': {
                'C': (0.1, 10.0),
                'penalty': ['l2'],
                'solver': ['lbfgs']
            },
            'type': 'sklearn'
        }
        
        return models
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Model configuration dictionary
        
        Raises:
            ValueError: If model not available
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available. Available: {list(self.available_models.keys())}")
        return self.available_models[model_name]
    
    def get_available_model_names(self) -> List[str]:
        """
        Get list of available model names.
        
        Returns:
            List of model names
        """
        return list(self.available_models.keys())
    
    def create_model(self, model_name: str, params: Optional[Dict] = None) -> Any:
        """
        Create a model instance with given parameters.
        
        Args:
            model_name: Name of the model to create
            params: Optional parameters to override defaults
        
        Returns:
            Instantiated model
        """
        config = self.get_model_config(model_name)
        final_params = config['params'].copy()
        
        if params:
            final_params.update(params)
        
        return config['class'](**final_params)
    
    def get_model_types(self) -> Dict[str, str]:
        """
        Get mapping of model names to types.
        
        Returns:
            Dictionary mapping model names to their types
        """
        return {name: config['type'] for name, config in self.available_models.items()}
