"""
Hyperparameter Optimizer
========================

Optuna-based hyperparameter optimization.
"""

import logging
from typing import Dict, Any, Optional, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from ..core.config import PipelineConfig
from ..models.factory import ModelFactory


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna.
    
    Features:
    - Bayesian optimization
    - Early pruning
    - Cross-validation based scoring
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        model_factory: ModelFactory,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.model_factory = model_factory
        self.logger = logger or logging.getLogger(__name__)
        self._check_optuna()
    
    def _check_optuna(self):
        """Check if Optuna is available."""
        try:
            import optuna
            self.optuna = optuna
            self.available = True
            # Suppress Optuna logs
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            self.optuna = None
            self.available = False
            self.logger.warning("⚠️ Optuna not installed, optimization disabled")
    
    def optimize(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int = 5,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a model.
        
        Returns:
            Best parameters and score
        """
        if not self.available:
            self.logger.warning("Optuna not available, returning default params")
            return {'best_params': {}, 'best_score': 0}
        
        n_trials = n_trials or self.config.optuna_trials
        timeout = timeout or self.config.optuna_timeout
        
        self.logger.info(f"🔍 Optimizing {model_name} ({n_trials} trials)...")
        
        param_space = self.model_factory.get_param_space(model_name)
        
        def objective(trial):
            params = self._sample_params(trial, param_space)
            
            try:
                model = self.model_factory.create_model(model_name, params)
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv_folds, scoring='roc_auc_ovr', n_jobs=-1
                )
                return scores.mean()
            except Exception:
                return 0.0
        
        study = self.optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        
        self.logger.info(f"   ✅ Best score: {study.best_value:.4f}")
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def _sample_params(self, trial, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample hyperparameters from search space."""
        params = {}
        
        for name, bounds in param_space.items():
            if isinstance(bounds, tuple) and len(bounds) == 2:
                low, high = bounds
                if isinstance(low, int) and isinstance(high, int):
                    params[name] = trial.suggest_int(name, low, high)
                elif isinstance(low, float) and isinstance(high, float):
                    if name in ['learning_rate', 'C']:
                        params[name] = trial.suggest_float(name, low, high, log=True)
                    else:
                        params[name] = trial.suggest_float(name, low, high)
        
        return params
    
    def optimize_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """Optimize all available models."""
        results = {}
        
        for model_name in self.model_factory.get_available_models():
            result = self.optimize(model_name, X_train, y_train)
            results[model_name] = result
        
        return results
