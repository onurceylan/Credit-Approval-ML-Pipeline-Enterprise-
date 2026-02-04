"""
Model Trainer
=============

Comprehensive model training with validation.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score
)
from sklearn.model_selection import cross_val_score

from ..core.config import PipelineConfig
from ..core.exceptions import ModelTrainingError
from ..models.factory import ModelFactory
from ..models.registry import ModelRegistry


class ModelTrainer:
    """
    Comprehensive model training.
    
    Features:
    - Training with validation
    - Cross-validation
    - Metric tracking
    - Model registration
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        model_factory: ModelFactory,
        model_registry: ModelRegistry,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.model_factory = model_factory
        self.model_registry = model_registry
        self.logger = logger or logging.getLogger(__name__)
    
    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train a single model.
        
        Returns:
            Training results dictionary
        """
        self.logger.info(f"🏋️ Training {model_name}...")
        start_time = datetime.now()
        
        try:
            # Create model
            model = self.model_factory.create_model(model_name, params)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = model.predict(X_val)
            
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_val)
                val_roc_auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
            else:
                val_roc_auc = 0.0
            
            val_accuracy = accuracy_score(y_val, y_pred)
            val_f1 = f1_score(y_val, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.config.cv_folds,
                scoring='roc_auc_ovr',
                n_jobs=self.config.n_jobs
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            metrics = {
                'val_accuracy': float(val_accuracy),
                'val_roc_auc': float(val_roc_auc),
                'val_f1': float(val_f1),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'training_time': training_time
            }
            
            # Register model
            model_id = self.model_registry.register_model(
                model, model_name, metrics,
                metadata={'params': params, 'timestamp': datetime.now().isoformat()}
            )
            
            self.logger.info(
                f"   ✅ {model_name}: Accuracy={val_accuracy:.4f}, "
                f"AUC={val_roc_auc:.4f}, CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}"
            )
            
            return {
                'success': True,
                'model': model,
                'model_id': model_id,
                'model_name': model_name,
                'metrics': metrics,
                'params': params
            }
            
        except Exception as e:
            self.logger.error(f"   ❌ {model_name} failed: {str(e)}")
            return {
                'success': False,
                'model_name': model_name,
                'error': str(e)
            }
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_params: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all available models.
        
        Returns:
            Dictionary of training results
        """
        self.logger.info("\n🚀 Training all models...")
        
        model_params = model_params or {}
        results = {}
        
        available_models = self.model_factory.get_available_models()
        
        for model_name in available_models:
            params = model_params.get(model_name, {})
            result = self.train_model(
                model_name, X_train, y_train, X_val, y_val, params
            )
            results[model_name] = result
        
        # Summary
        self._generate_summary_table(results)
        return results

    def _generate_summary_table(self, results: Dict[str, Dict]):
        """Generate a high-impact performance ranking table."""
        successful = [r for r in results.values() if r.get('success')]
        if not successful:
            self.logger.warning("\n⚠️ No models trained successfully to summarize.")
            return

        # Sort by Validation ROC-AUC
        ranked = sorted(successful, key=lambda x: x['metrics'].get('val_roc_auc', 0), reverse=True)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 TRAINING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info("\n🏆 Model Performance Ranking (by Validation ROC-AUC):")
        self.logger.info("-" * 80)

        for i, res in enumerate(ranked):
            m = res['metrics']
            name = res['model_name']
            self.logger.info(
                f" {i+1}. {name:<18} | Val AUC: {m.get('val_roc_auc', 0):.4f} | "
                f"Val Acc: {m.get('val_accuracy', 0):.4f} | CV: {m.get('cv_mean', 0):.4f} | "
                f"Time: {m.get('training_time',0):.1f}s"
            )
        self.logger.info("-" * 80 + "\n")
        self.logger.info(f"📊 Total Successful: {len(successful)}/{len(results)}")
        self.logger.info("=" * 80 + "\n")
