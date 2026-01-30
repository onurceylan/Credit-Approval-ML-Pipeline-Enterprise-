"""
Model Evaluator
===============

Comprehensive model evaluation and comparison.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from ..core.config import PipelineConfig
from ..core.exceptions import ModelEvaluationError


class ModelEvaluator:
    """
    Comprehensive model evaluation.
    
    Features:
    - Test set evaluation
    - Model comparison
    - Best model selection
    - Report generation
    """
    
    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
    
    def evaluate_model(
        self,
        model: Any,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate a single model on test set."""
        self.logger.info(f"📊 Evaluating {model_name}...")
        
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            test_roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        else:
            y_pred_proba = None
            test_roc_auc = 0.0
        
        metrics = {
            'test_accuracy': float(accuracy_score(y_test, y_pred)),
            'test_precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'test_recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'test_f1': float(f1_score(y_test, y_pred, average='weighted')),
            'test_roc_auc': float(test_roc_auc),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        self.logger.info(f"   ✅ Accuracy={metrics['test_accuracy']:.4f}, AUC={metrics['test_roc_auc']:.4f}")
        
        return metrics
    
    def evaluate_all(
        self,
        training_results: Dict[str, Dict],
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate all trained models."""
        self.logger.info("\n📊 Evaluating all models on test set...")
        
        evaluation_results = {}
        
        for model_name, result in training_results.items():
            if not result.get('success') or 'model' not in result:
                continue
            
            metrics = self.evaluate_model(result['model'], model_name, X_test, y_test)
            evaluation_results[model_name] = metrics
        
        # Save evaluation report
        self._save_report(evaluation_results)
        
        return evaluation_results
    
    def select_best_model(
        self,
        training_results: Dict[str, Dict],
        evaluation_results: Dict[str, Dict],
        primary_metric: str = 'test_roc_auc'
    ) -> Tuple[str, Dict[str, Any]]:
        """Select best model based on metrics."""
        self.logger.info("\n🏆 Selecting best model...")
        
        model_scores = {}
        
        for model_name, eval_result in evaluation_results.items():
            train_result = training_results.get(model_name, {})
            
            # Composite score
            test_auc = eval_result.get('test_roc_auc', 0)
            cv_mean = train_result.get('metrics', {}).get('cv_mean', 0)
            cv_std = train_result.get('metrics', {}).get('cv_std', 1)
            
            stability = 1 / (1 + cv_std)
            composite = 0.5 * test_auc + 0.3 * cv_mean + 0.2 * stability
            
            model_scores[model_name] = {
                'composite_score': composite,
                'test_roc_auc': test_auc,
                'cv_mean': cv_mean,
                'cv_std': cv_std
            }
        
        best_model = max(model_scores.keys(), key=lambda k: model_scores[k]['composite_score'])
        
        self.logger.info(f"   🥇 Best model: {best_model}")
        self.logger.info(f"      Composite: {model_scores[best_model]['composite_score']:.4f}")
        self.logger.info(f"      Test AUC: {model_scores[best_model]['test_roc_auc']:.4f}")
        
        return best_model, model_scores
    
    def _save_report(self, evaluation_results: Dict[str, Dict]):
        """Save evaluation report."""
        try:
            report_path = Path(self.config.output_dir) / self.config.results_dir / "evaluation_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            clean_results = {}
            for name, result in evaluation_results.items():
                clean_results[name] = {
                    k: v for k, v in result.items()
                    if k not in ['predictions', 'probabilities']
                }
            
            with open(report_path, 'w') as f:
                json.dump(clean_results, f, indent=2, default=str)
            
            self.logger.info(f"   💾 Evaluation report saved")
        except Exception as e:
            self.logger.warning(f"Could not save report: {e}")
