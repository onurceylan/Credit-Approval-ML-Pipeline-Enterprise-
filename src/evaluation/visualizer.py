"""
Pipeline Visualizer
===================

Handles all plotting and visualization for the pipeline.
Supports saving to file and displaying in notebooks.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix

from ..core.config import PipelineConfig


class PipelineVisualizer:
    """
    Centralized visualization handler.
    
    Generates:
    - Target distribution plots
    - ROC curves
    - Confusion matrices
    - Feature importance plots
    - Model comparison charts
    - Business impact analysis plots
    """
    
    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.output_dir = Path(config.output_dir) / config.plots_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
    
    def plot_target_distribution(self, y: pd.Series, title: str = "Target Distribution"):
        """Plot target class distribution."""
        plt.figure(figsize=(8, 5))
        counts = y.value_counts(normalize=True) * 100
        ax = sns.barplot(x=counts.index, y=counts.values)
        
        plt.title(title)
        plt.xlabel("Class")
        plt.ylabel("Percentage (%)")
        
        # Add labels
        for i, v in enumerate(counts.values):
            ax.text(i, v + 1, f"{v:.1f}%", ha='center')
            
        plt.tight_layout()
        self._save_and_show("target_distribution.png")
    
    def plot_model_comparison(self, evaluation_results: Dict[str, Dict]):
        """Plot model comparison metrics."""
        metrics = ['test_accuracy', 'test_roc_auc', 'test_f1']
        data = []
        
        for model_name, res in evaluation_results.items():
            for metric in metrics:
                data.append({
                    'Model': model_name,
                    'Metric': metric.replace('test_', '').upper(),
                    'Score': res.get(metric, 0)
                })
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Model', y='Score', hue='Metric')
        plt.title("Model Comparison Metrics")
        plt.ylim(0, 1.1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        self._save_and_show("model_comparison.png")

    def plot_roc_curves(self, training_results: Dict[str, Dict], X_test: pd.DataFrame, y_test: pd.Series):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))
        
        for model_name, result in training_results.items():
            if not result.get('success') or 'model' not in result:
                continue
                
            model = result['model']
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_score = result['metrics'].get('roc_auc', 0)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        self._save_and_show("roc_curves.png")

    def plot_confusion_matrices(self, evaluation_results: Dict[str, Dict]):
        """Plot confusion matrix for each model."""
        models = [m for m in evaluation_results.keys()]
        n_models = len(models)
        
        if n_models == 0:
            return

        cols = 2
        rows = (n_models + 1) // 2
        
        plt.figure(figsize=(12, 5 * rows))
        
        for i, model_name in enumerate(models):
            cm = np.array(evaluation_results[model_name]['confusion_matrix'])
            
            plt.subplot(rows, cols, i + 1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f"Confusion Matrix: {model_name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            
        plt.tight_layout()
        self._save_and_show("confusion_matrices.png")

    def plot_feature_importance(self, model: Any, feature_names: List[str], model_name: str, top_n: int = 20):
        """Plot feature importance for a specific model."""
        importance = None
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
            
        if importance is None:
            self.logger.warning(f"Model {model_name} does not support feature importance.")
            return
            
        # Create dataframe
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=df, x='Importance', y='Feature')
        plt.title(f"Top {top_n} Feature Importance ({model_name})")
        plt.xlabel("Importance")
        plt.tight_layout()
        self._save_and_show(f"feature_importance_{model_name}.png")

    def plot_business_impact(self, business_results: Dict[str, Dict]):
        """Plot business metrics (Profit, ROI)."""
        data = []
        for model, res in business_results.items():
            data.append({
                'Model': model,
                'Net Profit': res['net_profit'],
                'ROI (%)': res['roi_percent']
            })
            
        df = pd.DataFrame(data)
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Profit Bar
        sns.barplot(data=df, x='Model', y='Net Profit', ax=ax1, color='lightblue', alpha=0.6)
        ax1.set_ylabel('Net Profit ($)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # ROI Line
        ax2 = ax1.twinx()
        sns.lineplot(data=df, x='Model', y='ROI (%)', ax=ax2, color='red', marker='o', linewidth=3)
        ax2.set_ylabel('ROI (%)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title("Business Impact Analysis: Profit vs ROI")
        plt.tight_layout()
        self._save_and_show("business_impact_analysis.png")

    def _save_and_show(self, filename: str):
        """Save plot to file and show via matplotlib."""
        path = self.output_dir / filename
        plt.savefig(path, dpi=300, bbox_inches='tight')
        self.logger.info(f"   📊 Plot saved: {filename}")
        plt.show()  # This will display inline in notebooks!
        plt.close()
