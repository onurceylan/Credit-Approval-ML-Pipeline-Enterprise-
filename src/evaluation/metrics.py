"""
Metrics and Business Analysis
=============================
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from ..core.config import PipelineConfig


class MetricsCalculator:
    """Calculates and tracks ML metrics."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.metrics_history = []
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate standard classification metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        
        return metrics
    
    def log_metrics(self, model_name: str, metrics: Dict[str, float]):
        """Log metrics for tracking."""
        self.metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'model': model_name,
            **metrics
        })


class BusinessAnalyzer:
    """
    Business impact analysis.
    
    Calculates cost-benefit, ROI, and business metrics.
    """
    
    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_impact(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ) -> Dict[str, Any]:
        """Calculate business impact metrics."""
        self.logger.info(f"💰 Analyzing business impact for {model_name}...")
        
        # Calculate confusion matrix elements
        is_predicted_bad = y_pred == 1
        is_actual_bad = y_true == 1
        
        tp = np.sum(is_predicted_bad & is_actual_bad)
        fp = np.sum(is_predicted_bad & ~is_actual_bad)
        tn = np.sum(~is_predicted_bad & ~is_actual_bad)
        fn = np.sum(~is_predicted_bad & is_actual_bad)
        
        # Cost calculations
        cost_fp = fp * self.config.cost_false_negative  # Rejected good
        cost_fn = fn * self.config.cost_false_positive  # Approved bad
        revenue_tn = tn * self.config.revenue_per_approval
        
        total_cost = cost_fp + cost_fn
        total_revenue = revenue_tn
        net_profit = total_revenue - total_cost
        roi = (net_profit / total_cost * 100) if total_cost > 0 else 0
        
        impact = {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'total_cost': float(total_cost),
            'total_revenue': float(total_revenue),
            'net_profit': float(net_profit),
            'roi_percent': float(roi)
        }
        
        self.logger.info(f"   💵 Net Profit: ${net_profit:,.0f}, ROI: {roi:.1f}%")
        
        return impact
    
    def generate_business_case(
        self,
        analysis_results: Dict[str, Dict],
        best_model: str
    ) -> str:
        """Generate business case document."""
        lines = [
            "=" * 60,
            "BUSINESS CASE DOCUMENT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            f"RECOMMENDED MODEL: {best_model}",
            f"Expected Net Profit: ${analysis_results[best_model]['net_profit']:,.0f}",
            f"ROI: {analysis_results[best_model]['roi_percent']:.1f}%",
            "",
            "MODEL COMPARISON",
            "-" * 40
        ]
        
        for model, impact in sorted(
            analysis_results.items(),
            key=lambda x: x[1]['net_profit'],
            reverse=True
        ):
            lines.append(f"{model}: ${impact['net_profit']:,.0f} profit, {impact['roi_percent']:.1f}% ROI")
        
        # Save document
        doc_path = Path(self.config.output_dir) / self.config.results_dir / "business_case.txt"
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = '\n'.join(lines)
        with open(doc_path, 'w') as f:
            f.write(content)
        
        return content
