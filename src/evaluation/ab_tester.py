"""
Offline A/B Simulator
=====================

Simulates A/B testing between a champion (baseline) and a challenger model.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy import stats

from ..core.config import PipelineConfig


class OfflineABSimulator:
    """
    Simulates A/B testing using historical data.
    
    Compares a 'Champion' (usually a baseline like Logistic Regression) 
    against a 'Challenger' (the best model from the pipeline).
    """
    
    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
    def run_simulation(
        self, 
        champion_model: Any, 
        challenger_model: Any, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Run randomization and performance comparison.
        """
        self.logger.info("🧪 Running Offline A/B Simulation...")
        
        # 1. Randomly split test set into Group A (Champion) and Group B (Challenger)
        np.random.seed(self.config.random_state)
        indices = np.random.permutation(len(X_test))
        split_idx = len(X_test) // 2
        
        indices_a = indices[:split_idx]
        indices_b = indices[split_idx:]
        
        X_a, y_a = X_test.iloc[indices_a], y_test.iloc[indices_a]
        X_b, y_b = X_test.iloc[indices_b], y_test.iloc[indices_b]
        
        # 2. Predictions
        y_pred_a = champion_model.predict(X_a)
        y_pred_b = challenger_model.predict(X_b)
        
        # 3. Calculate Performance Metrics
        metrics_a = self._calculate_basic_metrics(y_a, y_pred_a)
        metrics_b = self._calculate_basic_metrics(y_b, y_pred_b)
        
        # 4. Financial Simulation
        profit_a = self._simulate_profit(y_a, y_pred_a)
        profit_b = self._simulate_profit(y_b, y_pred_b)
        
        # 5. Statistical Significance (T-test on profit samples)
        t_stat, p_value = stats.ttest_ind(profit_a['samples'], profit_b['samples'])
        
        # 6. Summary
        lift = ((metrics_b['accuracy'] - metrics_a['accuracy']) / metrics_a['accuracy']) * 100
        profit_lift = ((profit_b['total'] - profit_a['total']) / profit_a['total']) * 100
        
        results = {
            'group_a': {
                'size': len(X_a),
                'accuracy': float(metrics_a['accuracy']),
                'precision': float(metrics_a['precision']),
                'total_profit': float(profit_a['total']),
                'approval_rate': float(np.mean(y_pred_a == 0))
            },
            'group_b': {
                'size': len(X_b),
                'accuracy': float(metrics_b['accuracy']),
                'precision': float(metrics_b['precision']),
                'total_profit': float(profit_b['total']),
                'approval_rate': float(np.mean(y_pred_b == 0))
            },
            'comparison': {
                'accuracy_lift_pct': float(lift),
                'profit_lift_pct': float(profit_lift),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            },
            'raw_data': {
                'profit_samples_a': profit_a['samples'].tolist(),
                'profit_samples_b': profit_b['samples'].tolist()
            }
        }
        
        self.logger.info(f"   📊 A/B Result: Profit Lift = {profit_lift:.2f}%, p-value = {p_value:.4f}")
        return results

    def _calculate_basic_metrics(self, y_true, y_pred):
        acc = np.mean(y_true == y_pred)
        # Simplified precision for 0 class (Good)
        tp = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 1) & (y_pred == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        return {'accuracy': acc, 'precision': prec}

    def _simulate_profit(self, y_true, y_pred) -> Dict[str, Any]:
        """Calculates per-decision profit samples for statistical testing."""
        # Financial logic (matches business.py)
        rev = self.config.revenue_per_approval
        cost_fp = self.config.cost_false_positive # Bad approved
        
        samples = []
        for yt, yp in zip(y_true, y_pred):
            if yp == 0: # Approved
                if yt == 0: # Good
                    samples.append(rev)
                else: # Bad (False Positive)
                    samples.append(-cost_fp)
            else: # Rejected
                samples.append(0)
                
        samples = np.array(samples)
        return {
            'total': np.sum(samples),
            'samples': samples
        }
