"""
Advanced Offline A/B Simulator
==============================

This module implements offline A/B testing simulation to compare Champion (baseline) 
and Challenger (new) models using bootstrap resampling and statistical validation.

Based on Enterprise ML Standards.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from dataclasses import dataclass
from ..core.config import PipelineConfig


@dataclass
class ABTestResult:
    """Data class to store comprehensive A/B test results"""
    champion_metrics: Dict[str, List[float]]
    challenger_metrics: Dict[str, List[float]]
    statistical_tests: Dict[str, Dict[str, Any]]
    winner: str
    confidence_level: float
    effect_size: float
    business_impact: Dict[str, float]
    traffic_split: float


class OfflineABSimulator:
    """
    Simulates A/B testing between Champion and Challenger models using Bootstrap.
    
    Attributes:
        config: Pipeline configuration (for random state and financial logic)
        n_iterations: Number of bootstrap iterations (default: 1000)
    """
    
    def __init__(
        self, 
        config: PipelineConfig, 
        n_iterations: int = 1000,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.n_iterations = n_iterations
        self.logger = logger or logging.getLogger(__name__)
        self.confidence_level = 0.95
        np.random.seed(self.config.random_state)
        
    def run_simulation(
        self, 
        champion_model: Any, 
        challenger_model: Any, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        traffic_split: float = 0.5
    ) -> ABTestResult:
        """
        Run A/B test simulation with bootstrap resampling.
        """
        self.logger.info(f"🧪 Running Advanced A/B Simulation ({self.n_iterations} iterations)...")
        self.logger.info(f"   Traffic Split: {traffic_split*100:.0f}% Champion / {(1-traffic_split)*100:.0f}% Challenger")
        
        # Storage for results
        champion_scores = {m: [] for m in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'roi']}
        challenger_scores = {m: [] for m in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'roi']}
        
        # Determine sample sizes based on traffic split
        total_size = len(X_test)
        champ_sample_size = int(total_size * traffic_split)
        chall_sample_size = total_size - champ_sample_size
        
        for i in range(self.n_iterations):
            # Create bootstrap samples for each group
            X_champ, y_champ = self._bootstrap_sample(X_test, y_test, champ_sample_size)
            X_chall, y_chall = self._bootstrap_sample(X_test, y_test, chall_sample_size)
            
            # Evaluate Champion
            champ_res = self._evaluate_model(champion_model, X_champ, y_champ)
            # Evaluate Challenger
            chall_res = self._evaluate_model(challenger_model, X_chall, y_chall)
            
            for m in champion_scores.keys():
                champion_scores[m].append(champ_res[m])
                challenger_scores[m].append(chall_res[m])
                
            if (i + 1) % 250 == 0:
                self.logger.info(f"   Processed {i+1} iterations...")
        
        # Statistical analysis
        statistical_tests = self._perform_statistical_tests(champion_scores, challenger_scores)
        winner = self._determine_winner(statistical_tests)
        effect_size = self._calculate_effect_size(champion_scores['f1'], challenger_scores['f1'])
        business_impact = self._calculate_business_impact(champion_scores['roi'], challenger_scores['roi'])
        
        result = ABTestResult(
            champion_metrics=champion_scores,
            challenger_metrics=challenger_scores,
            statistical_tests=statistical_tests,
            winner=winner,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            business_impact=business_impact,
            traffic_split=traffic_split
        )
        
        self.logger.info(f"✅ Simulation complete! Winner: {winner} (Profit Lift: {business_impact['roi_improvement_pct']:.2f}%)")
        return result

    def _bootstrap_sample(self, X: pd.DataFrame, y: pd.Series, size: int) -> Tuple[pd.DataFrame, pd.Series]:
        indices = np.random.choice(len(X), size=size, replace=True)
        return X.iloc[indices], y.iloc[indices]

    def _evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        y_pred = model.predict(X)
        
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X)[:, 1]
                auc = roc_auc_score(y, y_proba)
            else:
                auc = 0.0
        except:
            auc = 0.0
            
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'auc': auc,
            'roi': self._calculate_roi(y, y_pred)
        }

    def _calculate_roi(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates ROI based on project-specific financial variables."""
        rev = self.config.revenue_per_approval
        cost_fp = self.config.cost_false_positive # Approved but default
        # Simple ROI per app logic
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[1, 0]).ravel() # Labels [Bad, Good]
        # tp: Good approved (+rev)
        # fp: Bad approved (-cost_fp)
        # fn: Good rejected (-rev cost? no, handled in business analyst as opportunity cost, here we stick to net profit)
        net_profit = (tp * rev) - (fp * cost_fp)
        return net_profit / len(y_true)

    def _perform_statistical_tests(self, champ_scores, chall_scores) -> Dict[str, Dict[str, Any]]:
        results = {}
        for m in champ_scores.keys():
            champ_vals = champ_scores[m]
            chall_vals = chall_scores[m]
            
            # Mann-Whitney U test (non-parametric)
            _, p_value = stats.mannwhitneyu(chall_vals, champ_vals, alternative='two-sided')
            
            results[m] = {
                'p_value': float(p_value),
                'is_significant': p_value < (1 - self.confidence_level),
                'champion_mean': float(np.mean(champ_vals)),
                'challenger_mean': float(np.mean(chall_vals)),
                'champion_ci': np.percentile(champ_vals, [2.5, 97.5]).tolist(),
                'challenger_ci': np.percentile(chall_vals, [2.5, 97.5]).tolist(),
                'relative_improvement': float(((np.mean(chall_vals) - np.mean(champ_vals)) / np.mean(champ_vals) * 100) if np.mean(champ_vals) != 0 else 0)
            }
        return results

    def _calculate_effect_size(self, champ_vals, chall_vals) -> float:
        pooled_std = np.sqrt((np.var(champ_vals) + np.var(chall_vals)) / 2)
        if pooled_std == 0: return 0.0
        return float((np.mean(chall_vals) - np.mean(champ_vals)) / pooled_std)

    def _determine_winner(self, stats_res: Dict) -> str:
        chall_wins = 0
        champ_wins = 0
        for m in ['accuracy', 'f1', 'auc', 'roi']:
            if stats_res[m]['is_significant']:
                if stats_res[m]['challenger_mean'] > stats_res[m]['champion_mean']:
                    chall_wins += 1
                else:
                    champ_wins += 1
        if chall_wins > champ_wins: return 'Challenger'
        if champ_wins > chall_wins: return 'Champion'
        return 'Tie'

    def _calculate_business_impact(self, champ_roi, chall_roi) -> Dict[str, float]:
        c_mean = np.mean(champ_roi)
        l_mean = np.mean(chall_roi)
        diff = l_mean - c_mean
        diff_pct = (diff / abs(c_mean) * 100) if c_mean != 0 else 0
        
        # Scale to 10k monthly apps
        annual_impact = diff * 10000 * 12
        return {
            'champion_roi_per_app': float(c_mean),
            'challenger_roi_per_app': float(l_mean),
            'roi_improvement_per_app': float(diff),
            'roi_improvement_pct': float(diff_pct),
            'annual_financial_impact': float(annual_impact)
        }
