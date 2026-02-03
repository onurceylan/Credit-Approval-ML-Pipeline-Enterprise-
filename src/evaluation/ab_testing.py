"""
Advanced Offline A/B Simulator
==============================

This module implements offline A/B testing simulation to compare Champion (baseline) 
and Challenger (new) models using bootstrap resampling and statistical validation.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
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


class ABTestSimulator:
    """
    Simulates A/B testing between Champion and Challenger models using Bootstrap.
    """
    
    def __init__(
        self, 
        champion_model: Any,
        challenger_model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_iterations: int = 1000,
        confidence_level: float = 0.95,
        random_state: int = 42,
        config: Optional[PipelineConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.champion = champion_model
        self.challenger = challenger_model
        self.X_test = X_test
        self.y_test = y_test
        self.n_iterations = n_iterations
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.config = config # Optional config for financial variables
        self.logger = logger or logging.getLogger(__name__)
        
        np.random.seed(self.random_state)
        
    def run_simulation(
        self, 
        traffic_split: float = 0.5,
        verbose: bool = True
    ) -> ABTestResult:
        """
        Run A/B test simulation with bootstrap resampling.
        """
        if verbose:
            self.logger.info(f"🧪 Running Advanced A/B Simulation ({self.n_iterations} iterations)...")
            self.logger.info(f"   Traffic Split: {traffic_split*100:.0f}% Champ / {(1-traffic_split)*100:.0f}% Chall")
        
        # Storage for results
        champion_scores = {m: [] for m in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'roi']}
        challenger_scores = {m: [] for m in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'roi']}
        
        # Determine sample sizes based on traffic split
        total_size = len(self.X_test)
        champ_sample_size = int(total_size * traffic_split)
        chall_sample_size = total_size - champ_sample_size
        
        for i in range(self.n_iterations):
            # Create bootstrap samples for each group
            X_champ, y_champ = self._bootstrap_sample(champ_sample_size)
            X_chall, y_chall = self._bootstrap_sample(chall_sample_size)
            
            # Evaluate both models
            champ_res = self._evaluate_model(self.champion, X_champ, y_champ)
            chall_res = self._evaluate_model(self.challenger, X_chall, y_chall)
            
            for m in champion_scores.keys():
                champion_scores[m].append(champ_res[m])
                challenger_scores[m].append(chall_res[m])
                
            if verbose and (i + 1) % 250 == 0:
                self.logger.info(f"   Processed {i+1}/{self.n_iterations} iterations...")
        
        # Statistical analysis
        statistical_tests = self._perform_statistical_tests(champion_scores, challenger_scores)
        winner = self._determine_winner(statistical_tests)
        effect_size = self._calculate_effect_size(champion_scores['f1'], challenger_scores['f1'])
        business_impact = self._calculate_business_impact(champion_scores['roi'], challenger_scores['roi'])
        
        return ABTestResult(
            champion_metrics=champion_scores,
            challenger_metrics=challenger_scores,
            statistical_tests=statistical_tests,
            winner=winner,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            business_impact=business_impact,
            traffic_split=traffic_split
        )

    def _bootstrap_sample(self, size: int) -> Tuple[pd.DataFrame, pd.Series]:
        indices = np.random.choice(len(self.X_test), size=size, replace=True)
        return self.X_test.iloc[indices], self.y_test.iloc[indices]

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
        # Use config if available, else default enterprise values
        rev = self.config.revenue_per_approval if self.config else 500
        cost_fp = self.config.cost_false_positive if self.config else 2000
        
        # Labels mapping: Good=0, Bad=1 in our dataset
        # confusion_matrix(y_true, y_pred, labels=[1, 0]) -> tn, fp, fn, tp (where 0 is positive/good)
        # But wait, TN/FP/FN/TP depends on what you define as positive.
        # In credit: Good=0, Bad=1. We want to approve Good.
        # So Approved Good (tp if labels=[1,0]?)
        # Let's iterate:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[1, 0]).ravel()
        # tp: Correct Good Approved (+rev)
        # fp: Bad Approved (-cost_fp)
        # fn: Good Rejected (opportunity cost, ignored for simple ROI per app)
        # tn: Bad Rejected (saved cost, ignored for simple ROI per app)
        net_profit = (tp * rev) - (fp * cost_fp)
        return net_profit / len(y_true)

    def _perform_statistical_tests(self, champ_scores, chall_scores) -> Dict[str, Dict[str, Any]]:
        results = {}
        for m in champ_scores.keys():
            champ_vals = champ_scores[m]
            chall_vals = chall_scores[m]
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
        annual_impact = diff * 10000 * 12 # 10k monthly apps
        return {
            'champion_roi_per_app': float(c_mean),
            'challenger_roi_per_app': float(l_mean),
            'roi_improvement_per_app': float(diff),
            'roi_improvement_pct': float(diff_pct),
            'annual_financial_impact': float(annual_impact)
        }

    def generate_report(self, results: ABTestResult) -> str:
        """Generate comprehensive text report matching user's requested style."""
        report = []
        report.append("=" * 80)
        report.append("A/B TEST SIMULATION REPORT")
        report.append("=" * 80)
        report.append(f"Iterations: {self.n_iterations}")
        report.append(f"Confidence Level: {self.confidence_level * 100:.0f}%")
        report.append(f"\n{'WINNER:':<20} {results.winner}")
        report.append(f"{'Effect Size (Cohen\'s d):':<20} {results.effect_size:.4f}")
        
        effect_interp = "Small" if abs(results.effect_size) < 0.2 else "Medium" if abs(results.effect_size) < 0.5 else "Large"
        report.append(f"{'Effect Interpretation:':<20} {effect_interp}\n")
        
        report.append("=" * 80)
        report.append("STATISTICAL TEST RESULTS")
        report.append("=" * 80)
        
        for metric, stats_data in results.statistical_tests.items():
            report.append(f"\n{metric.upper()}:")
            report.append(f"  Champion Mean:     {stats_data['champion_mean']:.4f}")
            report.append(f"  Challenger Mean:   {stats_data['challenger_mean']:.4f}")
            report.append(f"  Improvement:       {stats_data['relative_improvement']:+.2f}%")
            report.append(f"  P-value:           {stats_data['p_value']:.6f}")
            report.append(f"  Significant:       {'✅ YES' if stats_data['is_significant'] else '❌ NO'}")
            report.append(f"  Champion 95% CI:   [{stats_data['champion_ci'][0]:.4f}, {stats_data['champion_ci'][1]:.4f}]")
            report.append(f"  Challenger 95% CI: [{stats_data['challenger_ci'][0]:.4f}, {stats_data['challenger_ci'][1]:.4f}]")
        
        report.append("\n" + "=" * 80)
        report.append("BUSINESS IMPACT ANALYSIS")
        report.append("=" * 80)
        
        bi = results.business_impact
        report.append(f"Champion ROI per Application:    ${bi['champion_roi_per_app']:,.2f}")
        report.append(f"Challenger ROI per Application:  ${bi['challenger_roi_per_app']:,.2f}")
        report.append(f"Improvement per Application:     ${bi['roi_improvement_per_app']:+,.2f} ({bi['roi_improvement_pct']:+.2f}%)")
        report.append(f"Annual Financial Impact:         ${bi['annual_financial_impact']:+,.0f}")
        
        report.append("\n" + "=" * 80)
        report.append("RECOMMENDATION")
        report.append("=" * 80)
        
        if results.winner == 'Challenger':
            report.append("✅ DEPLOY CHALLENGER MODEL - Significant improvements detected.")
        elif results.winner == 'Champion':
            report.append("⚠️  KEEP CHAMPION MODEL - Current model is equal or better.")
        else:
            report.append("⚖️  TIE - FURTHER INVESTIGATION NEEDED.")
        
        return "\n".join(report)

    def plot_results(self, results: ABTestResult, save_path: Optional[str] = None):
        """Advanced A/B visualization dashboard."""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle("Advanced Offline A/B Simulation: Bootstrap Analysis", fontsize=20, fontweight='bold', y=0.98)
        
        champ_color, chall_color = '#95a5a6', '#3498db'
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'roi']
        
        for idx, m in enumerate(metrics[:3]):
            ax = fig.add_subplot(gs[0, idx])
            ax.hist(results.champion_metrics[m], bins=30, alpha=0.6, label='Champion', color=champ_color)
            ax.hist(results.challenger_metrics[m], bins=30, alpha=0.6, label='Challenger', color=chall_color)
            ax.axvline(np.mean(results.champion_metrics[m]), color=champ_color, linestyle='--')
            ax.axvline(np.mean(results.challenger_metrics[m]), color=chall_color, linestyle='--')
            ax.set_title(f'{m.upper()} Distribution')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        for idx, m in enumerate(metrics[3:6]):
            ax = fig.add_subplot(gs[1, idx])
            ax.boxplot([results.champion_metrics[m], results.challenger_metrics[m]], labels=['Champ', 'Chall'], patch_artist=True)
            ax.set_title(f'{m.upper()} Comparison')
            ax.grid(True, alpha=0.3, axis='y')

        ax_p = fig.add_subplot(gs[2,0])
        names = list(results.statistical_tests.keys())
        p_vals = [results.statistical_tests[n]['p_value'] for n in names]
        ax_p.barh(names, p_vals, color=[chall_color if p < 0.05 else champ_color for p in p_vals])
        ax_p.axvline(0.05, color='red', linestyle='--')
        ax_p.set_title('Significance (p-values)')

        ax_i = fig.add_subplot(gs[2,1])
        imps = [results.statistical_tests[n]['relative_improvement'] for n in names]
        ax_i.barh(names, imps, color=[chall_color if i > 0 else champ_color for i in imps])
        ax_i.axvline(0, color='black')
        ax_i.set_title('Relative Improvement (%)')

        ax_s = fig.add_subplot(gs[2,2]); ax_s.axis('off')
        bi = results.business_impact
        summary = f"SUMMARY\n=======\nWinner: {results.winner}\nROI Lift: {bi['roi_improvement_pct']:+.2f}%\nAnnual: ${bi['annual_financial_impact']:+,.0f}"
        ax_s.text(0, 0.9, summary, transform=ax_s.transAxes, fontsize=12, family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
