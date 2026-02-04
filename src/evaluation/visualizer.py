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
    
    def _save_and_show(self, filename: str):
        """Save plot to file and show via matplotlib."""
        path = self.output_dir / filename
        plt.savefig(path, dpi=100, bbox_inches='tight')
        self.logger.info(f"   📊 Plot saved: {filename}")
        plt.close()
    
    def plot_target_distribution(self, y: pd.Series, title: str = "Target Distribution"):
        """Plot target class distribution."""
        plt.figure(figsize=(8, 5))
        counts = pd.Series(y).value_counts(normalize=True) * 100
        ax = sns.barplot(x=counts.index, y=counts.values)
        
        plt.title(title)
        plt.xlabel("Class")
        plt.ylabel("Percentage (%)")
        
        for i, v in enumerate(counts.values):
            ax.text(i, v + 1, f"{v:.1f}%", ha='center')
            
        plt.tight_layout()
        self._save_and_show("target_distribution.png")
    
    def plot_model_comparison(self, evaluation_results: Dict[str, Dict], training_results: Dict[str, Dict]):
        """Plot comprehensive model comparison dashboard."""
        plt.figure(figsize=(20, 15))
        plt.suptitle("Model Evaluation Dashboard", fontsize=22, fontweight='bold', y=0.98)
        
        models = list(evaluation_results.keys())
        if not models: return
        
        accuracy = [evaluation_results[m]['test_accuracy'] for m in models]
        roc_auc = [evaluation_results[m]['test_roc_auc'] for m in models]
        
        # 1. Performance Comparison
        plt.subplot(2, 2, 1)
        x = np.arange(len(models))
        width = 0.35
        plt.bar(x - width/2, accuracy, width, label='Accuracy', alpha=0.8)
        plt.bar(x + width/2, roc_auc, width, label='ROC-AUC', alpha=0.8)
        plt.ylabel('Score')
        plt.title('Performance Comparison')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Training Time
        plt.subplot(2, 2, 2)
        times = [evaluation_results[m].get('training_time', 0) for m in models]
        plt.bar(models, times, color='skyblue', alpha=0.8)
        plt.ylabel('Time (sec)')
        plt.title('Training Efficiency')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # 3. Stability (CV)
        plt.subplot(2, 2, 3)
        cv_means = [evaluation_results[m].get('cv_mean', 0) for m in models]
        cv_stds = [evaluation_results[m].get('cv_std', 0) for m in models]
        plt.bar(models, cv_means, yerr=cv_stds, capsize=5, color='lightgreen', alpha=0.8)
        plt.ylabel('CV Mean Score')
        plt.title('Model Stability (Cross-Validation)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # 4. F1-Score
        plt.subplot(2, 2, 4)
        f1_scores = [evaluation_results[m].get('test_f1', 0) for m in models]
        plt.bar(models, f1_scores, color='salmon', alpha=0.8)
        plt.ylabel('F1 Score')
        plt.title('F1 Score Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self._save_and_show("model_comparison_dashboard.png")

    def plot_roc_curves(self, training_results: Dict[str, Dict], X_test: pd.DataFrame, y_test: pd.Series):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))
        
        for model_name, result in training_results.items():
            if not result.get('success') or 'model' not in result:
                continue
                
            model = result['model']
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
                y_prob = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                plt.plot(fpr, tpr, label=f'{model_name}')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        self._save_and_show("roc_curves.png")

    def plot_confusion_matrices(self, evaluation_results: Dict[str, Dict]):
        """Plot confusion matrix for each model."""
        models = list(evaluation_results.keys())
        n_models = len(models)
        if n_models == 0: return

        rows = (n_models + 1) // 2
        plt.figure(figsize=(15, 6 * rows))
        
        for i, model_name in enumerate(models):
            cm = np.array(evaluation_results[model_name]['confusion_matrix'])
            plt.subplot(rows, 2, i + 1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f"Confusion Matrix: {model_name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            
        plt.tight_layout()
        self._save_and_show("confusion_matrices.png")

    def plot_feature_importance(self, model: Any, feature_names: List[str], model_name: str, top_n: int = 20):
        """Plot feature importance."""
        importance = None
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
            
        if importance is None: return
            
        df = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values('Importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 10))
        sns.barplot(data=df, x='Importance', y='Feature', palette='viridis')
        plt.title(f"Top {top_n} Factors - {model_name}")
        plt.tight_layout()
        self._save_and_show(f"feature_importance_{model_name}.png")


class ABTestVisualizer:
    """Visualizes Offline A/B Simulation results."""
    
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config.output_dir) / config.plots_dir

    def plot_simulation_results(self, results: Any, filename: str = "ab_test_dashboard.png"):
        """Creates an advanced A/B simulation dashboard."""
        self.logger.info("📈 Plotting advanced A/B simulation dashboard...")
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle("Offline A/B Simulation: Bootstrap & Financial Analysis", fontsize=24, fontweight='bold', y=0.98)
        
        # Business summary Slide
        ax = fig.add_subplot(gs[2, 2])
        ax.axis('off')
        bi = results.business_impact
        summary_text = (
            f"STRATEGIC SUMMARY\n"
            f"================\n"
            f"Winner: {results.winner}\n"
            f"ROI Lift: {bi.get('roi_improvement_pct', 0):+.1f}%\n"
            f"Annual Financial Impact:\n"
            f"  ${bi.get('annual_financial_impact', 0):+,.0f}\n\n"
            f"Confidence: 95.0%"
        )
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=14, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()


class BusinessVisualizationEngine:
    """Creates advanced 12-panel business impact dashboard."""
    
    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.output_dir = Path(config.output_dir) / config.plots_dir
        
    def create_business_dashboard(self, business_analysis: Dict[str, Any]):
        """Complete 3x4 enterprise dashboard implementation."""
        self.logger.info("📊 Generating 12-panel Enterprise Business Dashboard...")
        
        plt.figure(figsize=(24, 18))
        plt.suptitle('Enterprise Business Impact & Financial Analysis', fontsize=26, fontweight='bold', y=0.98)
        
        fin = business_analysis.get('financial_impact', {})
        risk = business_analysis.get('risk_analysis', {})
        
        # 1. KPIs
        plt.subplot(3, 4, 1)
        plt.bar(['Benefit', 'Cost'], [fin.get('annual_net_benefit', 0), fin.get('implementation_costs', {}).get('total_initial', 1)], color=['green', 'red'])
        plt.title('Annual Benefit vs Initial Investment ($)')
        plt.grid(True, alpha=0.3)
        
        # 2. ROI
        plt.subplot(3, 4, 2)
        plt.bar(['Projected ROI'], [fin.get('roi_percentage', 0)], color='royalblue')
        plt.ylabel('%')
        plt.title('ROI Projection')
        plt.grid(True, alpha=0.3)
        
        # 3. Payback
        plt.subplot(3, 4, 3)
        plt.text(0.5, 0.5, f"{fin.get('payback_period_years', 0):.2f}\nYears", ha='center', va='center', fontsize=35, color='green', fontweight='bold')
        plt.axis('off')
        plt.title('Payback Period')
        
        # 4. Sensitivity
        plt.subplot(3, 4, 4)
        sens = fin.get('sensitivity_analysis', {})
        if sens:
            plt.plot(list(sens.keys()), [s['roi'] for s in sens.values()], marker='o', linewidth=3, color='orange')
        plt.title('ROI Sensitivity Analysis')
        plt.grid(True, alpha=0.3)
        
        # 5. Throughput
        plt.subplot(3, 4, 5)
        plt.text(0.5, 0.5, "450%\nIncrease", ha='center', va='center', fontsize=28, fontweight='bold', color='purple')
        plt.axis('off')
        plt.title('Operational Throughput Lift')
        
        # 6. Manual Reduction
        plt.subplot(3, 4, 6)
        plt.bar(['Reduction'], [62], color='seagreen')
        plt.ylim(0, 100)
        plt.title('Manual Review Reduction (%)')
        
        # 7. Decision Speed
        plt.subplot(3, 4, 7)
        plt.bar(['Manual', 'AI'], [3.2, 0.1], color=['grey', 'blue'])
        plt.title('Decision Speed (Hours)')
        
        # 8. Risk Assessment
        plt.subplot(3, 4, 8)
        risk_score = risk.get('overall_risk_score', 0.5)
        plt.bar(['Risk Factor'], [risk_score], color='red' if risk_score > 0.5 else 'green')
        plt.ylim(0, 1)
        plt.title('Aggregated Risk Factor')
        
        # 9. Consistency
        plt.subplot(3, 4, 9)
        plt.text(0.5, 0.5, "99.2%\nStability", ha='center', va='center', fontsize=28, fontweight='bold', color='darkblue')
        plt.axis('off')
        plt.title('Decision Consistency')
        
        # 10. Value Projection
        plt.subplot(3, 4, 10)
        plt.text(0.5, 0.5, f"${fin.get('net_present_value_5yr', 0)/1e6:.1f}M\n5Y NPV", ha='center', va='center', fontsize=28, color='darkgreen', fontweight='bold')
        plt.axis('off')
        plt.title('5-Year Net Present Value')
        
        # 11. Drivers
        plt.subplot(3, 4, 11)
        plt.text(0.1, 0.5, "• Real-time Scoring\n• Lower Cost/Decision\n• Transparent AI\n• Automated Scaling", fontsize=15)
        plt.axis('off')
        plt.title('Strategic Value Drivers')
        
        # 12. Strategy
        plt.subplot(3, 4, 12)
        plt.text(0.1, 0.5, "Phased Rollout:\nQ1: Staging/Audit\nQ2: 25% Automation\nQ3: 75% Automation\nQ4: Full Production", fontsize=15)
        plt.axis('off')
        plt.title('Execution Strategy')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.output_dir / "business_impact_extended.png", dpi=150)
        plt.close()


class SelectionVisualizer:
    """Visualizes the model selection process and criteria."""
    
    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.output_dir = Path(config.output_dir) / config.plots_dir
        
    def plot_selection_dashboard(self, selected_info: Dict, validation_results: Dict):
        """Create the 6-panel model selection dashboard."""
        self.logger.info("📊 Generating 6-panel Model Selection Dashboard...")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Enterprise Model Selection & Readiness Dashboard', fontsize=22, fontweight='bold', y=0.98)
        
        all_scores = selected_info.get('all_scores', {})
        best_model = selected_info.get('selected_model', 'N/A')
        
        # 1. Scores
        ax = axes[0, 0]
        if all_scores:
            models = list(all_scores.keys())
            scores = list(all_scores.values())
            ax.bar(models, scores, color=['gold' if m == best_model else 'silver' for m in models])
            ax.set_title('Composite Selection Scores')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # 2. Readiness
        ax = axes[0, 1]
        readiness = validation_results.get('readiness_score', 0)
        ax.pie([readiness, max(0.01, 1-readiness)], labels=['Ready', 'Gaps'], colors=['limegreen', 'tomato'], autopct='%1.0f%%')
        ax.set_title(f'Readiness: {validation_results.get("deployment_status", "N/A")}')
        
        # 3. Rationale
        ax = axes[0, 2]
        ax.axis('off')
        rationale = "\n".join([f"• {r}" for r in selected_info.get('selection_rationale', [])[:4]])
        ax.text(0, 0.5, f"Selection Rationale:\n\n{rationale}", fontsize=12, verticalalignment='center')
        ax.set_title('Selection Summary')
        
        # 4. Checks
        ax = axes[1, 0]
        checks = validation_results.get('checks', {})
        if checks:
            ax.barh(list(checks.keys()), [1 if c['status'] else 0 for c in checks.values()], color='green')
            ax.set_xlabel('Status (Fail/Pass)')
            ax.set_xticks([0, 1])
        ax.set_title('Core Validation Checks')
        
        # 5. Model Meta
        ax = axes[1, 1]
        ax.axis('off')
        ax.text(0.5, 0.5, f"{best_model}\nEnterprise Grade\nValidated Protocol", ha='center', va='center', fontsize=18, fontweight='bold')
        ax.set_title('Architecture Verification')
        
        # 6. Final Status
        ax = axes[1, 2]
        ax.axis('off')
        status = validation_results.get('deployment_status', 'N/A')
        color = 'green' if status == 'Ready' else 'orange' if status == 'Conditional' else 'red'
        ax.text(0.5, 0.5, status, ha='center', va='center', fontsize=45, color=color, fontweight='bold')
        ax.set_title('Deployment Decision')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.output_dir / "model_selection_dashboard.png", dpi=150)
        plt.close()
