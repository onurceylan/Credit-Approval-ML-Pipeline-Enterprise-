"""
Evaluation Module
=================

Contains classes for statistical validation, model evaluation,
business impact analysis, model selection, and visualization.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)

from .config import ModelConfig, OUTPUT_FILES, PLOT_FILES
from .utils import handle_errors, DependencyManager


# =============================================================================
# STATISTICAL VALIDATOR
# =============================================================================

class StatisticalValidator:
    """
    Performs statistical validation of model comparisons.
    
    Uses Friedman test and post-hoc analysis to determine
    statistically significant differences between models.
    """
    
    def __init__(
        self, 
        config: ModelConfig, 
        dependency_manager: DependencyManager, 
        logger: logging.Logger
    ):
        self.config = config
        self.dependency_manager = dependency_manager
        self.logger = logger
        self.scipy_available = dependency_manager.is_available('scipy')
        
        if self.scipy_available:
            self.scipy_stats = dependency_manager.get_package('scipy')
    
    @handle_errors
    def run_statistical_tests(
        self, 
        training_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run comprehensive statistical tests on model results.
        
        Args:
            training_results: Dictionary of training results
        
        Returns:
            Statistical test results dictionary
        """
        self.logger.info("📊 Running statistical validation...")
        
        if not self.scipy_available:
            self.logger.warning("   ⚠️ SciPy not available, skipping statistical tests")
            return {'scipy_available': False}
        
        successful_models = {
            name: result for name, result in training_results.items()
            if result.get('success', False)
        }
        
        if len(successful_models) < 3:
            self.logger.warning("   ⚠️ Need at least 3 models for statistical comparison")
            return {'sufficient_models': False}
        
        # Collect CV scores for all models
        cv_scores_matrix = []
        model_names = []
        
        for name, result in successful_models.items():
            cv_results = result.get('cv_results', {})
            cv_scores = cv_results.get('cv_scores', [])
            
            if cv_scores:
                cv_scores_matrix.append(cv_scores)
                model_names.append(name)
        
        if len(cv_scores_matrix) < 3:
            self.logger.warning("   ⚠️ Insufficient CV scores for statistical testing")
            return {'sufficient_cv_scores': False}
        
        # Run Friedman test
        try:
            from scipy.stats import friedmanchisquare
            
            stat, p_value = friedmanchisquare(*cv_scores_matrix)
            
            self.logger.info(f"   📊 Friedman Test:")
            self.logger.info(f"      • Statistic: {stat:.4f}")
            self.logger.info(f"      • P-value: {p_value:.6f}")
            
            # Determine significance
            alpha = self.config.statistical_alpha
            if self.config.bonferroni_correction:
                adjusted_alpha = alpha / len(model_names)
            else:
                adjusted_alpha = alpha
            
            significant = p_value < adjusted_alpha
            
            if significant:
                self.logger.info(f"   ✅ Significant difference detected (p < {adjusted_alpha:.4f})")
            else:
                self.logger.info(f"   ℹ️ No significant difference (p >= {adjusted_alpha:.4f})")
            
            return {
                'scipy_available': True,
                'friedman_statistic': stat,
                'friedman_p_value': p_value,
                'alpha': alpha,
                'adjusted_alpha': adjusted_alpha,
                'significant_difference': significant,
                'model_names': model_names,
                'n_models': len(model_names)
            }
            
        except Exception as e:
            self.logger.error(f"   ❌ Statistical test failed: {str(e)}")
            return {'error': str(e)}


# =============================================================================
# MODEL EVALUATOR
# =============================================================================

class ModelEvaluator:
    """
    Comprehensive model evaluation on test set.
    
    Generates detailed performance metrics, confusion matrices,
    and classification reports.
    """
    
    def __init__(self, config: ModelConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    @handle_errors
    def evaluate_all_models(
        self, 
        training_results: Dict[str, Any],
        processed_splits: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate all trained models on test set.
        
        Args:
            training_results: Dictionary of training results
            processed_splits: Processed data splits
        
        Returns:
            Evaluation results dictionary
        """
        self.logger.info("📊 Evaluating models on test set...")
        
        X_test = processed_splits['X_test']
        y_test = processed_splits['y_test']
        
        evaluation_results = {}
        
        for model_name, result in training_results.items():
            if not result.get('success', False):
                continue
            
            if 'model' not in result:
                continue
            
            model = result['model']
            
            try:
                eval_result = self._evaluate_single_model(model, model_name, X_test, y_test)
                evaluation_results[model_name] = eval_result
                
                self.logger.info(f"   ✅ {model_name}: Test Accuracy={eval_result['test_accuracy']:.4f}, Test AUC={eval_result['test_roc_auc']:.4f}")
                
            except Exception as e:
                self.logger.error(f"   ❌ {model_name} evaluation failed: {str(e)}")
                evaluation_results[model_name] = {'error': str(e)}
        
        # Save evaluation report
        self._save_evaluation_report(evaluation_results)
        
        return evaluation_results
    
    def _evaluate_single_model(
        self, 
        model, 
        model_name: str,
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate a single model."""
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            test_roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        else:
            y_pred_proba = None
            test_roc_auc = 0.0
        
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'model_name': model_name,
            'test_accuracy': test_accuracy,
            'test_roc_auc': test_roc_auc,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
        }
    
    def _save_evaluation_report(self, evaluation_results: Dict[str, Any]):
        """Save evaluation report to JSON."""
        try:
            report_data = {}
            for model_name, result in evaluation_results.items():
                if 'error' not in result:
                    clean_result = {k: v for k, v in result.items() 
                                   if k not in ['predictions', 'probabilities']}
                    report_data[model_name] = clean_result
            
            report_path = Path(self.config.output_dir) / self.config.results_dir / OUTPUT_FILES['evaluation_report']
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"   💾 Evaluation report saved: {report_path}")
            
        except Exception as e:
            self.logger.warning(f"   ⚠️ Could not save evaluation report: {e}")


# =============================================================================
# BUSINESS IMPACT ANALYST
# =============================================================================

class BusinessImpactAnalyst:
    """
    Analyzes business impact of model predictions.
    
    Calculates cost-benefit analysis, ROI, and business metrics.
    """
    
    def __init__(self, config: ModelConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    @handle_errors
    def analyze_business_impact(
        self, 
        evaluation_results: Dict[str, Any],
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculate business impact metrics for all models.
        
        Args:
            evaluation_results: Model evaluation results
            y_test: Test target values
        
        Returns:
            Business impact analysis dictionary
        """
        self.logger.info("💰 Analyzing business impact...")
        
        business_results = {}
        
        for model_name, result in evaluation_results.items():
            if 'error' in result:
                continue
            
            try:
                impact = self._calculate_impact(result, y_test)
                business_results[model_name] = impact
                
                self.logger.info(f"   💵 {model_name}: Net Profit=${impact['net_profit']:,.0f}")
                
            except Exception as e:
                self.logger.warning(f"   ⚠️ {model_name} impact calculation failed: {e}")
        
        # Save business case document
        self._generate_business_case(business_results)
        
        return business_results
    
    def _calculate_impact(
        self, 
        result: Dict[str, Any], 
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Calculate detailed business impact."""
        
        predictions = np.array(result['predictions'])
        y_true = y_test.values
        
        # Calculate confusion matrix for binary bad credit (class 1)
        # True Positive: Predicted bad, was bad
        # False Positive: Predicted bad, was good
        # True Negative: Predicted good, was good
        # False Negative: Predicted good, was bad
        
        # For simplicity, treat class 1 as "bad credit"
        is_predicted_bad = predictions == 1
        is_actual_bad = y_true == 1
        
        tp = np.sum(is_predicted_bad & is_actual_bad)
        fp = np.sum(is_predicted_bad & ~is_actual_bad)
        tn = np.sum(~is_predicted_bad & ~is_actual_bad)
        fn = np.sum(~is_predicted_bad & is_actual_bad)
        
        # Cost calculations
        cost_fp = fp * self.config.cost_false_negative  # Rejected good customer
        cost_fn = fn * self.config.cost_false_positive  # Approved bad customer
        revenue_tp = tp * self.config.revenue_per_approval  # Correctly identified bad (avoided loss)
        revenue_tn = tn * self.config.revenue_per_approval  # Correctly approved good customer
        
        total_cost = cost_fp + cost_fn
        total_revenue = revenue_tp + revenue_tn
        net_profit = total_revenue - total_cost
        
        # Calculate baseline (random model)
        random_accuracy = max(np.mean(y_true == 0), np.mean(y_true == 1))
        
        return {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'cost_false_positives': float(cost_fp),
            'cost_false_negatives': float(cost_fn),
            'total_cost': float(total_cost),
            'revenue_true_positives': float(revenue_tp),
            'revenue_true_negatives': float(revenue_tn),
            'total_revenue': float(total_revenue),
            'net_profit': float(net_profit),
            'roi': float(net_profit / total_cost * 100) if total_cost > 0 else 0,
            'baseline_accuracy': float(random_accuracy)
        }
    
    def _generate_business_case(self, business_results: Dict[str, Any]):
        """Generate business case document."""
        try:
            best_model = max(
                business_results.items(),
                key=lambda x: x[1]['net_profit']
            )
            
            doc_lines = [
                "=" * 60,
                "BUSINESS CASE DOCUMENT",
                "Credit Approval ML Pipeline",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "=" * 60,
                "",
                "EXECUTIVE SUMMARY",
                "-" * 40,
                f"Best Model: {best_model[0]}",
                f"Expected Net Profit: ${best_model[1]['net_profit']:,.0f}",
                f"Return on Investment: {best_model[1]['roi']:.1f}%",
                "",
                "MODEL COMPARISON",
                "-" * 40
            ]
            
            for model_name, impact in sorted(
                business_results.items(),
                key=lambda x: x[1]['net_profit'],
                reverse=True
            ):
                doc_lines.extend([
                    f"\n{model_name}:",
                    f"  Net Profit: ${impact['net_profit']:,.0f}",
                    f"  ROI: {impact['roi']:.1f}%",
                    f"  False Positive Cost: ${impact['cost_false_positives']:,.0f}",
                    f"  False Negative Cost: ${impact['cost_false_negatives']:,.0f}"
                ])
            
            doc_lines.extend([
                "",
                "RECOMMENDATION",
                "-" * 40,
                f"We recommend deploying the {best_model[0]} model.",
                f"Expected annual benefit: ${best_model[1]['net_profit']:,.0f}",
                "",
                "=" * 60
            ])
            
            doc_path = Path(self.config.output_dir) / self.config.results_dir / OUTPUT_FILES['business_case']
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(doc_path, 'w') as f:
                f.write('\n'.join(doc_lines))
            
            self.logger.info(f"   💾 Business case saved: {doc_path}")
            
        except Exception as e:
            self.logger.warning(f"   ⚠️ Could not generate business case: {e}")


# =============================================================================
# MODEL SELECTOR
# =============================================================================

class ModelSelector:
    """
    Selects the best model based on multiple criteria.
    
    Considers accuracy, ROC-AUC, training time, and business impact.
    """
    
    def __init__(self, config: ModelConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    @handle_errors
    def select_best_model(
        self, 
        training_results: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        business_results: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select best model using multiple criteria.
        
        Args:
            training_results: Training results dictionary
            evaluation_results: Evaluation results dictionary
            business_results: Optional business impact results
        
        Returns:
            Tuple of (best_model_name, selection_rationale)
        """
        self.logger.info("🏆 Selecting best model...")
        
        model_scores = {}
        
        for model_name, train_result in training_results.items():
            if not train_result.get('success', False):
                continue
            
            eval_result = evaluation_results.get(model_name, {})
            
            if 'error' in eval_result:
                continue
            
            # Calculate composite score
            val_auc = train_result.get('val_roc_auc', 0)
            test_auc = eval_result.get('test_roc_auc', 0)
            cv_mean = train_result.get('cv_results', {}).get('cv_mean', 0)
            cv_std = train_result.get('cv_results', {}).get('cv_std', 1)
            
            # Stability score (low variance is good)
            stability_score = 1 / (1 + cv_std)
            
            # Business impact score
            if business_results and model_name in business_results:
                business_score = business_results[model_name]['net_profit'] / 1e6
            else:
                business_score = 0
            
            # Composite score (weighted)
            composite_score = (
                0.30 * test_auc +
                0.25 * val_auc +
                0.20 * cv_mean +
                0.15 * stability_score +
                0.10 * min(business_score, 1)  # Cap business score contribution
            )
            
            model_scores[model_name] = {
                'composite_score': composite_score,
                'test_auc': test_auc,
                'val_auc': val_auc,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'stability_score': stability_score,
                'business_score': business_score
            }
        
        if not model_scores:
            self.logger.error("   ❌ No models available for selection")
            return None, {}
        
        # Select best model
        best_model_name = max(
            model_scores.keys(),
            key=lambda x: model_scores[x]['composite_score']
        )
        
        best_scores = model_scores[best_model_name]
        
        self.logger.info(f"   🥇 Best Model: {best_model_name}")
        self.logger.info(f"      • Composite Score: {best_scores['composite_score']:.4f}")
        self.logger.info(f"      • Test ROC-AUC: {best_scores['test_auc']:.4f}")
        self.logger.info(f"      • CV Mean: {best_scores['cv_mean']:.4f} ± {best_scores['cv_std']:.4f}")
        
        # Generate selection rationale
        rationale = {
            'selected_model': best_model_name,
            'selection_criteria': model_scores,
            'timestamp': datetime.now().isoformat(),
            'weights': {
                'test_auc': 0.30,
                'val_auc': 0.25,
                'cv_mean': 0.20,
                'stability': 0.15,
                'business': 0.10
            }
        }
        
        # Save executive summary
        self._generate_executive_summary(best_model_name, model_scores, training_results)
        
        return best_model_name, rationale
    
    def _generate_executive_summary(
        self, 
        best_model: str,
        model_scores: Dict[str, Dict],
        training_results: Dict[str, Any]
    ):
        """Generate executive summary report."""
        try:
            sorted_models = sorted(
                model_scores.items(),
                key=lambda x: x[1]['composite_score'],
                reverse=True
            )
            
            lines = [
                "=" * 70,
                "EXECUTIVE SUMMARY REPORT",
                "Credit Approval ML Pipeline - Model Selection",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "=" * 70,
                "",
                "SELECTED MODEL",
                "-" * 40,
                f"  Model: {best_model}",
                f"  Composite Score: {model_scores[best_model]['composite_score']:.4f}",
                f"  Test ROC-AUC: {model_scores[best_model]['test_auc']:.4f}",
                f"  Cross-Validation: {model_scores[best_model]['cv_mean']:.4f} ± {model_scores[best_model]['cv_std']:.4f}",
                "",
                "MODEL RANKING",
                "-" * 40
            ]
            
            for rank, (name, scores) in enumerate(sorted_models, 1):
                lines.append(
                    f"  {rank}. {name:<20} Score: {scores['composite_score']:.4f} "
                    f"| Test AUC: {scores['test_auc']:.4f}"
                )
            
            lines.extend([
                "",
                "DEPLOYMENT READINESS",
                "-" * 40
            ])
            
            best_scores = model_scores[best_model]
            
            if best_scores['test_auc'] >= self.config.deployment_accuracy_threshold:
                lines.append("  ✅ Accuracy threshold met")
            else:
                lines.append("  ⚠️ Accuracy below deployment threshold")
            
            if best_scores['cv_std'] <= self.config.deployment_stability_threshold:
                lines.append("  ✅ Stability threshold met")
            else:
                lines.append("  ⚠️ Model variance above threshold")
            
            lines.extend([
                "",
                "RECOMMENDATION",
                "-" * 40,
                f"  Deploy {best_model} to production.",
                "",
                "=" * 70
            ])
            
            summary_path = Path(self.config.output_dir) / self.config.results_dir / OUTPUT_FILES['executive_summary']
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(summary_path, 'w') as f:
                f.write('\n'.join(lines))
            
            self.logger.info(f"   💾 Executive summary saved: {summary_path}")
            
        except Exception as e:
            self.logger.warning(f"   ⚠️ Could not generate executive summary: {e}")


# =============================================================================
# EVALUATION VISUALIZER
# =============================================================================

class EvaluationVisualizer:
    """
    Creates evaluation visualizations.
    
    Generates confusion matrices, ROC curves, feature importance,
    and business impact charts.
    """
    
    def __init__(self, config: ModelConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    @handle_errors
    def create_evaluation_visualizations(
        self, 
        training_results: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        processed_splits: Dict[str, Any]
    ) -> None:
        """
        Create comprehensive evaluation visualizations.
        
        Args:
            training_results: Training results dictionary
            evaluation_results: Evaluation results dictionary
            processed_splits: Processed data splits
        """
        self.logger.info("📊 Creating evaluation visualizations...")
        
        # Create model comparison plot
        self._create_model_comparison(evaluation_results)
        
        # Create confusion matrices
        self._create_confusion_matrices(evaluation_results)
        
        # Create feature importance (if available)
        self._create_feature_importance(training_results, processed_splits)
    
    def _create_model_comparison(self, evaluation_results: Dict[str, Any]):
        """Create model comparison visualization."""
        successful_models = {
            name: result for name, result in evaluation_results.items()
            if 'error' not in result
        }
        
        if not successful_models:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Model Evaluation Comparison', fontsize=14, fontweight='bold')
        
        model_names = list(successful_models.keys())
        metrics = ['test_accuracy', 'test_roc_auc', 'test_f1', 'test_precision', 'test_recall']
        
        # Metrics comparison
        data = []
        for name in model_names:
            row = [successful_models[name].get(m, 0) for m in metrics]
            data.append(row)
        
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            values = [successful_models[name].get(metric, 0) for name in model_names]
            axes[0].bar(x + i * width, values, width, label=metric.replace('test_', '').replace('_', ' ').title())
        
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Test Set Metrics Comparison')
        axes[0].set_xticks(x + width * 2)
        axes[0].set_xticklabels([n[:12] for n in model_names], rotation=45)
        axes[0].legend(loc='lower right', fontsize=8)
        axes[0].grid(True, alpha=0.3)
        
        # ROC-AUC ranking
        sorted_models = sorted(
            [(name, result['test_roc_auc']) for name, result in successful_models.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        names, scores = zip(*sorted_models)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
        
        bars = axes[1].barh(range(len(names)), scores, color=colors)
        axes[1].set_yticks(range(len(names)))
        axes[1].set_yticklabels(names)
        axes[1].set_xlabel('Test ROC-AUC')
        axes[1].set_title('Model Ranking by ROC-AUC')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        for i, (bar, score) in enumerate(zip(bars, scores)):
            axes[1].text(score + 0.01, i, f'{score:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        plot_path = Path(self.config.output_dir) / self.config.plots_dir / PLOT_FILES['model_comparison']
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"   💾 Model comparison plot saved: {plot_path}")
        
        plt.show()
        plt.close()
    
    def _create_confusion_matrices(self, evaluation_results: Dict[str, Any]):
        """Create confusion matrix visualizations."""
        successful_models = {
            name: result for name, result in evaluation_results.items()
            if 'error' not in result and 'confusion_matrix' in result
        }
        
        if not successful_models:
            return
        
        n_models = len(successful_models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
        
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, (name, result) in enumerate(successful_models.items()):
            cm = np.array(result['confusion_matrix'])
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                ax=axes[i],
                cbar=False
            )
            axes[i].set_title(f'{name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        plot_path = Path(self.config.output_dir) / self.config.plots_dir / PLOT_FILES['confusion_matrices']
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"   💾 Confusion matrices saved: {plot_path}")
        
        plt.show()
        plt.close()
    
    def _create_feature_importance(
        self, 
        training_results: Dict[str, Any],
        processed_splits: Dict[str, Any]
    ):
        """Create feature importance visualization."""
        feature_names = processed_splits.get('preprocessing_info', {}).get('final_features', [])
        
        if not feature_names:
            return
        
        importance_data = {}
        
        for model_name, result in training_results.items():
            if not result.get('success', False):
                continue
            
            model = result.get('model')
            if model is None:
                continue
            
            try:
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    importance_data[model_name] = importance
                elif hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_).mean(axis=0)
                    importance_data[model_name] = importance
            except Exception:
                continue
        
        if not importance_data:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('Feature Importance Comparison', fontsize=14, fontweight='bold')
        
        # Average importance across models
        avg_importance = np.zeros(len(feature_names))
        for importance in importance_data.values():
            if len(importance) == len(feature_names):
                avg_importance += importance
        avg_importance /= len(importance_data)
        
        # Sort by importance
        sorted_idx = np.argsort(avg_importance)[-20:]  # Top 20 features
        
        y_pos = np.arange(len(sorted_idx))
        ax.barh(y_pos, avg_importance[sorted_idx], alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in sorted_idx])
        ax.set_xlabel('Average Importance')
        ax.set_title('Top 20 Features by Average Importance')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        plot_path = Path(self.config.output_dir) / self.config.plots_dir / PLOT_FILES['feature_importance']
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"   💾 Feature importance plot saved: {plot_path}")
        
        plt.show()
        plt.close()
    
    @handle_errors
    def create_business_impact_visualization(
        self, 
        business_results: Dict[str, Any]
    ) -> None:
        """Create business impact visualization."""
        if not business_results:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Business Impact Analysis', fontsize=14, fontweight='bold')
        
        model_names = list(business_results.keys())
        
        # Net profit comparison
        net_profits = [business_results[name]['net_profit'] for name in model_names]
        colors = ['green' if p > 0 else 'red' for p in net_profits]
        
        bars = axes[0].bar(model_names, net_profits, color=colors, alpha=0.7)
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('Net Profit ($)')
        axes[0].set_title('Net Profit by Model')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0].grid(True, alpha=0.3)
        
        for bar, profit in zip(bars, net_profits):
            y_pos = bar.get_height() + max(net_profits) * 0.01
            axes[0].text(
                bar.get_x() + bar.get_width()/2, 
                y_pos,
                f'${profit:,.0f}', 
                ha='center', 
                va='bottom', 
                fontsize=8,
                rotation=45
            )
        
        # Cost breakdown
        sorted_models = sorted(
            model_names,
            key=lambda x: business_results[x]['net_profit'],
            reverse=True
        )[:5]  # Top 5
        
        costs = []
        revenues = []
        
        for name in sorted_models:
            costs.append(business_results[name]['total_cost'])
            revenues.append(business_results[name]['total_revenue'])
        
        x = np.arange(len(sorted_models))
        width = 0.35
        
        axes[1].bar(x - width/2, costs, width, label='Total Cost', color='red', alpha=0.7)
        axes[1].bar(x + width/2, revenues, width, label='Total Revenue', color='green', alpha=0.7)
        
        axes[1].set_xlabel('Models')
        axes[1].set_ylabel('Amount ($)')
        axes[1].set_title('Cost vs Revenue (Top 5 Models)')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([n[:12] for n in sorted_models], rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = Path(self.config.output_dir) / self.config.plots_dir / PLOT_FILES['business_impact']
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"   💾 Business impact plot saved: {plot_path}")
        
        plt.show()
        plt.close()
    
    @handle_errors
    def create_model_selection_visualization(
        self, 
        selection_rationale: Dict[str, Any]
    ) -> None:
        """Create model selection visualization."""
        criteria = selection_rationale.get('selection_criteria', {})
        
        if not criteria:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle('Model Selection Analysis', fontsize=14, fontweight='bold')
        
        model_names = list(criteria.keys())
        composite_scores = [criteria[name]['composite_score'] for name in model_names]
        
        # Sort by composite score
        sorted_idx = np.argsort(composite_scores)[::-1]
        sorted_names = [model_names[i] for i in sorted_idx]
        sorted_scores = [composite_scores[i] for i in sorted_idx]
        
        colors = ['gold'] + ['steelblue'] * (len(sorted_names) - 1)
        
        bars = ax.barh(range(len(sorted_names)), sorted_scores, color=colors, alpha=0.8)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Composite Score')
        ax.set_title(f'Model Ranking - Selected: {selection_rationale.get("selected_model", "N/A")}')
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            ax.text(score + 0.01, i, f'{score:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        plot_path = Path(self.config.output_dir) / self.config.plots_dir / PLOT_FILES['model_selection']
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"   💾 Model selection plot saved: {plot_path}")
        
        plt.show()
        plt.close()


# =============================================================================
# COMPREHENSIVE EVALUATION RUNNER
# =============================================================================

def run_comprehensive_evaluation(
    training_results: Dict[str, Any],
    processed_splits: Dict[str, Any],
    config: ModelConfig,
    dependency_manager: DependencyManager,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Run complete evaluation pipeline.
    
    Args:
        training_results: Dictionary of training results
        processed_splits: Processed data splits
        config: Model configuration
        dependency_manager: Dependency manager
        logger: Logger instance
    
    Returns:
        Complete evaluation results dictionary
    """
    logger.info("\n" + "=" * 60)
    logger.info("📊 COMPREHENSIVE MODEL EVALUATION")
    logger.info("=" * 60)
    
    # Initialize components
    statistical_validator = StatisticalValidator(config, dependency_manager, logger)
    model_evaluator = ModelEvaluator(config, logger)
    business_analyst = BusinessImpactAnalyst(config, logger)
    model_selector = ModelSelector(config, logger)
    visualizer = EvaluationVisualizer(config, logger)
    
    # Run evaluation pipeline
    evaluation_results = model_evaluator.evaluate_all_models(training_results, processed_splits)
    
    statistical_results = statistical_validator.run_statistical_tests(training_results)
    
    business_results = business_analyst.analyze_business_impact(
        evaluation_results, 
        processed_splits['y_test']
    )
    
    best_model_name, selection_rationale = model_selector.select_best_model(
        training_results, 
        evaluation_results, 
        business_results
    )
    
    # Create visualizations
    visualizer.create_evaluation_visualizations(training_results, evaluation_results, processed_splits)
    visualizer.create_business_impact_visualization(business_results)
    visualizer.create_model_selection_visualization(selection_rationale)
    
    logger.info("\n✅ Comprehensive evaluation completed!")
    
    return {
        'evaluation_results': evaluation_results,
        'statistical_results': statistical_results,
        'business_results': business_results,
        'best_model': best_model_name,
        'selection_rationale': selection_rationale
    }
