"""
Training Module
===============

Contains classes for hyperparameter optimization, model training,
persistence, and visualization of training results.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

from .config import ModelConfig, OUTPUT_FILES, PLOT_FILES
from .utils import handle_errors, memory_cleanup, DependencyManager
from .models import ModelFactory


# =============================================================================
# HYPERPARAMETER OPTIMIZER
# =============================================================================

class HyperparameterOptimizer:
    """
    Handles hyperparameter optimization using Optuna.
    
    Falls back to default parameters if Optuna is not available.
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
        self.optuna_available = dependency_manager.is_available('optuna')
        
        if self.optuna_available:
            self.optuna = dependency_manager.get_package('optuna')
    
    @handle_errors
    def optimize_hyperparameters(
        self, 
        model_config: Dict, 
        X_train: pd.DataFrame,
        y_train: pd.Series, 
        cv_folds: List
    ) -> Tuple[Dict, float]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            model_config: Model configuration dictionary
            X_train: Training features
            y_train: Training target
            cv_folds: Cross-validation fold indices
        
        Returns:
            Tuple of (best_params, best_cv_score)
        """
        if not self.optuna_available:
            self.logger.warning("   ⚠️ Optuna not available, using default parameters")
            return model_config['params'], 0.0
        
        self.logger.info(f"   🔍 Optimizing hyperparameters with Optuna...")
        self.logger.info(f"      • Trials: {self.config.optuna_trials}")
        self.logger.info(f"      • Timeout: {self.config.optuna_timeout}s")
        
        # Create study
        study = self.optuna.create_study(
            direction='maximize',
            sampler=self.optuna.samplers.TPESampler(seed=self.config.random_state),
            pruner=self.optuna.pruners.MedianPruner(n_startup_trials=5)
        )
        
        # Define objective function
        def objective(trial):
            try:
                params = self._suggest_parameters(trial, model_config['param_space'])
                final_params = {**model_config['params'], **params}
                model = model_config['class'](**final_params)
                
                cv_scores = []
                for train_idx, val_idx in cv_folds[:3]:  # Use first 3 folds for speed
                    X_fold_train = X_train.iloc[train_idx]
                    X_fold_val = X_train.iloc[val_idx]
                    y_fold_train = y_train.iloc[train_idx]
                    y_fold_val = y_train.iloc[val_idx]
                    
                    model.fit(X_fold_train, y_fold_train)
                    
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_fold_val)
                        score = roc_auc_score(y_fold_val, y_pred_proba, multi_class='ovr')
                    else:
                        y_pred = model.predict(X_fold_val)
                        score = accuracy_score(y_fold_val, y_pred)
                    
                    cv_scores.append(score)
                    
                    trial.report(np.mean(cv_scores), len(cv_scores))
                    
                    if trial.should_prune():
                        raise self.optuna.TrialPruned()
                
                return np.mean(cv_scores)
                
            except self.optuna.TrialPruned:
                raise
            except Exception as e:
                self.logger.warning(f"      ⚠️ Trial failed: {str(e)}")
                return 0.0
        
        # Run optimization
        try:
            study.optimize(
                objective, 
                n_trials=self.config.optuna_trials,
                timeout=self.config.optuna_timeout
            )
            
            best_params = study.best_params
            best_score = study.best_value
            
            self.logger.info(f"      ✅ Optimization completed!")
            self.logger.info(f"      🏆 Best score: {best_score:.4f}")
            self.logger.info(f"      📊 Trials: {len(study.trials)}")
            
            return {**model_config['params'], **best_params}, best_score
            
        except Exception as e:
            self.logger.error(f"      ❌ Optimization failed: {str(e)}")
            return model_config['params'], 0.0
    
    def _suggest_parameters(self, trial, param_space: Dict) -> Dict:
        """Suggest parameters based on parameter space."""
        params = {}
        
        for param_name, param_config in param_space.items():
            if isinstance(param_config, tuple) and len(param_config) == 2:
                low, high = param_config
                if isinstance(low, int) and isinstance(high, int):
                    params[param_name] = trial.suggest_int(param_name, low, high)
                else:
                    params[param_name] = trial.suggest_float(param_name, float(low), float(high))
            elif isinstance(param_config, list):
                params[param_name] = trial.suggest_categorical(param_name, param_config)
        
        return params


# =============================================================================
# MODEL TRAINER
# =============================================================================

class ModelTrainer:
    """
    Comprehensive model training with evaluation.
    
    Handles training all available models, hyperparameter optimization,
    cross-validation, and performance tracking.
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
        self.model_factory = ModelFactory(dependency_manager, logger)
        self.optimizer = HyperparameterOptimizer(config, dependency_manager, logger)
        self.training_results: Dict[str, Any] = {}
    
    @handle_errors
    def train_all_models(self, processed_splits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train all available models with hyperparameter optimization.
        
        Args:
            processed_splits: Preprocessed data splits
        
        Returns:
            Dictionary of training results for all models
        """
        self.logger.info("🚀 Starting comprehensive model training...")
        
        available_models = self.model_factory.get_available_model_names()
        self.logger.info(f"   📊 Available models: {len(available_models)}")
        
        training_start_time = datetime.now()
        
        for i, model_name in enumerate(available_models, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🤖 TRAINING MODEL {i}/{len(available_models)}: {model_name}")
            self.logger.info(f"{'='*60}")
            
            try:
                model_result = self._train_single_model(model_name, processed_splits)
                self.training_results[model_name] = model_result
                
                self.logger.info(f"✅ {model_name} training completed successfully")
                
            except Exception as e:
                self.logger.error(f"❌ {model_name} training failed: {str(e)}")
                self.training_results[model_name] = {
                    'model_name': model_name,
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            
            memory_cleanup()
        
        self._generate_training_summary()
        
        training_time = (datetime.now() - training_start_time).total_seconds()
        self.logger.info(f"\n✅ All model training completed in {training_time:.1f}s")
        
        return self.training_results
    
    def _train_single_model(
        self, 
        model_name: str, 
        processed_splits: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train a single model with comprehensive evaluation."""
        model_start_time = datetime.now()
        
        model_config = self.model_factory.get_model_config(model_name)
        
        X_train = processed_splits['X_train']
        y_train = processed_splits['y_train']
        X_val = processed_splits['X_val']
        y_val = processed_splits['y_val']
        cv_folds = processed_splits['cv_folds']
        
        # Step 1: Hyperparameter optimization
        self.logger.info("   🔍 Step 1: Hyperparameter optimization...")
        best_params, best_cv_score = self.optimizer.optimize_hyperparameters(
            model_config, X_train, y_train, cv_folds
        )
        
        # Step 2: Train final model
        self.logger.info("   🏋️ Step 2: Training final model...")
        final_model = model_config['class'](**best_params)
        final_model.fit(X_train, y_train)
        
        # Step 3: Evaluate model
        self.logger.info("   📊 Step 3: Evaluating model...")
        evaluation_results = self._evaluate_model(final_model, processed_splits)
        
        # Step 4: Cross-validation
        self.logger.info("   🔄 Step 4: Cross-validation...")
        cv_results = self._cross_validate_model(final_model, X_train, y_train, cv_folds)
        
        training_time = (datetime.now() - model_start_time).total_seconds()
        
        model_result = {
            'model_name': model_name,
            'model_type': model_config['type'],
            'success': True,
            'model': final_model,
            'best_params': best_params,
            'base_params': model_config['params'],
            'training_time': training_time,
            'cv_score': best_cv_score,
            'cv_results': cv_results,
            **evaluation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log performance summary
        self.logger.info(f"   📈 Performance Summary:")
        self.logger.info(f"      • CV Score: {best_cv_score:.4f}")
        self.logger.info(f"      • Val Accuracy: {evaluation_results['val_accuracy']:.4f}")
        self.logger.info(f"      • Val ROC-AUC: {evaluation_results['val_roc_auc']:.4f}")
        self.logger.info(f"      • Training Time: {training_time:.1f}s")
        
        return model_result
    
    def _evaluate_model(
        self, 
        model, 
        processed_splits: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate model on validation set."""
        X_val = processed_splits['X_val']
        y_val = processed_splits['y_val']
        
        y_pred = model.predict(X_val)
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val)
            val_roc_auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
        else:
            y_pred_proba = None
            val_roc_auc = 0.0
        
        val_accuracy = accuracy_score(y_val, y_pred)
        val_f1 = f1_score(y_val, y_pred, average='weighted')
        val_precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        val_recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        
        return {
            'val_accuracy': val_accuracy,
            'val_roc_auc': val_roc_auc,
            'val_f1': val_f1,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_predictions': y_pred.tolist(),
            'val_probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
        }
    
    def _cross_validate_model(
        self, 
        model, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        cv_folds: List
    ) -> Dict[str, Any]:
        """Perform cross-validation."""
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv_folds):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            fold_model = model.__class__(**model.get_params())
            fold_model.fit(X_fold_train, y_fold_train)
            
            if hasattr(fold_model, 'predict_proba'):
                y_pred_proba = fold_model.predict_proba(X_fold_val)
                score = roc_auc_score(y_fold_val, y_pred_proba, multi_class='ovr')
            else:
                y_pred = fold_model.predict(X_fold_val)
                score = accuracy_score(y_fold_val, y_pred)
            
            cv_scores.append(score)
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'cv_confidence_interval': (
                np.mean(cv_scores) - 1.96 * np.std(cv_scores) / np.sqrt(len(cv_scores)),
                np.mean(cv_scores) + 1.96 * np.std(cv_scores) / np.sqrt(len(cv_scores))
            )
        }
    
    def _generate_training_summary(self):
        """Generate comprehensive training summary."""
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 TRAINING SUMMARY")
        self.logger.info("="*60)
        
        successful_models = {
            name: result for name, result in self.training_results.items()
            if result.get('success', False)
        }
        
        if not successful_models:
            self.logger.error("❌ No models were successfully trained!")
            return
        
        sorted_models = sorted(
            successful_models.items(),
            key=lambda x: x[1]['val_roc_auc'],
            reverse=True
        )
        
        self.logger.info("\n🏆 Model Performance Ranking (by Validation ROC-AUC):")
        self.logger.info("-" * 70)
        
        for i, (name, result) in enumerate(sorted_models, 1):
            cv_score = result.get('cv_score', 0)
            val_acc = result['val_accuracy']
            val_auc = result['val_roc_auc']
            time_taken = result['training_time']
            
            self.logger.info(
                f"{i:2d}. {name:<15} | "
                f"Val AUC: {val_auc:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"CV: {cv_score:.4f} | "
                f"Time: {time_taken:.1f}s"
            )
        
        best_model_name, best_result = sorted_models[0]
        self.logger.info(f"\n🥇 BEST MODEL: {best_model_name}")
        self.logger.info(f"   • Validation ROC-AUC: {best_result['val_roc_auc']:.4f}")
        self.logger.info(f"   • Validation Accuracy: {best_result['val_accuracy']:.4f}")
        self.logger.info(f"   • Cross-validation Score: {best_result.get('cv_score', 0):.4f}")
        self.logger.info(f"   • Training Time: {best_result['training_time']:.1f}s")


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

class ModelPersistence:
    """
    Handles model saving and loading.
    
    Saves models, preprocessors, and metadata to output directories.
    """
    
    def __init__(self, config: ModelConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    @handle_errors
    def save_training_results(
        self, 
        training_results: Dict[str, Any],
        processed_splits: Dict[str, Any]
    ) -> None:
        """
        Save comprehensive training results.
        
        Args:
            training_results: Dictionary of model training results
            processed_splits: Processed data splits with feature engineer
        """
        self.logger.info("💾 Saving training results...")
        
        models_dir = Path(self.config.output_dir) / self.config.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        
        saved_models = {}
        for model_name, result in training_results.items():
            if result.get('success', False) and 'model' in result:
                try:
                    model_path = models_dir / f"{model_name}_model.joblib"
                    joblib.dump(result['model'], model_path)
                    saved_models[model_name] = str(model_path)
                    self.logger.info(f"   💾 Saved {model_name} model")
                except Exception as e:
                    self.logger.warning(f"   ⚠️ Failed to save {model_name}: {e}")
        
        # Save feature engineer
        if 'feature_engineer' in processed_splits:
            try:
                fe_path = models_dir / "feature_engineer.joblib"
                joblib.dump(processed_splits['feature_engineer'], fe_path)
                self.logger.info("   💾 Saved feature engineer")
            except Exception as e:
                self.logger.warning(f"   ⚠️ Failed to save feature engineer: {e}")
        
        # Save summary results
        self._save_summary_json(training_results)
        self._save_results_csv(training_results)
    
    def _save_summary_json(self, training_results: Dict[str, Any]):
        """Save summary as JSON."""
        summary_results = {}
        for model_name, result in training_results.items():
            if result.get('success', False):
                summary_result = result.copy()
                summary_result.pop('model', None)
                summary_result.pop('val_predictions', None)
                summary_result.pop('val_probabilities', None)
                summary_results[model_name] = summary_result
        
        try:
            summary_path = Path(self.config.output_dir) / self.config.results_dir / OUTPUT_FILES['training_summary']
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(summary_path, 'w') as f:
                json.dump(summary_results, f, indent=2, default=str)
            self.logger.info(f"   💾 Training summary saved: {summary_path}")
        except Exception as e:
            self.logger.warning(f"   ⚠️ Could not save training summary: {e}")
    
    def _save_results_csv(self, training_results: Dict[str, Any]):
        """Save results as CSV."""
        try:
            results_data = []
            for model_name, result in training_results.items():
                if result.get('success', False):
                    results_data.append({
                        'Model': model_name,
                        'Model_Type': result.get('model_type', ''),
                        'Val_Accuracy': result.get('val_accuracy', 0),
                        'Val_ROC_AUC': result.get('val_roc_auc', 0),
                        'Val_F1': result.get('val_f1', 0),
                        'Val_Precision': result.get('val_precision', 0),
                        'Val_Recall': result.get('val_recall', 0),
                        'CV_Score': result.get('cv_score', 0),
                        'CV_Mean': result.get('cv_results', {}).get('cv_mean', 0),
                        'CV_Std': result.get('cv_results', {}).get('cv_std', 0),
                        'Training_Time': result.get('training_time', 0),
                        'Timestamp': result.get('timestamp', '')
                    })
            
            if results_data:
                results_df = pd.DataFrame(results_data)
                csv_path = Path(self.config.output_dir) / self.config.results_dir / OUTPUT_FILES['model_comparison']
                results_df.to_csv(csv_path, index=False)
                self.logger.info(f"   💾 Results CSV saved: {csv_path}")
        
        except Exception as e:
            self.logger.warning(f"   ⚠️ Could not save results CSV: {e}")
    
    def save_final_model(
        self, 
        model, 
        model_name: str,
        feature_engineer,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Save final deployment-ready model.
        
        Args:
            model: Trained model
            model_name: Name of the model
            feature_engineer: Fitted feature engineer
            metadata: Model metadata dictionary
        """
        final_model_dir = Path(self.config.output_dir) / self.config.final_model_dir
        final_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = final_model_dir / f"{model_name}_final.joblib"
        joblib.dump(model, model_path)
        self.logger.info(f"   💾 Final model saved: {model_path}")
        
        # Save preprocessor
        preprocessor_path = final_model_dir / "preprocessor_final.joblib"
        joblib.dump(feature_engineer, preprocessor_path)
        self.logger.info(f"   💾 Preprocessor saved: {preprocessor_path}")
        
        # Save metadata
        metadata_path = final_model_dir / OUTPUT_FILES['model_metadata']
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        self.logger.info(f"   💾 Metadata saved: {metadata_path}")


# =============================================================================
# TRAINING VISUALIZER
# =============================================================================

class TrainingVisualizer:
    """
    Creates visualizations for training results.
    
    Generates comparison plots, performance charts, and model dashboards.
    """
    
    def __init__(self, config: ModelConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    @handle_errors
    def create_training_visualizations(self, training_results: Dict[str, Any]) -> None:
        """
        Create comprehensive training visualizations.
        
        Args:
            training_results: Dictionary of model training results
        """
        self.logger.info("📊 Creating training visualizations...")
        
        successful_models = {
            name: result for name, result in training_results.items()
            if result.get('success', False)
        }
        
        if not successful_models:
            self.logger.warning("   ⚠️ No successful models to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Training Results', fontsize=16, fontweight='bold')
        
        self._plot_performance_comparison(successful_models, axes[0, 0])
        self._plot_training_time(successful_models, axes[0, 1])
        self._plot_cv_results(successful_models, axes[1, 0])
        self._plot_model_type_performance(successful_models, axes[1, 1])
        
        plt.tight_layout()
        
        plot_path = Path(self.config.output_dir) / self.config.plots_dir / PLOT_FILES['training_results']
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"   💾 Training visualizations saved: {plot_path}")
        
        plt.show()
        plt.close()
    
    def _plot_performance_comparison(self, models: Dict, ax):
        """Plot model performance comparison."""
        model_names = list(models.keys())
        accuracies = [models[name]['val_accuracy'] for name in model_names]
        roc_aucs = [models[name]['val_roc_auc'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        ax.bar(x + width/2, roc_aucs, width, label='ROC-AUC', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([name[:10] for name in model_names], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_training_time(self, models: Dict, ax):
        """Plot training time comparison."""
        model_names = list(models.keys())
        times = [models[name]['training_time'] for name in model_names]
        
        bars = ax.bar(model_names, times, alpha=0.7, color='skyblue')
        ax.set_xlabel('Models')
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Training Time Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        for bar, time_val in zip(bars, times):
            ax.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + max(times)*0.01,
                f'{time_val:.1f}s', 
                ha='center', 
                va='bottom', 
                fontsize=8
            )
    
    def _plot_cv_results(self, models: Dict, ax):
        """Plot cross-validation results."""
        model_names = []
        cv_means = []
        cv_stds = []
        
        for name, result in models.items():
            cv_results = result.get('cv_results', {})
            if cv_results:
                model_names.append(name)
                cv_means.append(cv_results.get('cv_mean', 0))
                cv_stds.append(cv_results.get('cv_std', 0))
        
        if model_names:
            x = np.arange(len(model_names))
            ax.bar(x, cv_means, yerr=cv_stds, capsize=5, alpha=0.7, color='lightgreen')
            ax.set_xlabel('Models')
            ax.set_ylabel('CV Score')
            ax.set_title('Cross-Validation Results')
            ax.set_xticks(x)
            ax.set_xticklabels([name[:10] for name in model_names], rotation=45)
            ax.grid(True, alpha=0.3)
    
    def _plot_model_type_performance(self, models: Dict, ax):
        """Plot performance by model type."""
        type_performance = {}
        for name, result in models.items():
            model_type = result.get('model_type', 'unknown')
            if model_type not in type_performance:
                type_performance[model_type] = []
            type_performance[model_type].append(result['val_roc_auc'])
        
        if len(type_performance) > 1:
            types = list(type_performance.keys())
            avg_scores = [np.mean(type_performance[t]) for t in types]
            
            ax.bar(types, avg_scores, alpha=0.7, color='coral')
            ax.set_xlabel('Model Type')
            ax.set_ylabel('Average ROC-AUC')
            ax.set_title('Performance by Model Type')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5, 0.5, 
                'Insufficient model types\nfor comparison',
                ha='center', 
                va='center', 
                transform=ax.transAxes
            )
            ax.set_title('Performance by Model Type')
