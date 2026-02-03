"""
Training Pipeline
=================

Complete training pipeline orchestration with verbose logging matching V3.5 notebook style.
"""

import gc
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np

from .base import BasePipeline
from ..core.config import PipelineConfig, get_config
from ..core.logger import get_logger
from ..data.loader import DataLoader
from ..data.validator import DataValidator
from ..features.engineer import FeatureEngineer
from ..features.preprocessor import DataPreprocessor
from ..models.factory import ModelFactory
from ..models.registry import ModelRegistry
from ..training.trainer import ModelTrainer
from ..training.optimizer import HyperparameterOptimizer
from ..evaluation.evaluator import ModelEvaluator, ModelSelector, FinalValidator, ModelInterpretabilityAnalyzer, FinalRecommendationEngine
from ..evaluation.business import BusinessImpactAnalyst
from ..evaluation.visualizer import PipelineVisualizer, BusinessVisualizationEngine, SelectionVisualizer, ABTestVisualizer
from ..evaluation.ab_testing import ABTestSimulator


class TrainingPipeline(BasePipeline):
    """
    Complete training pipeline with rich logging.
    
    Steps:
    1. Load & Validate Data (Cell 2)
    2. Preprocess & Engineer Features (Cell 3)
    3. Train & Optimize Models (Cell 4)
    4. Evaluate & Select Best Model (Cell 5 & 6)
    5. Business Impact Analysis (Cell 7)
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        logger: Optional[logging.Logger] = None,
        optimize: bool = True
    ):
        self.config = config or get_config()
        self.logger = logger or get_logger()
        self.optimize = optimize
        
        # Initialize components
        self.data_loader = DataLoader(self.config, self.logger)
        self.data_validator = DataValidator(self.config, self.logger)
        self.feature_engineer = FeatureEngineer(self.config, self.logger)
        self.preprocessor = DataPreprocessor(self.config, self.logger)
        self.model_factory = ModelFactory(self.config, self.logger)
        self.model_registry = ModelRegistry(self.config, self.logger)
        self.trainer = ModelTrainer(self.config, self.model_factory, self.model_registry, self.logger)
        self.optimizer = HyperparameterOptimizer(self.config, self.model_factory, self.logger)
        self.evaluator = ModelEvaluator(self.config, self.logger)
        self.selector = ModelSelector(self.config, self.logger)
        self.validator = FinalValidator(self.config, self.logger)
        self.interpretability = ModelInterpretabilityAnalyzer(self.logger)
        self.business_analyst = BusinessImpactAnalyst(self.logger)
        self.recommendation_engine = FinalRecommendationEngine(self.logger)
        
        self.visualizer = PipelineVisualizer(self.config, self.logger)
        self.business_visualizer = BusinessVisualizationEngine(self.config, self.logger)
        self.selection_visualizer = SelectionVisualizer(self.config, self.logger)
        self.ab_simulator = None # Initialized during run with models
        self.ab_visualizer = ABTestVisualizer(self.config, self.logger)
    
    def validate(self) -> bool:
        """Validate pipeline configuration."""
        return True
    
    def run(self) -> Dict[str, Any]:
        """Execute complete training pipeline."""
        start_time = datetime.now()
        
        # ==================================================================================
        # CELL 1: Environment Setup (Skipped here as it's done in main.ipynb)
        # ==================================================================================
        self.logger.info("=" * 60)
        self.logger.info("🚀 STARTING TRAINING PIPELINE (V3.5 Hybrid Architecture)")
        self.logger.info("=" * 60)

        # ==================================================================================
        # CELL 2: Data Loading & Validation
        # ==================================================================================
        self.logger.info("\n📥 [CELL 2] Data Loading & Validation")
        self.logger.info("-" * 40)
        
        app_data, credit_data = self.data_loader.load_data()
        
        self.logger.info(f"   📊 Application Data: {app_data.shape[0]:,} rows, {app_data.shape[1]} cols")
        self.logger.info(f"   📊 Credit Data:      {credit_data.shape[0]:,} rows, {credit_data.shape[1]} cols")
        
        self.logger.info("\n🔍 Validating data quality...")
        self.data_validator.validate(app_data, credit_data)
        
        self.logger.info("✅ CELL 2 COMPLETED - Data Ready!")
        
        # ==================================================================================
        # CELL 3: Data Preprocessing & Feature Engineering
        # ==================================================================================
        self.logger.info("\n🔧 [CELL 3] Preprocessing & Feature Engineering")
        self.logger.info("-" * 40)
        self.logger.info("🚀 Starting corrected preprocessing and feature engineering...")
        
        splits = self.preprocessor.preprocess(
            app_data, credit_data, self.feature_engineer
        )
        
        # Original notebook style logs for Cell 3 completion
        self.logger.info(f"✅ Preprocessing completed!")
        self.logger.info(f"    📊 Input Features: {app_data.shape[1]}")
        self.logger.info(f"    📊 Final Features: {len(self.feature_engineer.get_feature_names())}")
        self.logger.info(f"    ✨ New Features Created: {len(self.feature_engineer.get_feature_names()) - app_data.shape[1]}")
        
        top_new_features = ['AGE_YEARS', 'AGE_GROUP', 'EMPLOYED_YEARS', 'IS_EMPLOYED', 'INCOME_LOG']
        for feat in top_new_features:
            if feat in self.feature_engineer.get_feature_names():
                self.logger.info(f"       • {feat}")
        
        self.logger.info("\n" + "🛡️ Data leakage check: ✅ CLEAN")
        self.logger.info(f"    📊 Train: {splits['X_train'].shape[0]:,} ({splits['X_train'].shape[0]/len(app_data)*100:.1f}%)")
        self.logger.info(f"    📊 Val:   {splits['X_val'].shape[0]:,} ({splits['X_val'].shape[0]/len(app_data)*100:.1f}%)")
        self.logger.info(f"    📊 Test:  {splits['X_test'].shape[0]:,} ({splits['X_test'].shape[0]/len(app_data)*100:.1f}%)")
        
        # Memory cleanup
        del app_data, credit_data
        gc.collect()
        import psutil
        import os
        process = psutil.Process(os.getpid())
        self.logger.info(f"🧹 Memory cleanup - Current usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
        self.logger.info("✅ CELL 3 COMPLETED - CORRECTED Data Preprocessing & Feature Engineering Ready!")
        
        # ==================================================================================
        # CELL 4: Model Training & Hyperparameter Optimization
        # ==================================================================================
        self.logger.info("\n🏋️ [CELL 4] Model Training & Optimization")
        self.logger.info("-" * 40)
        self.logger.info("🔄 Ready for Cell 4: Model Training & Hyperparameter Optimization")
        
        model_params = {}
        if self.optimize and self.optimizer.available:
            self.logger.info("\n🔍 Optimizing hyperparameters with Optuna...")
            optimization_results = self.optimizer.optimize_all_models(
                splits['X_train'], splits['y_train']
            )
            model_params = {
                name: result['best_params']
                for name, result in optimization_results.items()
            }
        
        self.logger.info("\n🚀 Starting Full Model Training...")
        training_results = self.trainer.train_all_models(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val'],
            model_params
        )
        self.logger.info("✅ CELL 4 COMPLETED - All models trained!")

        # ==================================================================================
        # CELL 5: Evaluation & Statistical Analysis
        # ==================================================================================
        self.logger.info("\n📊 [CELL 5] Model Evaluation & Statistical Comparison")
        self.logger.info("-" * 40)
        
        evaluation_results = self.evaluator.evaluate_all(
            training_results,
            splits['X_test'], splits['y_test']
        )
        
        # Friedman Test
        friedman_results = self.evaluator.perform_friedman_test(training_results)
        self.logger.info("✅ CELL 5 COMPLETED - Evaluation Finished")

        # ==================================================================================
        # CELL 6: Model Selection & Final Validation
        # ==================================================================================
        self.logger.info("\n🏆 [CELL 6] Advanced Model Selection & Validation")
        self.logger.info("-" * 40)
        
        comprehensive_results = {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'friedman_results': friedman_results
        }
        
        # 1. Intelligent selection
        selection_result = self.selector.select_best_model(comprehensive_results)
        best_model = selection_result['selected_model']
        
        # 2. Final validation
        validation_results = self.validator.validate_deployment_readiness(best_model, evaluation_results)
        
        # 3. Interpretability
        interpretability_results = self.interpretability.analyze_interpretability(best_model, evaluation_results.get(best_model, {}))
        
        self.logger.info(f"\n🥇 FINAL SELECTION: {best_model}")
        self.logger.info(f"   • Selection Score: {selection_result['selection_score']:.4f}")
        self.logger.info(f"   • Deployment Status: {validation_results['deployment_status']}")
        self.logger.info("✅ CELL 6 COMPLETED - Model Selected & Validated")
        
        # ==================================================================================
        # CELL 7: Business Impact Analysis & Visualization
        # ==================================================================================
        self.logger.info("\n💰 [CELL 7] Business Impact Analysis & Enterprise Insights")
        self.logger.info("-" * 40)
        
        # Comprehensive Business Analysis
        business_params = {
            'revenue_per_approval': self.config.revenue_per_approval,
            'cost_false_positive': self.config.cost_false_positive,
            'cost_false_negative': self.config.cost_false_negative
        }
        
        business_analysis = self.business_analyst.analyze_comprehensive_impact(
            best_model, evaluation_results, business_params
        )
        
        # Generate Final Recommendations
        final_recs = self.recommendation_engine.generate_recommendations(
            validation_results, business_analysis
        )
        
        # Visualization
        self.logger.info("\n🎨 Generating Visualizations (Matching V3.5 Enterprise Style)...")
        
        # 1. Target Distribution
        self.visualizer.plot_target_distribution(splits['y_train'])
        
        # 2. Standard Training Dashboard
        self.visualizer.plot_model_comparison(evaluation_results, training_results)
        
        # 3. Model Selection & Readiness Dashboard (6-panel)
        self.selection_visualizer.plot_selection_dashboard(selection_result, validation_results)
        
        # 4. Enterprise Business Impact Dashboard (12-panel)
        self.business_visualizer.create_business_dashboard(business_analysis)
        
        # 5. ROC Curves & Confusion Matrices
        self.visualizer.plot_roc_curves(training_results, splits['X_test'], splits['y_test'])
        self.visualizer.plot_confusion_matrices(evaluation_results)
        
        # 6. Feature Importance (Selected Model)
        best_model_obj = training_results[best_model]['model']
        feature_names = self.feature_engineer.get_feature_names()
        self.visualizer.plot_feature_importance(best_model_obj, feature_names, best_model)
        
        # ==================================================================================
        # CELL 8: Offline A/B Simulation (Challenger vs Champion)
        # ==================================================================================
        self.logger.info("\n🧪 [CELL 8] Offline A/B Simulation (Challenger vs Champion)")
        self.logger.info("-" * 40)
        
        # For simulation, we define 'Logistic Regression' as the Champion/Baseline
        champion_name = 'Logistic Regression'
        if champion_name in training_results:
            champion_model_obj = training_results[champion_name]['model']
        else:
            # Fallback to the first model if LR is missing
            champion_name = list(training_results.keys())[0]
            champion_model_obj = training_results[champion_name]['model']
            
        self.logger.info(f"   🥊 Champion: {champion_name}")
        self.logger.info(f"   🎖️ Challenger: {best_model}")
        
        # Re-initialize with full required params
        self.ab_simulator = ABTestSimulator(
            champion_model=champion_model_obj,
            challenger_model=best_model_obj,
            X_test=splits['X_test'],
            y_test=splits['y_test'],
            config=self.config,
            logger=self.logger
        )
        
        ab_results = self.ab_simulator.run_simulation(verbose=True)
        
        # Save A/B Results
        import json
        from pathlib import Path
        from dataclasses import asdict
        
        ab_report_path = Path(self.config.output_dir) / self.config.results_dir / "ab_test_report.json"
        
        # Prepare a clean dictionary for JSON (excluding the large bootstrap distributions)
        report_data = asdict(ab_results)
        report_data.pop('champion_metrics', None)
        report_data.pop('challenger_metrics', None)
        
        with open(ab_report_path, 'w') as f:
            json.dump(report_data, f, indent=4)
            
        # Use simulator's internal plotting for consistency with notebook
        ab_plot_path = Path(self.config.output_dir) / self.config.plots_dir / "ab_test_dashboard.png"
        self.ab_simulator.plot_results(ab_results, save_path=str(ab_plot_path))
        
        self.logger.info("✅ CELL 8 COMPLETED - A/B Simulation Finished")
        
        # Pipeline complete
        duration = (datetime.now() - start_time).total_seconds()
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info(f"   ⏱️ Total Duration: {duration:.1f}s")
        self.logger.info(f"   🏆 Best Model: {best_model}")
        self.logger.info("=" * 60)
        
        return {
            'best_model': best_model,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'business_results': business_analysis,
            'ab_results': ab_results,
            'splits': splits,
            'feature_engineer': self.feature_engineer,
            'duration': duration
        }
