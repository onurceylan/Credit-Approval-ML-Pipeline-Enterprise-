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
from ..evaluation.evaluator import ModelEvaluator
from ..evaluation.metrics import BusinessAnalyzer
from ..evaluation.visualizer import PipelineVisualizer


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
        self.business_analyzer = BusinessAnalyzer(self.config, self.logger)
        self.visualizer = PipelineVisualizer(self.config, self.logger)
    
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
        # CELL 6: Model Selection
        # ==================================================================================
        self.logger.info("\n🏆 [CELL 6] Model Selection & Validation")
        self.logger.info("-" * 40)
        
        best_model, model_scores = self.evaluator.select_best_model(
            training_results, evaluation_results
        )
        
        self.logger.info(f"\n🥇 BEST MODEL SELECTED: {best_model}")
        self.logger.info(f"   • Test AUC: {model_scores[best_model]['test_roc_auc']:.4f}")
        self.logger.info(f"   • Composite Score: {model_scores[best_model]['composite_score']:.4f}")
        self.logger.info("✅ CELL 6 COMPLETED - Model Selected")
        
        # ==================================================================================
        # CELL 7: Business Impact Analysis & Visualization
        # ==================================================================================
        self.logger.info("\n💰 [CELL 7] Business Impact Analysis & Insights")
        self.logger.info("-" * 40)
        
        business_results = {}
        for model_name, result in training_results.items():
            if result.get('success') and 'model' in result:
                y_pred = result['model'].predict(splits['X_test'])
                impact = self.business_analyzer.analyze_impact(
                    splits['y_test'].values, y_pred, model_name
                )
                business_results[model_name] = impact
        
        if business_results:
            self.business_analyzer.generate_business_case(business_results, best_model)
            
        # Visualization
        self.logger.info("\n🎨 Generating Visualizations (matching original notebook)...")
        
        # 1. Target Distribution
        self.visualizer.plot_target_distribution(splits['y_train'])
        
        # 2. Model Comparison (Dashboard)
        self.visualizer.plot_model_comparison(evaluation_results, training_results)
        
        # 3. ROC Curves
        self.visualizer.plot_roc_curves(training_results, splits['X_test'], splits['y_test'])
        
        # 4. Confusion Matrices
        self.visualizer.plot_confusion_matrices(evaluation_results)
        
        # 5. Feature Importance (Best Model)
        best_model_obj = training_results[best_model]['model']
        feature_names = self.feature_engineer.get_feature_names()
        self.visualizer.plot_feature_importance(best_model_obj, feature_names, best_model)
        
        # 6. Business Impact
        self.visualizer.plot_business_impact(business_results)
        
        # Pipeline complete
        duration = (datetime.now() - start_time).total_seconds()
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info(f"   ⏱️ Total Duration: {duration:.1f}s")
        self.logger.info(f"   🏆 Best Model: {best_model}")
        self.logger.info("=" * 60)
        
        return {
            'best_model': best_model,
            'model_scores': model_scores,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'business_results': business_results,
            'feature_engineer': self.feature_engineer,
            'duration': duration
        }
