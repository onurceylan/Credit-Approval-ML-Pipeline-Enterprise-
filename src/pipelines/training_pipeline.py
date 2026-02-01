"""
Training Pipeline
=================

Complete training pipeline orchestration.
"""

import gc
import logging
from datetime import datetime
from typing import Dict, Any, Optional

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
from ..evaluation.evaluator import ModelEvaluator
from ..evaluation.metrics import BusinessAnalyzer
from ..evaluation.visualizer import PipelineVisualizer


class TrainingPipeline(BasePipeline):
    """
    Complete training pipeline.
    
    Steps:
    1. Load data
    2. Validate data
    3. Preprocess and engineer features
    4. Optimize hyperparameters (optional)
    5. Train models
    6. Evaluate models
    7. Select best model
    8. Generate reports
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
    
    def validate(self) -> bool:
        """Validate pipeline configuration."""
        return True
    
    def run(self) -> Dict[str, Any]:
        """Execute complete training pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("🚀 STARTING TRAINING PIPELINE")
        self.logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Step 1: Load data
        self.logger.info("\n📥 Step 1: Loading data...")
        app_data, credit_data = self.data_loader.load_data()
        
        # Step 2: Validate data
        self.logger.info("\n🔍 Step 2: Validating data...")
        self.data_validator.validate(app_data, credit_data)
        
        # Step 3: Preprocess
        self.logger.info("\n🔧 Step 3: Preprocessing data...")
        splits = self.preprocessor.preprocess(
            app_data, credit_data, self.feature_engineer
        )
        
        # Clean up
        del app_data, credit_data
        gc.collect()
        
        # Step 4: Optimize (optional)
        model_params = {}
        if self.optimize and self.optimizer.available:
            self.logger.info("\n🔍 Step 4: Optimizing hyperparameters...")
            optimization_results = self.optimizer.optimize_all_models(
                splits['X_train'], splits['y_train']
            )
            model_params = {
                name: result['best_params']
                for name, result in optimization_results.items()
            }
        
        # Step 5: Train models
        self.logger.info("\n🏋️ Step 5: Training models...")
        training_results = self.trainer.train_all_models(
            splits['X_train'], splits['y_train'],
            splits['X_val'], splits['y_val'],
            model_params
        )
        
        # Step 6: Evaluate
        self.logger.info("\n📊 Step 6: Evaluating models...")
        evaluation_results = self.evaluator.evaluate_all(
            training_results,
            splits['X_test'], splits['y_test']
        )
        
        # Step 7: Select best model
        self.logger.info("\n🏆 Step 7: Selecting best model...")
        
        # Perform Friedman Statistical Test (from original project)
        friedman_results = self.evaluator.perform_friedman_test(training_results)
        
        best_model, model_scores = self.evaluator.select_best_model(
            training_results, evaluation_results
        )
        
        # Step 8: Business analysis
        self.logger.info("\n💰 Step 8: Business impact analysis...")
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
            
        # Step 9: Visualize Everything
        self.logger.info("\n🎨 Step 9: Generating Visualizations...")
        self.visualizer = PipelineVisualizer(self.config, self.logger)
        
        # 1. Target Distribution
        self.visualizer.plot_target_distribution(splits['y_train'])
        
        # 2. Model Comparison
        self.visualizer.plot_model_comparison(evaluation_results)
        
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
        self.logger.info("✅ TRAINING PIPELINE COMPLETE")
        self.logger.info(f"   Duration: {duration:.1f}s")
        self.logger.info(f"   Best Model: {best_model}")
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
