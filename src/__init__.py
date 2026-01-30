"""
Credit Approval ML Pipeline - Source Package
=============================================

Enterprise-grade machine learning pipeline for credit approval prediction.

Modules:
    - config: Configuration classes and constants
    - utils: Utility functions and logging
    - data_loader: Data loading and validation
    - features: Feature engineering pipeline
    - models: Model factory and configurations
    - train: Training and optimization
    - evaluate: Evaluation and business analysis
"""

from .config import ModelConfig, MLPipelineError, DataValidationError, ModelTrainingError, DeploymentError
from .utils import MLPipelineLogger, DependencyManager, handle_errors, setup_output_directories, memory_cleanup
from .data_loader import RobustDataLoader, TemporalDataSplitter
from .features import SafeDataSplitter, RobustFeatureEngineer, DataPreprocessingPipeline, DataQualityAnalyzer
from .models import ModelFactory
from .train import HyperparameterOptimizer, ModelTrainer, ModelPersistence, TrainingVisualizer
from .evaluate import (
    StatisticalValidator, 
    ModelEvaluator, 
    BusinessImpactAnalyst, 
    ModelSelector,
    EvaluationVisualizer
)

__version__ = "3.0.0"
__author__ = "Credit Approval Team"

__all__ = [
    # Config
    "ModelConfig",
    "MLPipelineError",
    "DataValidationError", 
    "ModelTrainingError",
    "DeploymentError",
    # Utils
    "MLPipelineLogger",
    "DependencyManager",
    "handle_errors",
    "setup_output_directories",
    "memory_cleanup",
    # Data
    "RobustDataLoader",
    "TemporalDataSplitter",
    # Features
    "SafeDataSplitter",
    "RobustFeatureEngineer",
    "DataPreprocessingPipeline",
    "DataQualityAnalyzer",
    # Models
    "ModelFactory",
    # Train
    "HyperparameterOptimizer",
    "ModelTrainer",
    "ModelPersistence",
    "TrainingVisualizer",
    # Evaluate
    "StatisticalValidator",
    "ModelEvaluator",
    "BusinessImpactAnalyst",
    "ModelSelector",
    "EvaluationVisualizer",
]
