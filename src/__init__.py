"""
Credit Approval ML Pipeline - Source Package
=============================================

MLOps-ready machine learning pipeline for credit approval prediction.

Modules:
    core: Configuration, logging, exceptions
    data: Data loading and validation
    features: Feature engineering and preprocessing
    models: Model factory and registry
    training: Model training and optimization
    evaluation: Model evaluation and metrics
    pipelines: Training and inference pipelines
    serving: Production model serving
"""

__version__ = "3.0.0"
__author__ = "Onur"

# Core
from .core.config import PipelineConfig, get_config
from .core.logger import setup_logger, get_logger
from .core.exceptions import (
    MLPipelineError,
    DataValidationError,
    ModelTrainingError,
    FeatureEngineeringError
)

# Data
from .data.loader import DataLoader
from .data.validator import DataValidator

# Features
from .features.engineer import FeatureEngineer
from .features.preprocessor import DataPreprocessor, TargetCreator, DataSplitter

# Models
from .models.factory import ModelFactory
from .models.registry import ModelRegistry

# Training
from .training.trainer import ModelTrainer
from .training.optimizer import HyperparameterOptimizer

# Evaluation
from .evaluation.evaluator import ModelEvaluator
from .evaluation.metrics import MetricsCalculator, BusinessAnalyzer

# Pipelines
from .pipelines.training_pipeline import TrainingPipeline
from .pipelines.inference_pipeline import InferencePipeline

# Serving
from .serving.predictor import ModelPredictor

__all__ = [
    # Version
    "__version__",
    # Core
    "PipelineConfig",
    "get_config",
    "setup_logger",
    "get_logger",
    "MLPipelineError",
    "DataValidationError",
    "ModelTrainingError",
    "FeatureEngineeringError",
    # Data
    "DataLoader",
    "DataValidator",
    # Features
    "FeatureEngineer",
    "DataPreprocessor",
    "TargetCreator",
    "DataSplitter",
    # Models
    "ModelFactory",
    "ModelRegistry",
    # Training
    "ModelTrainer",
    "HyperparameterOptimizer",
    # Evaluation
    "ModelEvaluator",
    "MetricsCalculator",
    "BusinessAnalyzer",
    # Pipelines
    "TrainingPipeline",
    "InferencePipeline",
    # Serving
    "ModelPredictor",
]
