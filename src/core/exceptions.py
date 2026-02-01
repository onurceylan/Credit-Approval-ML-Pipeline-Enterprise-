"""
Custom Exceptions
=================

Hierarchical exception classes for the ML pipeline.
"""


class MLPipelineError(Exception):
    """Base exception for all ML pipeline errors."""
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(MLPipelineError):
    """Raised when configuration is invalid or missing."""
    pass


class DataValidationError(MLPipelineError):
    """Raised when data validation fails."""
    pass


class FeatureEngineeringError(MLPipelineError):
    """Raised when feature engineering fails."""
    pass


class ModelTrainingError(MLPipelineError):
    """Raised when model training fails."""
    pass


class ModelEvaluationError(MLPipelineError):
    """Raised when model evaluation fails."""
    pass


class DeploymentError(MLPipelineError):
    """Raised when deployment preparation fails."""
    pass


class PipelineError(MLPipelineError):
    """Raised when pipeline execution fails."""
    pass


class InferenceError(MLPipelineError):
    """Raised when inference/prediction fails."""
    pass
