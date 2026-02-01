"""
Inference Pipeline
==================

Production inference pipeline for predictions.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import pandas as pd
import numpy as np
import joblib

from .base import BasePipeline
from ..core.config import PipelineConfig, get_config
from ..core.logger import get_logger
from ..core.exceptions import InferenceError
from ..features.engineer import FeatureEngineer
from ..models.registry import ModelRegistry


class InferencePipeline(BasePipeline):
    """
    Production inference pipeline.
    
    Features:
    - Batch and single predictions
    - Probability outputs
    - Error handling
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_engineer_path: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config or get_config()
        self.logger = logger or get_logger()
        
        self.model = None
        self.feature_engineer = None
        
        if model_path:
            self.load_model(model_path)
        
        if feature_engineer_path:
            self.load_feature_engineer(feature_engineer_path)
    
    def validate(self) -> bool:
        """Validate pipeline is ready for inference."""
        if self.model is None:
            raise InferenceError("Model not loaded")
        if self.feature_engineer is None:
            raise InferenceError("Feature engineer not loaded")
        return True
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        path = Path(model_path)
        if not path.exists():
            raise InferenceError(f"Model not found: {model_path}")
        
        self.model = joblib.load(path)
        self.logger.info(f"📦 Model loaded: {path.name}")
    
    def load_feature_engineer(self, path: str):
        """Load a fitted feature engineer."""
        fe_path = Path(path)
        if not fe_path.exists():
            raise InferenceError(f"Feature engineer not found: {path}")
        
        self.feature_engineer = joblib.load(fe_path)
        self.logger.info(f"📦 Feature engineer loaded")
    
    def load_from_registry(self, model_id: Optional[str] = None):
        """Load model from registry."""
        registry = ModelRegistry(self.config, self.logger)
        
        if model_id is None:
            model_id = registry.get_best_model()
            if model_id is None:
                raise InferenceError("No models in registry")
        
        self.model = registry.load_model(model_id)
        self.logger.info(f"📦 Model loaded from registry: {model_id}")
    
    def run(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Run inference on input data.
        
        Args:
            data: Input features
            return_probabilities: Include class probabilities
        
        Returns:
            Predictions dictionary
        """
        self.validate()
        
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Transform features
        X = self.feature_engineer.transform(data)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        result = {
            'predictions': predictions.tolist(),
            'n_samples': len(predictions)
        }
        
        if return_probabilities and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            result['probabilities'] = probabilities.tolist()
        
        return result
    
    def predict_single(
        self,
        features: Dict[str, Any],
        return_probability: bool = True
    ) -> Dict[str, Any]:
        """Make prediction for a single sample."""
        result = self.run(features, return_probabilities=return_probability)
        
        output = {
            'prediction': result['predictions'][0],
            'label': 'Bad Credit' if result['predictions'][0] == 1 else 'Good Credit'
        }
        
        if 'probabilities' in result:
            output['confidence'] = max(result['probabilities'][0])
            output['probabilities'] = result['probabilities'][0]
        
        return output
    
    def predict_batch(
        self,
        data: pd.DataFrame,
        return_probabilities: bool = False
    ) -> pd.DataFrame:
        """Make predictions for a batch of samples."""
        result = self.run(data, return_probabilities=return_probabilities)
        
        output_df = data.copy()
        output_df['prediction'] = result['predictions']
        output_df['label'] = output_df['prediction'].map({0: 'Good Credit', 1: 'Bad Credit'})
        
        if 'probabilities' in result:
            probs = np.array(result['probabilities'])
            output_df['confidence'] = probs.max(axis=1)
        
        return output_df
