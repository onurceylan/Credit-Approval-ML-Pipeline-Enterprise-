"""
Model Predictor
===============

API-ready model serving class.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

from ..core.config import PipelineConfig, get_config
from ..core.exceptions import InferenceError


class ModelPredictor:
    """
    Production-ready model predictor.
    
    Designed for API integration and batch processing.
    """
    
    def __init__(
        self,
        model_path: str,
        feature_engineer_path: str,
        config: Optional[PipelineConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config or get_config()
        self.logger = logger or logging.getLogger(__name__)
        
        self._load(model_path, feature_engineer_path)
        self._request_count = 0
    
    def _load(self, model_path: str, feature_engineer_path: str):
        """Load model and feature engineer."""
        if not Path(model_path).exists():
            raise InferenceError(f"Model not found: {model_path}")
        if not Path(feature_engineer_path).exists():
            raise InferenceError(f"Feature engineer not found: {feature_engineer_path}")
        
        self.model = joblib.load(model_path)
        self.feature_engineer = joblib.load(feature_engineer_path)
        self.logger.info("✅ Model and feature engineer loaded")
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a single prediction.
        
        Args:
            features: Dictionary of feature values
        
        Returns:
            Prediction result
        """
        self._request_count += 1
        
        df = pd.DataFrame([features])
        X = self.feature_engineer.transform(df)
        
        prediction = int(self.model.predict(X)[0])
        
        result = {
            'prediction': prediction,
            'label': 'Bad Credit' if prediction == 1 else 'Good Credit',
            'timestamp': datetime.now().isoformat(),
            'request_id': self._request_count
        }
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            result['confidence'] = float(max(proba))
            result['probabilities'] = {
                'good_credit': float(proba[0]),
                'bad_credit': float(proba[1]) if len(proba) > 1 else 0
            }
        
        return result
    
    def predict_batch(
        self,
        data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Make batch predictions."""
        return [self.predict(row) for row in data]
    
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            'status': 'healthy',
            'model_loaded': self.model is not None,
            'feature_engineer_loaded': self.feature_engineer is not None,
            'total_requests': self._request_count,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_feature_names(self) -> List[str]:
        """Get expected feature names."""
        return self.feature_engineer.get_feature_names()
