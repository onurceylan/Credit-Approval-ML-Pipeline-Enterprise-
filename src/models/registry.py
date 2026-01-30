"""
Model Registry
==============

Manages trained models and their metadata.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import joblib

from ..core.config import PipelineConfig


class ModelRegistry:
    """
    Registry for trained models.
    
    Provides:
    - Model serialization
    - Version tracking
    - Metadata management
    """
    
    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.registry_path = Path(config.output_dir) / config.models_dir
        self.registry_file = self.registry_path / "registry.json"
        self._ensure_registry()
    
    def _ensure_registry(self):
        """Ensure registry directory and file exist."""
        self.registry_path.mkdir(parents=True, exist_ok=True)
        if not self.registry_file.exists():
            self._save_registry({})
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from file."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self, registry: Dict[str, Any]):
        """Save registry to file."""
        with open(self.registry_file, 'w') as f:
            json.dump(registry, f, indent=2, default=str)
    
    def register_model(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a trained model.
        
        Returns:
            Model ID
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_id = f"{model_name}_{timestamp}"
        
        # Save model file
        model_path = self.registry_path / f"{model_id}.joblib"
        joblib.dump(model, model_path)
        
        # Update registry
        registry = self._load_registry()
        registry[model_id] = {
            'model_name': model_name,
            'model_path': str(model_path),
            'metrics': metrics,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
            'version': len([k for k in registry if k.startswith(model_name)]) + 1
        }
        self._save_registry(registry)
        
        self.logger.info(f"   💾 Registered model: {model_id}")
        return model_id
    
    def load_model(self, model_id: str) -> Any:
        """Load a model by ID."""
        registry = self._load_registry()
        
        if model_id not in registry:
            raise ValueError(f"Model not found: {model_id}")
        
        model_path = registry[model_id]['model_path']
        return joblib.load(model_path)
    
    def get_best_model(self, metric: str = 'test_roc_auc') -> Optional[str]:
        """Get the best model ID by a metric."""
        registry = self._load_registry()
        
        if not registry:
            return None
        
        best_id = max(
            registry.keys(),
            key=lambda k: registry[k]['metrics'].get(metric, 0)
        )
        return best_id
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        registry = self._load_registry()
        return [
            {'id': k, **v}
            for k, v in registry.items()
        ]
    
    def delete_model(self, model_id: str):
        """Delete a model from registry."""
        registry = self._load_registry()
        
        if model_id in registry:
            model_path = Path(registry[model_id]['model_path'])
            if model_path.exists():
                model_path.unlink()
            del registry[model_id]
            self._save_registry(registry)
