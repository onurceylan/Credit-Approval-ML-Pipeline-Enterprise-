"""
Configuration Loader
====================

Loads and merges YAML configuration files with environment overrides.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

import yaml

from .exceptions import ConfigurationError


class ConfigLoader:
    """
    Loads configuration from YAML files with environment overrides.
    
    Supports:
    - Multiple config files (base, training, deployment)
    - Environment variable overrides
    - Runtime parameter overrides
    """
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self._config: Dict[str, Any] = {}
        self._loaded = False
    
    def load(self, env: str = "base") -> Dict[str, Any]:
        """
        Load configuration files.
        
        Args:
            env: Environment name (base, training, deployment)
        
        Returns:
            Merged configuration dictionary
        """
        config_files = ["base.yaml", "training.yaml", "deployment.yaml"]
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f) or {}
                    self._config = self._deep_merge(self._config, file_config)
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        self._loaded = True
        return self._config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        env_mappings = {
            'ML_RANDOM_STATE': ('model', 'random_state', int),
            'ML_CV_FOLDS': ('model', 'cv_folds', int),
            'ML_GPU_ENABLED': ('gpu', 'enabled', lambda x: x.lower() == 'true'),
            'ML_OPTUNA_TRIALS': ('training', 'optuna_trials', int),
        }
        
        for env_var, (section, key, converter) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                if section not in self._config:
                    self._config[section] = {}
                self._config[section][key] = converter(value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        if not self._loaded:
            self.load()
        
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        if not self._loaded:
            self.load()
        return self._config


@dataclass
class PipelineConfig:
    """
    Pipeline configuration dataclass.
    
    Provides typed access to configuration values.
    """
    # Project
    project_name: str = "credit-approval-ml"
    version: str = "3.0.0"
    
    # Data paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    application_file: str = "application_record.csv"
    credit_file: str = "credit_record.csv"
    
    # Model settings
    random_state: int = 42
    cv_folds: int = 5
    test_size: float = 0.1
    val_size: float = 0.1
    n_jobs: int = -1
    verbose: int = 1
    
    # Training settings
    optuna_trials: int = 50
    optuna_timeout: int = 1800
    
    # GPU settings
    use_gpu: bool = True
    gpu_device_id: int = 0
    
    # Output directories
    output_dir: str = "ml_pipeline_output"
    models_dir: str = "models"
    plots_dir: str = "plots"
    results_dir: str = "results"
    logs_dir: str = "logs"
    final_model_dir: str = "final_model"
    
    # Business parameters
    cost_false_positive: float = 5000
    cost_false_negative: float = 500
    revenue_per_approval: float = 1200
    
    # Implementation costs
    infrastructure_cost: float = 50000
    development_cost: float = 100000
    training_cost: float = 25000
    maintenance_cost: float = 30000
    discount_rate: float = 0.1
    
    # Deployment thresholds
    accuracy_threshold: float = 0.75
    confidence_threshold: float = 0.70
    stability_threshold: float = 0.05
    interpretability_threshold: float = 0.70
    
    @classmethod
    def from_yaml(cls, config_loader: ConfigLoader) -> 'PipelineConfig':
        """Create PipelineConfig from ConfigLoader."""
        return cls(
            project_name=config_loader.get('project.name', cls.project_name),
            version=config_loader.get('project.version', cls.version),
            raw_data_dir=config_loader.get('data.raw_dir', cls.raw_data_dir),
            processed_data_dir=config_loader.get('data.processed_dir', cls.processed_data_dir),
            random_state=config_loader.get('model.random_state', cls.random_state),
            cv_folds=config_loader.get('model.cv_folds', cls.cv_folds),
            test_size=config_loader.get('model.test_size', cls.test_size),
            val_size=config_loader.get('model.val_size', cls.val_size),
            n_jobs=config_loader.get('model.n_jobs', cls.n_jobs),
            optuna_trials=config_loader.get('training.optuna_trials', cls.optuna_trials),
            optuna_timeout=config_loader.get('training.optuna_timeout', cls.optuna_timeout),
            use_gpu=config_loader.get('gpu.enabled', cls.use_gpu),
            gpu_device_id=config_loader.get('gpu.device_id', cls.gpu_device_id),
            output_dir=config_loader.get('output.base_dir', cls.output_dir),
            models_dir=config_loader.get('output.models_dir', cls.models_dir),
            plots_dir=config_loader.get('output.plots_dir', cls.plots_dir),
            results_dir=config_loader.get('output.results_dir', cls.results_dir),
            logs_dir=config_loader.get('output.logs_dir', cls.logs_dir),
            final_model_dir=config_loader.get('output.final_model_dir', cls.final_model_dir),
            cost_false_positive=config_loader.get('business.cost_false_positive', cls.cost_false_positive),
            cost_false_negative=config_loader.get('business.cost_false_negative', cls.cost_false_negative),
            revenue_per_approval=config_loader.get('business.revenue_per_approval', cls.revenue_per_approval),
            infrastructure_cost=config_loader.get('business.infrastructure_cost', cls.infrastructure_cost),
            development_cost=config_loader.get('business.development_cost', cls.development_cost),
            training_cost=config_loader.get('business.training_cost', cls.training_cost),
            maintenance_cost=config_loader.get('business.maintenance_cost', cls.maintenance_cost),
            discount_rate=config_loader.get('business.discount_rate', cls.discount_rate),
            accuracy_threshold=config_loader.get('deployment.accuracy_threshold', cls.accuracy_threshold),
            confidence_threshold=config_loader.get('deployment.confidence_threshold', cls.confidence_threshold),
            stability_threshold=config_loader.get('deployment.stability_threshold', cls.stability_threshold),
            interpretability_threshold=config_loader.get('deployment.interpretability_threshold', cls.interpretability_threshold),
        )
    
    def get_output_path(self, subdir: str) -> Path:
        """Get full output path for a subdirectory."""
        return Path(self.output_dir) / subdir


# Global config instance
_config_loader: Optional[ConfigLoader] = None
_config: Optional[PipelineConfig] = None


def get_config(reload: bool = False) -> PipelineConfig:
    """Get global pipeline configuration."""
    global _config_loader, _config
    
    if _config is None or reload:
        _config_loader = ConfigLoader()
        _config_loader.load()
        _config = PipelineConfig.from_yaml(_config_loader)
    
    return _config
