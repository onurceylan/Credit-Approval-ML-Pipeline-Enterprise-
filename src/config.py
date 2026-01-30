"""
Configuration Module
====================

Contains configuration classes, constants, and custom exceptions
for the Credit Approval ML Pipeline.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


# =============================================================================
# CONFIGURATION CLASS
# =============================================================================

@dataclass
class ModelConfig:
    """
    Enhanced configuration class for the ML pipeline.
    
    Supports multiple environments: Kaggle, Colab, Local
    """
    
    # Data paths (configurable based on environment)
    data_paths: Dict[str, str] = field(default_factory=dict)
    
    # Model parameters
    cv_folds: int = 5
    test_size: float = 0.1
    val_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    
    # Optuna parameters
    optuna_trials: int = 50
    optuna_timeout: int = 1800  # 30 minutes
    
    # Statistical testing parameters
    statistical_alpha: float = 0.05
    bonferroni_correction: bool = True
    
    # Business parameters
    cost_false_positive: float = 5000  # Cost of approving bad credit
    cost_false_negative: float = 500   # Cost of rejecting good credit
    revenue_per_approval: float = 1200 # Revenue from approved good credit
    
    # Deployment parameters
    deployment_accuracy_threshold: float = 0.75
    deployment_confidence_threshold: float = 0.70
    deployment_stability_threshold: float = 0.05
    
    # Output directories
    output_dir: str = "ml_pipeline_output"
    models_dir: str = "models"
    plots_dir: str = "plots"
    results_dir: str = "results"
    logs_dir: str = "logs"
    final_model_dir: str = "final_model"
    
    # GPU settings
    use_gpu: bool = True
    gpu_device_id: int = 0
    
    def __post_init__(self):
        """Initialize default data paths based on environment."""
        if not self.data_paths:
            self.data_paths = self._detect_data_paths()
    
    def _detect_data_paths(self) -> Dict[str, str]:
        """Detect data paths based on running environment."""
        
        # Environment detection order: Colab -> Kaggle -> Local
        possible_paths = [
            # Google Colab with Drive mount
            {
                'application': '/content/drive/MyDrive/credit-approval/data/raw/application_record.csv',
                'credit': '/content/drive/MyDrive/credit-approval/data/raw/credit_record.csv',
                'env': 'Colab (Drive)'
            },
            # Google Colab local
            {
                'application': '/content/application_record.csv',
                'credit': '/content/credit_record.csv',
                'env': 'Colab (Local)'
            },
            # Kaggle environment - primary
            {
                'application': '/kaggle/input/credit-card-approval-prediction/application_record.csv',
                'credit': '/kaggle/input/credit-card-approval-prediction/credit_record.csv',
                'env': 'Kaggle'
            },
            # Kaggle environment - alternative
            {
                'application': '/kaggle/input/application_record.csv',
                'credit': '/kaggle/input/credit_record.csv',
                'env': 'Kaggle (Alt)'
            },
            # Local environment - data folder
            {
                'application': 'data/raw/application_record.csv',
                'credit': 'data/raw/credit_record.csv',
                'env': 'Local (data/raw)'
            },
            # Local environment - current directory
            {
                'application': 'application_record.csv',
                'credit': 'credit_record.csv',
                'env': 'Local (current)'
            }
        ]
        
        for paths in possible_paths:
            app_path = paths['application']
            credit_path = paths['credit']
            
            if Path(app_path).exists() and Path(credit_path).exists():
                print(f"✅ Found data files in: {paths['env']}")
                return {'application': app_path, 'credit': credit_path}
        
        # Default to first option (user will need to adjust)
        print(f"⚠️ Data files not found, using default paths. Please update config.")
        return {
            'application': 'data/raw/application_record.csv',
            'credit': 'data/raw/credit_record.csv'
        }
    
    def get_output_path(self, subdir: str) -> Path:
        """Get full output path for a subdirectory."""
        return Path(self.output_dir) / subdir
    
    def get_models_path(self) -> Path:
        """Get models output directory path."""
        return self.get_output_path(self.models_dir)
    
    def get_plots_path(self) -> Path:
        """Get plots output directory path."""
        return self.get_output_path(self.plots_dir)
    
    def get_results_path(self) -> Path:
        """Get results output directory path."""
        return self.get_output_path(self.results_dir)
    
    def get_logs_path(self) -> Path:
        """Get logs output directory path."""
        return self.get_output_path(self.logs_dir)
    
    def get_final_model_path(self) -> Path:
        """Get final model output directory path."""
        return self.get_output_path(self.final_model_dir)


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class MLPipelineError(Exception):
    """Base exception for ML pipeline errors."""
    pass


class DataValidationError(MLPipelineError):
    """Exception for data validation errors."""
    pass


class ModelTrainingError(MLPipelineError):
    """Exception for model training errors."""
    pass


class DeploymentError(MLPipelineError):
    """Exception for deployment preparation errors."""
    pass


class FeatureEngineeringError(MLPipelineError):
    """Exception for feature engineering errors."""
    pass


class EvaluationError(MLPipelineError):
    """Exception for model evaluation errors."""
    pass


# =============================================================================
# CONSTANTS
# =============================================================================

# Target variable definitions
BAD_CREDIT_STATUSES = ['2', '3', '4', '5']  # 60+ days overdue

# Default cutoff for temporal split (prevents data leakage)
TEMPORAL_CUTOFF_MONTHS = -6

# Model types
MODEL_TYPES = {
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM', 
    'catboost': 'CatBoost',
    'sklearn': 'Scikit-learn'
}

# Visualization settings
PLOT_STYLE = {
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white'
}

# Output file names
OUTPUT_FILES = {
    'validation_report': 'data_validation_report.json',
    'preprocessing_info': 'preprocessing_info.json',
    'training_summary': 'training_summary.json',
    'evaluation_report': 'evaluation_report.json',
    'model_comparison': 'model_comparison.csv',
    'executive_summary': 'executive_summary_report.txt',
    'business_case': 'business_case_document.txt',
    'implementation_guide': 'implementation_guide.txt',
    'model_metadata': 'model_metadata.json'
}

# Plot file names
PLOT_FILES = {
    'training_results': 'training_results.png',
    'model_comparison': 'model_evaluation_comparison.png',
    'business_impact': 'business_impact_analysis.png',
    'model_selection': 'model_selection_final.png',
    'confusion_matrices': 'confusion_matrices.png',
    'roc_curves': 'roc_curves.png',
    'feature_importance': 'feature_importance.png'
}
