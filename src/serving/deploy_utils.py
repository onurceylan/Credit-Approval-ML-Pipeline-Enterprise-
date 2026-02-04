"""
Deployment Utilities
====================

Utilities for model deployment, artifact packaging, and MLOps readiness.
"""

import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import joblib

from ..core.config import PipelineConfig


class DeploymentPackager:
    """
    Handles packaging of model artifacts for production deployment.
    """
    
    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
    
    def package_for_deployment(
        self,
        model_name: str,
        model_path: str,
        feature_engineer_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Package model and its dependencies for deployment.
        """
        self.logger.info(f"📦 Packaging {model_name} for deployment...")
        
        # Create deployment directory
        deploy_dir = Path(self.config.output_dir) / self.config.final_model_dir
        deploy_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for packaged files
        target_model = deploy_dir / "model.joblib"
        target_fe = deploy_dir / "feature_engineer.joblib"
        target_manifest = deploy_dir / "manifest.json"
        
        # Copy artifacts
        shutil.copy2(model_path, target_model)
        shutil.copy2(feature_engineer_path, target_fe)
        
        # Create manifest
        manifest = {
            'model_name': model_name,
            'version': self.config.version,
            'packaged_at': datetime.now().isoformat(),
            'features': metadata.get('features', []),
            'performance': metadata.get('performance', {}),
            'thresholds': {
                'accuracy': self.config.accuracy_threshold,
                'confidence': self.config.confidence_threshold
            }
        }
        
        with open(target_manifest, 'w') as f:
            json.dump(manifest, f, indent=4)
            
        # Create a zip for easy transfer
        zip_path = Path(self.config.output_dir) / f"deployment_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(target_model, arcname="model.joblib")
            zipf.write(target_fe, arcname="feature_engineer.joblib")
            zipf.write(target_manifest, arcname="manifest.json")
            
        self.logger.info(f"   ✅ Deployment package created: {zip_path}")
        return zip_path


class ProductionValidator:
    """
    Performs final sanity checks on deployment artifacts.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def validate_package(self, package_dir: Path) -> bool:
        """
        Validate that the package contains all required files and is loadable.
        """
        required_files = ['model.joblib', 'feature_engineer.joblib', 'manifest.json']
        
        for f in required_files:
            if not (package_dir / f).exists():
                self.logger.error(f"❌ Missing required file in package: {f}")
                return False
                
        try:
            # Test loading
            joblib.load(package_dir / "model.joblib")
            joblib.load(package_dir / "feature_engineer.joblib")
            with open(package_dir / "manifest.json", 'r') as manifest_f:
                json.load(manifest_f)
            self.logger.info("   ✅ Production package validation passed")
            return True
        except Exception as e:
            self.logger.error(f"❌ Production package validation failed: {str(e)}")
            return False
