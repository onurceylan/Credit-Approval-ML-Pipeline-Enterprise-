#!/usr/bin/env python3
"""
Credit Approval ML Pipeline - Main Entry Point
===============================================

MLOps-ready machine learning pipeline for credit approval prediction.

This script provides the main entry point for the pipeline, supporting:
- Training pipeline execution
- Inference pipeline execution
- Google Colab / Kaggle / Local environments

Usage:
    python main.py                    # Run training pipeline
    python main.py --mode train       # Explicit training mode
    python main.py --mode infer       # Run inference mode
    python main.py --dry-run          # Test imports only

Author: Credit Approval Team
Version: 3.0.0
Architecture: MLOps-Ready Production Architecture
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime


def setup_environment():
    """Setup environment for Colab/Kaggle/Local."""
    # Check for Colab
    try:
        import google.colab
        from google.colab import drive
        print("🌐 Google Colab detected")
        drive.mount('/content/drive')
        return 'colab'
    except ImportError:
        pass
    
    # Check for Kaggle
    if Path('/kaggle').exists():
        print("🌐 Kaggle environment detected")
        return 'kaggle'
    
    print("💻 Local environment detected")
    return 'local'


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Credit Approval ML Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mode', 
        choices=['train', 'infer'],
        default='train',
        help='Pipeline mode'
    )
    parser.add_argument('--trials', type=int, default=50, help='Optuna trials')
    parser.add_argument('--timeout', type=int, default=1800, help='Optuna timeout (seconds)')
    parser.add_argument('--cv-folds', type=int, default=5, help='CV folds')
    parser.add_argument('--no-optimize', action='store_true', help='Skip optimization')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU')
    parser.add_argument('--output-dir', type=str, default='ml_pipeline_output', help='Output directory')
    parser.add_argument('--dry-run', action='store_true', help='Test imports only')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    print("=" * 70)
    print("🚀 CREDIT APPROVAL ML PIPELINE v3.0")
    print("   MLOps-Ready Production Architecture")
    print("=" * 70)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    args = parse_arguments()
    env = setup_environment()
    
    # Import modules
    print("\n📦 Importing pipeline modules...")
    
    try:
        from src.core.config import PipelineConfig
        from src.core.logger import setup_logger
        from src.pipelines.training_pipeline import TrainingPipeline
        from src.pipelines.inference_pipeline import InferencePipeline
        
        print("✅ All modules imported successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Install dependencies: pip install -r requirements.txt")
        sys.exit(1)
    
    if args.dry_run:
        print("\n🧪 Dry run complete - imports verified!")
        return 0
    
    # Setup
    logger = setup_logger().logger
    
    config = PipelineConfig(
        optuna_trials=args.trials,
        optuna_timeout=args.timeout,
        cv_folds=args.cv_folds,
        use_gpu=not args.no_gpu,
        output_dir=args.output_dir
    )
    
    if args.mode == 'train':
        # Training mode
        pipeline = TrainingPipeline(
            config=config,
            logger=logger,
            optimize=not args.no_optimize
        )
        
        results = pipeline.run()
        
        print(f"\n🏆 Best Model: {results['best_model']}")
        print(f"📁 Outputs saved to: {config.output_dir}/")
        
    else:
        # Inference mode
        model_path = f"{config.output_dir}/{config.final_model_dir}/model.joblib"
        fe_path = f"{config.output_dir}/{config.final_model_dir}/feature_engineer.joblib"
        
        pipeline = InferencePipeline(
            model_path=model_path,
            feature_engineer_path=fe_path,
            config=config,
            logger=logger
        )
        
        print("✅ Inference pipeline ready")
        print("   Use pipeline.predict_single(features) for predictions")
    
    print(f"\n📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
