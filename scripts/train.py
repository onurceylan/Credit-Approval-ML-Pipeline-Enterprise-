#!/usr/bin/env python3
"""
Training Script
===============

CLI script for running the training pipeline.

Usage:
    python scripts/train.py
    python scripts/train.py --trials 100 --no-optimize
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import get_config, PipelineConfig
from src.core.logger import setup_logger
from src.pipelines.training_pipeline import TrainingPipeline


def parse_args():
    parser = argparse.ArgumentParser(description='Train Credit Approval ML Models')
    
    parser.add_argument('--trials', type=int, default=50, help='Optuna optimization trials')
    parser.add_argument('--timeout', type=int, default=1800, help='Optuna timeout in seconds')
    parser.add_argument('--no-optimize', action='store_true', help='Skip hyperparameter optimization')
    parser.add_argument('--cv-folds', type=int, default=5, help='Cross-validation folds')
    parser.add_argument('--output-dir', type=str, default='ml_pipeline_output', help='Output directory')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logger
    logger = setup_logger().logger
    
    # Override config
    config = PipelineConfig(
        optuna_trials=args.trials,
        optuna_timeout=args.timeout,
        cv_folds=args.cv_folds,
        output_dir=args.output_dir
    )
    
    # Run pipeline
    pipeline = TrainingPipeline(
        config=config,
        logger=logger,
        optimize=not args.no_optimize
    )
    
    results = pipeline.run()
    
    print(f"\n✅ Training complete! Best model: {results['best_model']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
