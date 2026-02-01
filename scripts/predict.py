#!/usr/bin/env python3
"""
Prediction Script
=================

CLI script for making predictions with trained models.

Usage:
    python scripts/predict.py --input data.csv --output predictions.csv
    python scripts/predict.py --single '{"DAYS_BIRTH": -10000, "AMT_INCOME_TOTAL": 100000}'
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.core.config import get_config
from src.pipelines.inference_pipeline import InferencePipeline


def parse_args():
    parser = argparse.ArgumentParser(description='Make Credit Approval Predictions')
    
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--feature-engineer', type=str, help='Path to feature engineer file')
    parser.add_argument('--input', type=str, help='Input CSV file')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV file')
    parser.add_argument('--single', type=str, help='Single prediction JSON')
    
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config()
    
    # Default paths
    model_path = args.model or f"{config.output_dir}/{config.final_model_dir}/model.joblib"
    fe_path = args.feature_engineer or f"{config.output_dir}/{config.final_model_dir}/feature_engineer.joblib"
    
    # Initialize pipeline
    pipeline = InferencePipeline(
        model_path=model_path,
        feature_engineer_path=fe_path,
        config=config
    )
    
    if args.single:
        # Single prediction
        features = json.loads(args.single)
        result = pipeline.predict_single(features)
        print(json.dumps(result, indent=2))
    
    elif args.input:
        # Batch prediction
        df = pd.read_csv(args.input)
        results = pipeline.predict_batch(df, return_probabilities=True)
        results.to_csv(args.output, index=False)
        print(f"✅ Predictions saved to {args.output}")
    
    else:
        print("Error: Provide --input for batch or --single for single prediction")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
