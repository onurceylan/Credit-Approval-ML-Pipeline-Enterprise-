#!/usr/bin/env python3
"""
Credit Approval ML Pipeline - Main Entry Point
===============================================

Enterprise-grade machine learning pipeline for credit approval prediction.

This script orchestrates the complete ML pipeline:
1. Load and validate data
2. Create target variable (with temporal split to prevent leakage)
3. Safe data splitting
4. Feature engineering and preprocessing
5. Model training with hyperparameter optimization
6. Comprehensive evaluation
7. Business impact analysis
8. Model selection and deployment preparation

Usage:
    # Local
    python main.py

    # Google Colab
    !python main.py

    # With custom config
    python main.py --trials 100 --timeout 3600

Author: Credit Approval Team
Version: 3.0.0
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime


def setup_colab_environment():
    """
    Setup environment for Google Colab.
    
    Mounts Google Drive if running in Colab.
    """
    try:
        import google.colab
        from google.colab import drive
        
        print("🌐 Google Colab environment detected")
        
        # Mount Google Drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")
        
        # Set working directory to project folder
        project_path = '/content/drive/MyDrive/credit-approval'
        
        if Path(project_path).exists():
            import os
            os.chdir(project_path)
            print(f"📁 Working directory: {project_path}")
        else:
            print(f"⚠️ Project folder not found at {project_path}")
            print("   Please upload your project to Google Drive or adjust the path.")
        
        # Add src to path
        sys.path.insert(0, str(Path.cwd()))
        
        return True
        
    except ImportError:
        print("💻 Running in local environment")
        return False


def setup_kaggle_environment():
    """
    Setup environment for Kaggle Notebooks.
    """
    try:
        if Path('/kaggle').exists():
            print("🌐 Kaggle environment detected")
            return True
        return False
    except:
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Credit Approval ML Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--trials', 
        type=int, 
        default=50,
        help='Number of Optuna optimization trials'
    )
    
    parser.add_argument(
        '--timeout', 
        type=int, 
        default=1800,
        help='Optuna optimization timeout in seconds'
    )
    
    parser.add_argument(
        '--cv-folds', 
        type=int, 
        default=5,
        help='Number of cross-validation folds'
    )
    
    parser.add_argument(
        '--test-size', 
        type=float, 
        default=0.1,
        help='Test set size ratio'
    )
    
    parser.add_argument(
        '--val-size', 
        type=float, 
        default=0.2,
        help='Validation set size ratio'
    )
    
    parser.add_argument(
        '--no-gpu', 
        action='store_true',
        help='Disable GPU usage'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='ml_pipeline_output',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Test imports only, do not run pipeline'
    )
    
    return parser.parse_args()


def main():
    """
    Main pipeline execution function.
    
    Runs the complete ML pipeline from data loading to model deployment.
    """
    print("=" * 70)
    print("🚀 CREDIT APPROVAL ML PIPELINE v3.0")
    print("=" * 70)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment (Colab/Kaggle/Local)
    is_colab = setup_colab_environment()
    is_kaggle = setup_kaggle_environment()
    
    # Import pipeline modules
    print("\n📦 Importing pipeline modules...")
    
    try:
        from src.config import ModelConfig
        from src.utils import initialize_pipeline, memory_cleanup
        from src.data_loader import RobustDataLoader, TemporalDataSplitter
        from src.features import SafeDataSplitter, DataPreprocessingPipeline, DataQualityAnalyzer
        from src.models import ModelFactory
        from src.train import ModelTrainer, ModelPersistence, TrainingVisualizer
        from src.evaluate import run_comprehensive_evaluation
        
        print("✅ All modules imported successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Make sure you're running from the project root directory")
        print("   and all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Dry run check
    if args.dry_run:
        print("\n🧪 Dry run completed - imports verified!")
        return
    
    # =========================================================================
    # STEP 1: INITIALIZE PIPELINE
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: PIPELINE INITIALIZATION")
    print("=" * 70)
    
    # Create configuration
    config = ModelConfig(
        optuna_trials=args.trials,
        optuna_timeout=args.timeout,
        cv_folds=args.cv_folds,
        test_size=args.test_size,
        val_size=args.val_size,
        use_gpu=not args.no_gpu,
        output_dir=args.output_dir
    )
    
    # Initialize pipeline
    config, logger, dependency_manager = initialize_pipeline(config)
    
    # =========================================================================
    # STEP 2: LOAD AND VALIDATE DATA
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: DATA LOADING AND VALIDATION")
    print("=" * 70)
    
    data_loader = RobustDataLoader(config, logger)
    
    try:
        app_data, credit_data = data_loader.load_and_validate_data()
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        logger.error("\n💡 Please ensure data files are in one of these locations:")
        logger.error("   - Colab: /content/drive/MyDrive/credit-approval/data/raw/")
        logger.error("   - Kaggle: /kaggle/input/credit-card-approval-prediction/")
        logger.error("   - Local: data/raw/")
        sys.exit(1)
    
    memory_cleanup()
    
    # =========================================================================
    # STEP 3: CREATE TARGET VARIABLE
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: TARGET VARIABLE CREATION")
    print("=" * 70)
    
    temporal_splitter = TemporalDataSplitter(logger)
    processed_data = temporal_splitter.create_target_variable(app_data, credit_data)
    
    # Free memory
    del app_data, credit_data
    memory_cleanup()
    
    # =========================================================================
    # STEP 4: SAFE DATA SPLITTING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: SAFE DATA SPLITTING")
    print("=" * 70)
    
    data_splitter = SafeDataSplitter(config, logger)
    splits = data_splitter.split_data_safely(processed_data)
    
    # Free memory
    del processed_data
    memory_cleanup()
    
    # =========================================================================
    # STEP 5: FEATURE ENGINEERING AND PREPROCESSING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: FEATURE ENGINEERING AND PREPROCESSING")
    print("=" * 70)
    
    preprocessing_pipeline = DataPreprocessingPipeline(config, logger)
    processed_splits = preprocessing_pipeline.preprocess_data(splits)
    
    # Analyze data quality
    quality_analyzer = DataQualityAnalyzer(logger)
    quality_report = quality_analyzer.analyze_data_quality(splits['X_train'], processed_splits)
    
    memory_cleanup()
    
    # =========================================================================
    # STEP 6: MODEL TRAINING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: MODEL TRAINING")
    print("=" * 70)
    
    model_trainer = ModelTrainer(config, dependency_manager, logger)
    training_results = model_trainer.train_all_models(processed_splits)
    
    # Save training results
    persistence = ModelPersistence(config, logger)
    persistence.save_training_results(training_results, processed_splits)
    
    # Create training visualizations
    training_visualizer = TrainingVisualizer(config, logger)
    training_visualizer.create_training_visualizations(training_results)
    
    memory_cleanup()
    
    # =========================================================================
    # STEP 7: COMPREHENSIVE EVALUATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: COMPREHENSIVE EVALUATION")
    print("=" * 70)
    
    evaluation_output = run_comprehensive_evaluation(
        training_results,
        processed_splits,
        config,
        dependency_manager,
        logger
    )
    
    # =========================================================================
    # STEP 8: SAVE FINAL MODEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: FINAL MODEL DEPLOYMENT PREPARATION")
    print("=" * 70)
    
    best_model_name = evaluation_output['best_model']
    
    if best_model_name and best_model_name in training_results:
        best_model = training_results[best_model_name]['model']
        feature_engineer = processed_splits['feature_engineer']
        
        # Create metadata
        metadata = {
            'model_name': best_model_name,
            'model_type': training_results[best_model_name].get('model_type', ''),
            'training_timestamp': training_results[best_model_name].get('timestamp', ''),
            'deployment_timestamp': datetime.now().isoformat(),
            'performance': {
                'val_roc_auc': training_results[best_model_name].get('val_roc_auc', 0),
                'val_accuracy': training_results[best_model_name].get('val_accuracy', 0),
                'test_roc_auc': evaluation_output['evaluation_results'].get(best_model_name, {}).get('test_roc_auc', 0),
                'test_accuracy': evaluation_output['evaluation_results'].get(best_model_name, {}).get('test_accuracy', 0)
            },
            'config': {
                'cv_folds': config.cv_folds,
                'test_size': config.test_size,
                'val_size': config.val_size,
                'optuna_trials': config.optuna_trials
            },
            'feature_info': processed_splits.get('preprocessing_info', {})
        }
        
        persistence.save_final_model(best_model, best_model_name, feature_engineer, metadata)
        
        logger.info(f"✅ Final model {best_model_name} saved for deployment!")
    
    # =========================================================================
    # PIPELINE COMPLETE
    # =========================================================================
    print("\n" + "=" * 70)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"   • Best Model: {best_model_name}")
    print(f"   • Models Trained: {len([r for r in training_results.values() if r.get('success', False)])}")
    print(f"   • Output Directory: {config.output_dir}")
    
    # Output locations
    print(f"\n📁 OUTPUT LOCATIONS:")
    print(f"   • Models: {config.output_dir}/{config.models_dir}/")
    print(f"   • Plots: {config.output_dir}/{config.plots_dir}/")
    print(f"   • Results: {config.output_dir}/{config.results_dir}/")
    print(f"   • Logs: {config.output_dir}/{config.logs_dir}/")
    print(f"   • Final Model: {config.output_dir}/{config.final_model_dir}/")
    
    print(f"\n📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return evaluation_output


if __name__ == "__main__":
    main()
