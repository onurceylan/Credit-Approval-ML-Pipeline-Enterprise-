# Credit Approval ML Pipeline

> **MLOps-Ready Production Architecture** for Credit Card Approval Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🏗️ Architecture Overview

This project implements **MLOps-Ready Production Architecture**, a design pattern optimized for enterprise ML systems. It separates concerns into distinct layers, enabling:

- **Modularity**: Each component is independently testable and replaceable
- **Scalability**: Easy to add new models, features, or data sources
- **Maintainability**: Clear code organization with single responsibility
- **Reproducibility**: YAML configs for experiment tracking
- **Deployability**: Docker support for containerized deployment

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      ENTRY POINTS                                │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ main.py  │  │ scripts/     │  │ Docker       │               │
│  └────┬─────┘  │ train.py     │  │ Container    │               │
│       │        │ predict.py   │  └──────────────┘               │
│       ▼        └──────────────┘                                  │
├─────────────────────────────────────────────────────────────────┤
│                      PIPELINE LAYER                              │
│  ┌─────────────────────────┐  ┌─────────────────────────┐       │
│  │  TrainingPipeline       │  │  InferencePipeline      │       │
│  │  - Orchestrates train   │  │  - Batch predictions    │       │
│  │  - Model selection      │  │  - Single predictions   │       │
│  └─────────────────────────┘  └─────────────────────────┘       │
├─────────────────────────────────────────────────────────────────┤
│                      BUSINESS LOGIC LAYER                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐    │
│  │ DataLoad │ │ Feature  │ │ Model    │ │ Model            │    │
│  │ Validate │ │ Engineer │ │ Factory  │ │ Trainer          │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                      CORE LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ ConfigLoader │  │ Logger       │  │ Custom Exceptions    │   │
│  │ (YAML)       │  │ (File+Term)  │  │ (Hierarchy)          │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                      INFRASTRUCTURE                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐     │
│  │ configs/ │  │ data/    │  │ docker/  │  │ tests/       │     │
│  │ (YAML)   │  │ (CSV)    │  │ (Deploy) │  │ (pytest)     │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
credit-approval/
├── configs/                    # YAML configuration files
│   ├── base.yaml              # Project settings, data paths
│   ├── training.yaml          # Model hyperparameters
│   └── deployment.yaml        # Business params, thresholds
│
├── src/                        # Source code package
│   ├── __init__.py            # Package exports
│   ├── core/                  # Core utilities
│   │   ├── config.py          # YAML config loader
│   │   ├── logger.py          # Logging system
│   │   └── exceptions.py      # Custom exceptions
│   │
│   ├── data/                  # Data layer
│   │   ├── loader.py          # Multi-env data loading
│   │   └── validator.py       # Data validation
│   │
│   ├── features/              # Feature engineering
│   │   ├── engineer.py        # Feature creation
│   │   └── preprocessor.py    # Preprocessing pipeline
│   │
│   ├── models/                # Model layer
│   │   ├── factory.py         # Model factory (GPU/CPU)
│   │   └── registry.py        # Model versioning
│   │
│   ├── training/              # Training layer
│   │   ├── trainer.py         # Model training
│   │   └── optimizer.py       # Optuna integration
│   │
│   ├── evaluation/            # Evaluation layer
│   │   ├── evaluator.py       # Model evaluation
│   │   └── metrics.py         # Business metrics
│   │
│   ├── pipelines/             # Pipeline orchestration
│   │   ├── base.py            # Abstract pipeline
│   │   ├── training_pipeline.py
│   │   └── inference_pipeline.py
│   │
│   └── serving/               # Production serving
│       └── predictor.py       # API-ready predictor
│
├── tests/                      # Unit tests
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
│
├── docker/                     # Containerization
│   ├── Dockerfile             # Multi-stage build
│   └── docker-compose.yml     # Service definitions
│
├── scripts/                    # CLI scripts
│   ├── train.py               # Training CLI
│   └── predict.py             # Prediction CLI
│
├── data/
│   ├── raw/                   # Original CSV files
│   └── processed/             # Transformed data
│
├── ml_pipeline_output/         # Pipeline outputs
│   ├── models/                # Trained models (.joblib)
│   ├── plots/                 # Visualizations
│   ├── results/               # Reports (JSON, CSV)
│   ├── logs/                  # Execution logs
│   └── final_model/           # Deployment artifacts
│
├── main.py                     # Main entry point
├── setup.py                    # Package installation
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/example/credit-approval.git
cd credit-approval

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### 2. Run Training Pipeline

```bash
# Basic training
python main.py

# With custom parameters
python main.py --trials 100 --cv-folds 10 --no-gpu

# Using CLI script
python scripts/train.py --trials 50
```

### 3. Make Predictions

```bash
# Single prediction
python scripts/predict.py --single '{"DAYS_BIRTH": -10000, "AMT_INCOME_TOTAL": 100000}'

# Batch prediction
python scripts/predict.py --input customers.csv --output predictions.csv
```

---

## 🌐 Environment Support

### Google Colab

```python
# 1. Upload project to Google Drive

# 2. In Colab notebook:
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/credit-approval

!pip install -r requirements.txt

!python main.py
```

### Kaggle

```python
# Data is auto-detected from /kaggle/input/
!pip install -r requirements.txt
!python main.py
```

### Docker

```bash
# Build and run training
docker-compose -f docker/docker-compose.yml up training

# Run inference service
docker-compose -f docker/docker-compose.yml up inference
```

---

## 🔧 Configuration

All settings are in YAML files under `configs/`:

### base.yaml
```yaml
project:
  name: "credit-approval-ml"
  version: "3.0.0"

model:
  random_state: 42
  cv_folds: 5
  test_size: 0.1
```

### Environment Variables

Override configs with environment variables:
```bash
export ML_OPTUNA_TRIALS=100
export ML_GPU_ENABLED=false
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📊 Supported Models

| Model | GPU Support | Auto-Optimization |
|-------|-------------|-------------------|
| XGBoost | ✅ | ✅ |
| LightGBM | ✅ | ✅ |
| CatBoost | ✅ | ✅ |
| RandomForest | ❌ | ✅ |
| GradientBoosting | ❌ | ✅ |
| LogisticRegression | ❌ | ✅ |

---

## 🏛️ Design Patterns Used

1. **Factory Pattern**: `ModelFactory` creates models with consistent interface
2. **Pipeline Pattern**: `TrainingPipeline` / `InferencePipeline` orchestrate workflows
3. **Registry Pattern**: `ModelRegistry` manages model versioning
4. **Strategy Pattern**: Different preprocessing strategies per data type
5. **Dependency Injection**: Components receive config/logger via constructor

---

## 📈 Pipeline Flow

```
1. Data Loading     → Load CSV files, detect environment
2. Data Validation  → Check columns, types, quality
3. Target Creation  → Temporal split to prevent leakage
4. Data Splitting   → Stratified train/val/test splits
5. Feature Engineering → Create derived features
6. Hyperparameter Optimization → Optuna-based tuning
7. Model Training   → Train all available models
8. Evaluation       → Test set metrics, cross-validation
9. Model Selection  → Composite scoring, best model
10. Business Analysis → Cost-benefit, ROI calculation
11. Deployment Prep  → Save final model and artifacts
```

---

## 📦 Outputs

After running the pipeline, find outputs in `ml_pipeline_output/`:

- `models/` - All trained models with registry
- `plots/` - Confusion matrices, ROC curves, feature importance
- `results/` - Evaluation reports, business case document
- `logs/` - Detailed execution logs
- `final_model/` - Deployment-ready model and feature engineer

---

## 📄 License

MIT License - see LICENSE file for details.