# Credit Approval ML Pipeline

> **Clean Architecture + MLOps-Ready Production Architecture**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🏗️ Architecture Overview

This project implements **Clean Architecture** combined with **MLOps-Ready Production Architecture** principles:

- **Clean Architecture**: Separation of concerns across layers (Entities → Use Cases → Adapters → Frameworks)
- **MLOps-Ready**: Reproducibility, configuration management, pipeline separation, model registry, containerization

```
┌─────────────────────────────────────────────────────────────────┐
│                      ENTRY POINTS                                │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ main.py  │  │ scripts/     │  │ docker/      │               │
│  └────┬─────┘  └──────────────┘  └──────────────┘               │
│       ▼                                                          │
├─────────────────────────────────────────────────────────────────┤
│                      PIPELINE LAYER                              │
│  ┌─────────────────────────┐  ┌─────────────────────────┐       │
│  │  TrainingPipeline       │  │  InferencePipeline      │       │
│  └─────────────────────────┘  └─────────────────────────┘       │
├─────────────────────────────────────────────────────────────────┤
│                      BUSINESS LOGIC LAYER                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐    │
│  │ Data     │ │ Features │ │ Models   │ │ Training         │    │
│  │ Loader   │ │ Engineer │ │ Factory  │ │ Evaluation       │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                      CORE LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ ConfigLoader │  │ Logger       │  │ Exceptions           │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                      INFRASTRUCTURE                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐     │
│  │ configs/ │  │ data/    │  │ docker/  │  │ tests/       │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
credit-approval/
│
├── configs/                        # 📋 YAML Configuration Files
│   ├── base.yaml                   #    Project settings, paths, random state
│   ├── training.yaml               #    Model hyperparameters, Optuna settings
│   └── deployment.yaml             #    Business costs, deployment thresholds
│
├── src/                            # 📦 Source Code Package
│   ├── __init__.py                 #    Package exports
│   │
│   ├── core/                       # 🔧 Core Utilities
│   │   ├── __init__.py
│   │   ├── config.py               #    YAML ConfigLoader + PipelineConfig dataclass
│   │   ├── logger.py               #    Colored logging with file output
│   │   └── exceptions.py           #    Custom exception hierarchy
│   │
│   ├── data/                       # 📥 Data Layer
│   │   ├── __init__.py
│   │   ├── loader.py               #    Multi-environment data loading
│   │   └── validator.py            #    Data validation and quality checks
│   │
│   ├── features/                   # 🔬 Feature Engineering
│   │   ├── __init__.py
│   │   ├── engineer.py             #    FeatureEngineer (fit-transform pattern)
│   │   └── preprocessor.py         #    TargetCreator, DataSplitter, Preprocessor
│   │
│   ├── models/                     # 🤖 Model Layer
│   │   ├── __init__.py
│   │   ├── factory.py              #    ModelFactory (GPU/CPU auto-detection)
│   │   └── registry.py             #    ModelRegistry (versioning, metadata)
│   │
│   ├── training/                   # 🏋️ Training Layer
│   │   ├── __init__.py
│   │   ├── trainer.py              #    ModelTrainer with CV and metrics
│   │   └── optimizer.py            #    Optuna HyperparameterOptimizer
│   │
│   ├── evaluation/                 # 📊 Evaluation Layer
│   │   ├── __init__.py
│   │   ├── evaluator.py            #    ModelEvaluator, model selection
│   │   └── metrics.py              #    MetricsCalculator, BusinessAnalyzer
│   │
│   ├── pipelines/                  # 🔄 Pipeline Orchestration
│   │   ├── __init__.py
│   │   ├── base.py                 #    BasePipeline abstract class
│   │   ├── training_pipeline.py    #    Complete training workflow
│   │   └── inference_pipeline.py   #    Batch/single prediction workflow
│   │
│   └── serving/                    # 🚀 Production Serving
│       ├── __init__.py
│       └── predictor.py            #    ModelPredictor (API-ready)
│
├── tests/                          # 🧪 Unit Tests (pytest)
│   ├── __init__.py
│   ├── test_data.py                #    Data module tests
│   ├── test_features.py            #    Feature engineering tests
│   └── test_models.py              #    Model factory/registry tests
│
├── docker/                         # 🐳 Containerization
│   ├── Dockerfile                  #    Multi-stage build (dev/prod/inference)
│   └── docker-compose.yml          #    Service definitions
│
├── scripts/                        # 💻 CLI Tools
│   ├── train.py                    #    Training CLI with arguments
│   └── predict.py                  #    Prediction CLI (batch/single)
│
├── data/                           # 📂 Data Directory
│   ├── raw/                        #    Original CSV files
│   │   ├── application_record.csv  #    (54 MB)
│   │   └── credit_record.csv       #    (15 MB)
│   └── processed/                  #    Transformed data (gitignored)
│
├── ml_pipeline_output/             # 📤 Pipeline Outputs (gitignored)
│   ├── models/                     #    Trained models (.joblib)
│   ├── plots/                      #    Visualizations (.png)
│   ├── results/                    #    Reports (JSON, CSV, TXT)
│   ├── logs/                       #    Execution logs
│   └── final_model/                #    Deployment artifacts
│
├── main.py                         # 🚀 Main Entry Point
├── setup.py                        # 📦 Package Installation
├── requirements.txt                # 📋 Dependencies
├── .gitignore                      # 🚫 Git Ignore Rules
└── README.md                       # 📖 This File
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone and setup
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

### Run Training

```bash
# Basic training
python main.py

# With custom parameters
python main.py --trials 100 --cv-folds 10 --no-gpu

# Using CLI script
python scripts/train.py --trials 50 --no-optimize
```

### Make Predictions

```bash
# Single prediction
python scripts/predict.py --single '{"DAYS_BIRTH": -10000, "AMT_INCOME_TOTAL": 100000}'

# Batch prediction
python scripts/predict.py --input customers.csv --output predictions.csv
```

---

## 🌐 Environment Support

| Environment | Status | Data Location |
|-------------|--------|---------------|
| **Local** | ✅ | `data/raw/` |
| **Google Colab** | ✅ | `/content/drive/MyDrive/...` |
| **Kaggle** | ✅ | `/kaggle/input/...` |
| **Docker** | ✅ | Mounted volumes |

### Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/credit-approval
!pip install -r requirements.txt
!python main.py
```

### Docker

```bash
# Run training
docker-compose -f docker/docker-compose.yml up training

# Run inference
docker-compose -f docker/docker-compose.yml up inference
```

---

## 🔧 Configuration

All settings are externalized in YAML files under `configs/`:

| File | Purpose |
|------|---------|
| `base.yaml` | Project name, version, data paths, random state |
| `training.yaml` | Model hyperparameters, Optuna settings, CV folds |
| `deployment.yaml` | Business costs, deployment thresholds |

### Environment Variable Overrides

```bash
export ML_OPTUNA_TRIALS=100
export ML_GPU_ENABLED=false
export ML_RANDOM_STATE=123
```

---

## 📊 Supported Models

| Model | GPU Support | Optuna Tuning |
|-------|:-----------:|:-------------:|
| XGBoost | ✅ | ✅ |
| LightGBM | ✅ | ✅ |
| CatBoost | ✅ | ✅ |
| RandomForest | ❌ | ✅ |
| GradientBoosting | ❌ | ✅ |
| LogisticRegression | ❌ | ✅ |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 🏛️ Design Patterns

| Pattern | Implementation | Purpose |
|---------|----------------|---------|
| **Factory** | `ModelFactory` | Create models with consistent interface |
| **Pipeline** | `TrainingPipeline`, `InferencePipeline` | Orchestrate workflows |
| **Registry** | `ModelRegistry` | Version and track models |
| **Strategy** | Feature preprocessing strategies | Flexible data transformations |
| **Dependency Injection** | Constructor-based config/logger | Testability |

---

## 📈 Pipeline Flow

```
 1. Load Data          → Multi-env data loading
 2. Validate Data      → Quality checks, ID overlap
 3. Create Target      → Temporal split (no leakage)
 4. Split Data         → Stratified train/val/test
 5. Engineer Features  → Derived features, scaling
 6. Optimize Params    → Optuna hyperparameter tuning
 7. Train Models       → All available models
 8. Evaluate           → Test metrics, cross-validation
 9. Select Best        → Composite scoring
10. Business Analysis  → Cost-benefit, ROI
11. Save Artifacts     → Models, reports, plots
```

---

## 📄 License

MIT License - see LICENSE file for details.