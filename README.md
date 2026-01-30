# Credit Approval ML Pipeline

> **Clean Architecture + MLOps - Google Colab Uyumlu (.ipynb)**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Colab Ready](https://img.shields.io/badge/Colab-Ready-orange.svg)](https://colab.research.google.com/)

---

## 🏗️ Architecture Overview

Bu proje **Clean Architecture** ve **MLOps-Ready** prensiplerine göre yapılandırılmıştır. Tüm dosyalar **Google Colab uyumlu** `.ipynb` formatındadır.

```
┌─────────────────────────────────────────────────────────────────┐
│                      ENTRY POINTS                                │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │ main.ipynb   │  │ scripts/     │                             │
│  └──────────────┘  └──────────────┘                             │
├─────────────────────────────────────────────────────────────────┤
│                      PIPELINE LAYER                              │
│  ┌─────────────────────────┐  ┌─────────────────────────┐       │
│  │  training_pipeline      │  │  inference_pipeline     │       │
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
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
credit-approval/
│
├── configs/                        # 📋 YAML Configuration Files
│   ├── base.yaml                   #    Project settings, paths
│   ├── training.yaml               #    Model hyperparameters
│   └── deployment.yaml             #    Business thresholds
│
├── src/                            # 📦 Source Code (All .ipynb)
│   ├── __init__.ipynb              #    Package exports
│   │
│   ├── core/                       # 🔧 Core Utilities
│   │   ├── __init__.ipynb
│   │   ├── config.ipynb            #    YAML ConfigLoader
│   │   ├── logger.ipynb            #    Colored logging
│   │   └── exceptions.ipynb        #    Custom exceptions
│   │
│   ├── data/                       # 📥 Data Layer
│   │   ├── __init__.ipynb
│   │   ├── loader.ipynb            #    Multi-environment data loading
│   │   └── validator.ipynb         #    Data validation
│   │
│   ├── features/                   # 🔬 Feature Engineering
│   │   ├── __init__.ipynb
│   │   ├── engineer.ipynb          #    FeatureEngineer
│   │   └── preprocessor.ipynb      #    DataPreprocessor
│   │
│   ├── models/                     # 🤖 Model Layer
│   │   ├── __init__.ipynb
│   │   ├── factory.ipynb           #    ModelFactory (GPU/CPU)
│   │   └── registry.ipynb          #    ModelRegistry
│   │
│   ├── training/                   # 🏋️ Training Layer
│   │   ├── __init__.ipynb
│   │   ├── trainer.ipynb           #    ModelTrainer
│   │   └── optimizer.ipynb         #    HyperparameterOptimizer
│   │
│   ├── evaluation/                 # 📊 Evaluation Layer
│   │   ├── __init__.ipynb
│   │   ├── evaluator.ipynb         #    ModelEvaluator
│   │   └── metrics.ipynb           #    BusinessAnalyzer
│   │
│   ├── pipelines/                  # 🔄 Pipeline Orchestration
│   │   ├── __init__.ipynb
│   │   ├── base.ipynb              #    BasePipeline
│   │   ├── training_pipeline.ipynb #    Training workflow
│   │   └── inference_pipeline.ipynb#    Prediction workflow
│   │
│   └── serving/                    # 🚀 Production Serving
│       ├── __init__.ipynb
│       └── predictor.ipynb         #    ModelPredictor
│
├── tests/                          # 🧪 Unit Tests (All .ipynb)
│   ├── __init__.ipynb
│   ├── test_data.ipynb
│   ├── test_features.ipynb
│   └── test_models.ipynb
│
├── scripts/                        # 💻 CLI Tools (All .ipynb)
│   ├── train.ipynb                 #    Training CLI
│   └── predict.ipynb               #    Prediction CLI
│
├── docker/                         # 🐳 Containerization
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── data/                           # 📂 Data Directory
│   ├── raw/                        #    Original CSV files
│   └── processed/                  #    Transformed data
│
├── ml_pipeline_output/             # 📤 Pipeline Outputs
│   ├── models/
│   ├── plots/
│   ├── results/
│   ├── logs/
│   └── final_model/
│
├── main.ipynb                      # 🚀 Main Entry Point
├── setup.ipynb                     # 📦 Package Installation
├── requirements.txt                # 📋 Dependencies
└── README.md                       # 📖 This File
```

---

## 🚀 Google Colab'da Kullanım

### 1. Projeyi Drive'a Yükle

Tüm proje klasörünü Google Drive'a yükleyin.

### 2. Drive'ı Bağla

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Proje Dizinine Git

```python
%cd /content/drive/MyDrive/credit-approval
```

### 4. Bağımlılıkları Yükle

```python
!pip install -r requirements.txt
```

### 5. Pipeline'ı Çalıştır

`main.ipynb` dosyasını açın ve hücreleri çalıştırın.

---

## 📊 Desteklenen Modeller

| Model | GPU Desteği | Optuna Tuning |
|-------|:-----------:|:-------------:|
| XGBoost | ✅ | ✅ |
| LightGBM | ✅ | ✅ |
| CatBoost | ✅ | ✅ |
| RandomForest | ❌ | ✅ |
| GradientBoosting | ❌ | ✅ |
| LogisticRegression | ❌ | ✅ |

---

## 📈 Pipeline Flow

```
 1. Load Data          → Veri yükleme
 2. Validate Data      → Veri doğrulama
 3. Create Target      → Hedef değişken oluşturma
 4. Split Data         → Train/Val/Test ayrımı
 5. Engineer Features  → Özellik mühendisliği
 6. Optimize Params    → Hiperparametre optimizasyonu
 7. Train Models       → Model eğitimi
 8. Evaluate           → Değerlendirme
 9. Select Best        → En iyi model seçimi
10. Business Analysis  → İş etkisi analizi
11. Save Artifacts     → Sonuçları kaydetme
```

---

## 📄 License

MIT License