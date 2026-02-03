# 📊 Credit Approval ML Pipeline

> **Hybrid MLOps Production Architecture** (Jupyter Notebook + Modular Python)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Colab Ready](https://img.shields.io/badge/Google_Colab-Ready-orange.svg)](COLAB.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Machine learning pipeline for credit approval prediction featuring statistical validation, comprehensive business impact analysis, and production deployment readiness. This system provides end-to-end ML workflow from data ingestion to stakeholder reporting.

---

## 🌟 Key Features

- 🤖 **Multi-Algorithm Training**: XGBoost, LightGBM, CatBoost, RandomForest, GradientBoosting, LogisticRegression.
- 📊 **Statistical Validation**: Friedman test with Bonferroni-corrected post-hoc analysis.
- 💼 **Business Impact Analysis**: ROI, NPV (5-Year), and Payback Period calculations.
- 🚀 **Production Ready**: Deployment artifacts, modular Python package, and CLI support.
- 🛡️ **Data Leakage Prevention**: Temporal splitting and comprehensive validation.
- ⚡ **GPU Acceleration**: CUDA support for XGBoost, LightGBM, and CatBoost.
-  **Comprehensive Visualization**: Automated 2x2 Dashboards (Performance, Time, CV, Model Type).

---

## �️ Architecture & Project Structure

This project follows a **Hybrid MLOps Architecture**, combining the interactivity of Jupyter Notebooks for exploration with the production-grade modularity of Python scripts.

```
credit-approval/
│
├── main.ipynb                    # 📓 INTERACTIVE ENTRY POINT (Google Colab / Jupyter)
├── main.py                       # 💻 CLI ENTRY POINT (Production / Terminal)
├── COLAB.md                      # 📖 Step-by-step Google Colab Guide
│
├── configs/                      # ⚙️ Pipeline Configurations (YAML)
│   ├── base.yaml                 #    General settings
│   ├── training.yaml             #    Model hyperparams & optimization spaces
│   └── deployment.yaml           #    Business logic & costs
│
├── src/                          # 📦 Core Python Package (Modular Logic)
│   ├── core/                     #    Config, Logger, Exceptions
│   ├── data/                     #    Data Loading & Validation
│   ├── features/                 #    Feature Engineering & Preprocessing
│   ├── models/                   #    Model Factory (GPU/CPU) & Registry
│   ├── training/                 #    Trainer & Optuna Optimizer
│   ├── evaluation/               #    Statistical & Financial Evaluators
│   └── pipelines/                #    End-to-end Pipeline Orchestration
│
├── scripts/                      # 🛠️ Task-specific Scripts
│   ├── train.py                  #    Standalone training script
│   └── predict.py                #    Standalone inference script
│
├── tests/                        # 🧪 Unit Tests & Data Quality Checks
├── docker/                       # 🐳 Containerization (Dockerfile, Compose)
├── requirements.txt              # 📋 Environment Dependencies
└── setup.py                      # � Package Setup (pip install -e .)
```

---

## � Output Structure

Execution results are organized into a standardized directory for versioning and reporting.

```
ml_pipeline_output/
├── 📁 models/                    # Serialized models (.joblib)
├── 📁 plots/                     # High-res visualizations (Training Results, ROC, ROI)
├── 📁 results/                   # Structured reports (JSON, Text)
│   ├── data_quality_report.json
│   ├── training_summary.json
│   ├── evaluation_report.json
│   └── business_case.txt
└── 📁 logs/                      # Execution trace logs
```

---

## 🚀 Quick Start (Google Colab)

The easiest way to run this pipeline is via Google Colab.

1.  Upload the project folder to your Google Drive.
2.  Open `main.ipynb` with Google Colab.
3.  Set Runtime to **T4 GPU** (`Runtime` -> `Change runtime type`).
4.  Follow the instructions in the notebook cells.

See **[COLAB.md](COLAB.md)** for a detailed walkthrough.

---

## 🔬 Statistical Validation (Friedman Test)

The pipeline implements rigorous statistical testing to compare model performance:

```python
# Friedman test for comparing multiple models across CV folds
statistic, p_value = friedmanchisquare(*cv_matrix)

# Post-hoc pairwise mapping
ranks = rankdata([-m for m in mean_scores])
```

---

## � Pipeline Outputs

Upon completion, the pipeline generates rich visualizations:

- **training_results_dashboard.png**: 2x2 Dashboard (Performance, Time, CV Results, Model Types).
- **roc_curves.png**: Comparative ROC curves for all models.
- **business_impact_analysis.png**: Profit vs ROI visualization.
- **feature_importance_[Model].png**: Top predictors for the selected best model.

---

## 📄 License

MIT License
