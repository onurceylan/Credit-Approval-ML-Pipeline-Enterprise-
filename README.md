multimodal-credit-approval/
│
├── data/                     # Ham veya işlenmiş veri (opsiyonel)
│   ├── raw/                  # Orijinal dataset (CSV, JSON, vb.)
│   └── processed/            # Preprocessed dataset
│
├── notebooks/                # Deneme amaçlı Jupyter Notebook'lar
│   └── exploration.ipynb
│
├── src/                      # Asıl proje kodları
│   ├── __init__.py
│   │
│   ├── config.py             # Genel ayarlar (ör. hyperparametreler, yol bilgileri)
│   │
│   ├── data_loader.py        # Veri yükleme fonksiyonları
│   │   ├── load_from_csv()
│   │   ├── load_from_bigquery()
│   │   └── preprocess_data()
│   │
│   ├── features.py           # Feature engineering işlemleri
│   │   ├── encode_categoricals()
│   │   ├── scale_features()
│   │   └── create_new_features()
│   │
│   ├── models.py             # Modellerin tanımları
│   │   ├── get_logistic_regression()
│   │   ├── get_xgboost()
│   │   └── get_neural_net()
│   │
│   ├── train.py              # Eğitim pipeline'ı
│   │   └── train_model(model, X_train, y_train)
│   │
│   ├── evaluate.py           # Performans metrikleri
│   │   ├── classification_report_custom()
│   │   ├── plot_confusion_matrix()
│   │   └── save_results()
│   │
│   └── utils.py              # Yardımcı fonksiyonlar (loglama, seed ayarı vb.)
│
├── main.py                   # Pipeline başlangıç noktası
│                             # Örn: veri yükle → preprocess → model eğit → evaluate
│
├── requirements.txt          # Bağımlılıklar (pandas, scikit-learn, xgboost, torch, vb.)
├── README.md                 # Proje açıklaması
└── .gitignore





# 📊 Credit Approval ML Pipeline

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Overview

Machine learning pipeline for credit approval prediction featuring statistical validation, comprehensive business impact analysis, and production deployment readiness. This system provides end-to-end ML workflow from data ingestion to stakeholder reporting.

---

## 🌟 Key Features

- 🤖 **Multi-Algorithm Training**: XGBoost, LightGBM, CatBoost, RandomForest, GradientBoosting, LogisticRegression
- 📊 **Statistical Validation**: Friedman test with Bonferroni-corrected post-hoc analysis
- 💼 **Business Impact Analysis**: ROI calculation, risk assessment, implementation roadmap
- 🚀 **Production Ready**: Deployment artifacts, model serving API, monitoring recommendations
- 📋 **Stakeholder Reports**: Executive summaries, technical guides, business case documentation
- 🛡️ **Data Leakage Prevention**: Temporal splitting and comprehensive validation
- ⚡ **GPU Acceleration**: CUDA support for XGBoost, LightGBM, and CatBoost
- 🔍 **Model Interpretability**: Feature importance, SHAP integration recommendations
- 📈 **Comprehensive Visualization**: 20+ business and technical dashboards

---

## 🏗️ Architecture

```
Credit Approval ML Pipeline
├── 📁 Data Layer
│   ├── Robust data loading with validation
│   ├── Temporal splitting (prevents data leakage)
│   └── Comprehensive quality checks
├── 🔧 Feature Engineering
│   ├── Safe preprocessing pipeline
│   ├── Advanced feature creation
│   └── Categorical encoding with validation
├── 🤖 Model Training
│   ├── Multi-algorithm support
│   ├── Optuna hyperparameter optimization
│   └── Cross-validation with stratification
├── 📊 Statistical Analysis
│   ├── Friedman test for model comparison
│   ├── Post-hoc pairwise testing
│   └── Effect size calculations
├── 🎯 Model Selection
│   ├── Multi-criteria decision making
│   ├── Performance vs business trade-offs
│   └── Deployment readiness assessment
├── 💼 Business Analysis
│   ├── ROI and NPV calculations
│   ├── Risk assessment and mitigation
│   └── Strategic impact analysis
└── 🚀 Deployment
    ├── Model serving API
    ├── Monitoring recommendations
    └── Stakeholder documentation
```

---

## 📂 Output Structure

```
ml_pipeline_output/
├── 📁 models/                    # Trained models and preprocessors
│   ├── XGBoost_model.joblib
│   ├── LightGBM_model.joblib
│   ├── feature_engineer.joblib
│   └── ...
├── 📁 plots/                     # Visualizations and dashboards
│   ├── training_results.png
│   ├── model_evaluation_comparison.png
│   ├── business_impact_analysis.png
│   └── model_selection_final.png
├── 📁 results/                   # Analysis reports and metrics
│   ├── data_validation_report.json
│   ├── training_summary.json
│   ├── evaluation_report.json
│   ├── executive_summary_report.txt
│   ├── business_case_document.txt
│   └── implementation_guide.txt
├── 📁 logs/                      # Execution logs
│   └── ml_pipeline_YYYYMMDD_HHMMSS.log
└── 📁 final_model/              # Deployment-ready artifacts
    ├── [ModelName]_final.joblib
    ├── preprocessor_final.joblib
    └── model_metadata.json
```

---

## 🔬 Statistical Validation

### Friedman Test Implementation

The pipeline implements rigorous statistical testing to compare model performance:

```python
# Friedman test for comparing multiple models across CV folds
statistic, p_value = friedmanchisquare(*cv_matrix)

# Post-hoc pairwise comparisons with Bonferroni correction
alpha_corrected = 0.05 / (n_models * (n_models - 1) / 2)
```

### Key Features

- **Non-parametric testing**: No assumptions about data distribution
- **Multiple comparison correction**: Bonferroni adjustment for family-wise error rate
- **Effect size calculation**: Practical significance assessment
- **Confidence intervals**: Statistical uncertainty quantification

### Statistical Output Example

```
📊 Friedman Test Results:
   • Chi-square statistic: 15.234567
   • p-value: 0.001234
   • Significant: Yes (α = 0.05)

🔍 Post-hoc pairwise comparisons:
   • Bonferroni-corrected α: 0.003333
   • XGBoost vs RandomForest: p=0.000123 *** (XGBoost better)
   • XGBoost vs LogisticRegression: p=0.000456 *** (XGBoost better)
   • 3 significant pairwise differences found
```

---

## 💼 Business Impact Analysis

### Financial Metrics

- **ROI Calculation**: Net present value with 10% discount rate
- **Payback Period**: Time to recover initial investment
- **Sensitivity Analysis**: Optimistic/realistic/pessimistic scenarios
- **Cost-Benefit Analysis**: Comprehensive cost modeling

### Business Case Components

- **Executive Summary**: C-level decision support
- **Financial Analysis**: ROI, NPV, payback calculations
- **Risk Assessment**: Financial, operational, regulatory risks
- **Implementation Roadmap**: 4-phase deployment plan
- **Success Metrics**: KPIs and monitoring framework

### Sample Business Output

```
💰 Financial Impact:
   • Annual Net Benefit: $1,234,567
   • ROI: 15.2%
   • Payback Period: Year 1
   • 5-Year NPV: $4,567,890

📊 Operational Efficiency:
   • Decision Speed: 3.2h → 0.1h (97% improvement)
   • Automated Decisions: 75% of applications
   • Processing Cost: 60-70% reduction
```

---

## 📊 Pipeline Stages

### Cell 1: Environment Setup & Configuration

Professional environment setup with dependency management, logging configuration, and GPU detection.

- Comprehensive dependency checking
- Professional logging system
- GPU acceleration detection
- Configuration validation

### Cell 2: Data Loading & Validation

Robust data loading with comprehensive validation, quality checks, and temporal integrity verification.

- Multi-path data loading with fallbacks
- Comprehensive data quality validation
- Temporal data splitting for leakage prevention
- Detailed validation reporting

### Cell 3: Data Preprocessing & Feature Engineering

Safe preprocessing pipeline with advanced feature engineering, proper fit-transform patterns, and leakage prevention.

- Safe train/validation/test splitting
- Advanced feature engineering (age groups, income ratios, employment categories)
- Categorical encoding with unseen category handling
- Outlier detection and treatment

### Cell 4: Model Training & Hyperparameter Optimization

Multi-algorithm training with Optuna optimization, GPU acceleration, and comprehensive evaluation.

- 6 different algorithms with GPU support
- Optuna hyperparameter optimization
- Cross-validation with stratification
- Performance tracking and comparison

### Cell 5: Model Evaluation & Statistical Comparison

Comprehensive model evaluation with statistical validation, Friedman tests, and business impact assessment.

- Statistical significance testing (Friedman + post-hoc)
- Business impact analysis
- Comprehensive visualizations
- Detailed comparison reports

### Cell 6: Model Selection & Final Validation

Multi-criteria model selection with deployment readiness assessment and interpretability analysis.

- Multi-criteria decision making
- Deployment readiness assessment
- Model interpretability analysis
- Final validation and recommendations

### Cell 7: Business Impact Analysis & Insights

Enterprise-grade business analysis with ROI calculations, stakeholder reports, and implementation roadmaps.

- Comprehensive financial analysis (ROI, NPV, payback)
- Risk assessment and mitigation strategies
- Stakeholder-specific reports
- Implementation roadmap and success metrics

---

## 👥 Target Audience

| Audience | Use Case |
|----------|----------|
| 🏦 **Banking & FinTech** | Optimize credit approval workflows |
| 📊 **Risk & Compliance** | Reduce default risk via robust ML validation |
| 💼 **Executives** | Business impact reports with ROI & roadmap |
| 👩‍💻 **Data Scientists** | End-to-end ML pipeline ready for deployment |

---

## 🔧 Advanced Usage

### Custom Model Integration

```python
# Add custom model to ModelFactory
class CustomModelFactory(ModelFactory):
    """
    Extended model factory with custom model support.
    
    Supports adding proprietary or specialized models
    to the comparison framework.
    """
    
    def _get_available_models(self) -> Dict[str, Dict]:
        """
        Get available model configurations including custom models.
        
        Returns:
            Dict[str, Dict]: Model configurations with parameters and search spaces
        """
        models = super()._get_available_models()
        
        # Add custom model
        models['CustomModel'] = {
            'class': YourCustomModel,
            'params': {'param1': 'value1'},
            'param_space': {'param1': (0.1, 1.0)},
            'type': 'custom'
        }
        
        return models
```

### Custom Business Metrics

```python
# Extend BusinessImpactAnalyst
class CustomBusinessAnalyst(BusinessImpactAnalyst):
    """
    Extended business analyst with industry-specific metrics.
    
    Adds domain-specific business calculations and
    industry-standard risk assessments.
    """
    
    def _calculate_industry_specific_metrics(self, model_result: Dict) -> Dict:
        """
        Calculate industry-specific business metrics.
        
        Args:
            model_result (Dict): Model evaluation results
            
        Returns:
            Dict: Industry-specific metrics and insights
        """
        # Your custom business logic here
        return custom_metrics
```

---

## ⚡ Performance

| Operation | Time |
|-----------|------|
| Friedman Test | <1 second for 6 models × 5 folds |
| Post-hoc Tests | <2 seconds for all pairwise comparisons |
| Visualization Generation | ~30-60 seconds for all plots |

---

## 🛠️ Troubleshooting

### Common Issues

#### GPU Not Detected

```bash
# Check CUDA installation
nvidia-smi

# Install GPU versions
pip install xgboost[gpu] lightgbm[gpu] catboost[gpu]

# Set GPU configuration
CONFIG.use_gpu = True
CONFIG.gpu_device_id = 0
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for better credit decisions**