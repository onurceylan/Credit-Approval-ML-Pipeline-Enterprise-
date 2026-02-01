# 📊 Credit Approval ML Pipeline

> **Hybrid MLOps Production Architecture** (Jupyter Notebook + Modular Python)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Colab Ready](https://img.shields.io/badge/Google_Colab-Ready-orange.svg)](COLAB.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Bu proje, kredi onayı tahminlemesi için geliştirilmiş **üretim seviyesinde (production-grade)** bir makine öğrenmesi boru hattıdır. Orijinal [V3.5 Monolitik Notebook](https://github.com/onurceylan/multimodal-credit-approval-V3.5) mimarisinden **Hibrit MLOps Mimarisine** dönüştürülmüştür.

---

## 🌟 Key Features

Orijinal projenin tüm gelişmiş özellikleri korunmuş ve modernize edilmiştir:

- 🤖 **Multi-Algorithm Training**: XGBoost, LightGBM, CatBoost, RandomForest, GradientBoosting, LogisticRegression.
- 📊 **Statistical Validation**: Modeller arası farkların anlamlılığını ölçen **Friedman Testi** ve post-hoc analizler.
- 💼 **Business Impact Analysis**: Sadece Accuracy değil, **ROI (Yatırım Getirisi)**, **NPV (Net Bugünkü Değer)** ve **Payback Period** hesaplamaları.
- 🛡️ **Bakım ve Güvenlik**: Data Leakage önlemek için "Temporal Splitting" ve "Stratified Cross-Validation".
- 🚀 **Hybrid Architecture**: Hem **Colab Notebook** (`main.ipynb`) hem de **Terminal CLI** (`main.py`) desteği.

---

## 🏗️ Architecture

```
Credit Approval ML Pipeline (Hybrid)
├── 📓 Interface Layer
│   ├── main.ipynb (Colab Entry Point)
│   └── main.py (CLI Entry Point)
├── 📦 Core Layer (src/)
│   ├── 🔧 Feature Engineering (Advanced preprocessing, categorical encoding)
│   ├── 🤖 Model Factory (GPU-accelerated training)
│   ├── 🔬 Statistical Evaluator (Friedman test, rank analysis)
│   └── 💰 Business Analyzer (Financial impact, ROI, NPV)
└── 📊 Output Layer
    ├── 📈 Plots (ROC, Confusion Matrix, Feature Importance)
    └── 📑 Reports (Business Case, Evaluation JSON)
```

---

## 🚀 Hızlı Başlangıç (Google Colab)

En kolay kullanım yolu Google Colab'dır. Detaylı rehber için **[COLAB.md](COLAB.md)** dosyasını okuyun.

1.  Projeyi Google Drive'a yükleyin.
2.  `main.ipynb` dosyasını açın.
3.  `Runtime` -> `Change runtime type` -> **T4 GPU** seçin.
4.  Hücreleri çalıştırın.

---

## 🔬 Statistical Validation (Friedman Test)

Bu pipeline, modelleri kıyaslarken sadece skora bakmaz, istatistiksel olarak anlamlı fark olup olmadığını test eder:

```python
# Pipeline otomatik olarak hesaplar:
stats, p_value = friedmanchisquare(*cv_matrix)
```

Eğer `p-value < 0.05` ise, modeller arasında şans eseri olmayan gerçek bir performans farkı olduğu kanıtlanır.

---

## 💼 Business Impact Analysis

Model başarısı finansal metriklere dökülür:

- **Net Profit**: Tahmin edilen kâr.
- **ROI %**: Yatırımın geri dönüş yüzdesi.
- **NPV (5-Year)**: 5 yıllık net bugünkü değer projeksiyonu.
- **Payback Period**: Yatırımın kendini amorti süresi.

---

## 📊 Pipeline çıktıları

Eğitim bittiğinde `ml_pipeline_output/plots` klasöründe şu grafikler oluşur:

1.  **model_comparison.png**: Tüm metriklerin kıyaslaması.
2.  **roc_curves.png**: Tüm modellerin ROC eğrileri.
3.  **confusion_matrices.png**: Hata matrisleri.
4.  **business_impact.png**: Kâr ve ROI analizi.
5.  **feature_importance.png**: En önemli öznitelikler.

---

## 🛠️ Troubleshooting

**Soru:** `ModuleNotFoundError: No module named 'src'`
**Çözüm:** `main.ipynb` içindeki `PROJECT_PATH` yolunu Drive'daki klasörünüzle eşleşecek şekilde güncelleyin.

**Soru:** `Cannot setitem on a Categorical with a new category`
**Çözüm:** Pipeline v3.1 güncellemesiyle bu sorun çözüldü (Kategorik veriler otomatik string'e çevriliyor).

---

## 📄 Lisans

MIT License