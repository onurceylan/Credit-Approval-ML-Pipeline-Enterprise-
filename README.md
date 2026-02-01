# Credit Approval ML Pipeline

> **Clean Architecture + MLOps-Ready Hybrid Ecosystem**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Colab Ready](https://img.shields.io/badge/Google_Colab-Ready-orange.svg)](COLAB.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Bu proje, kredi onayı tahminlemesi için geliştirilmiş **üretim seviyesinde (production-grade)** bir makine öğrenmesi boru hattıdır (pipeline). 

**Hibrit Mimari** kullanır:
- **Modülerlik:** Kaynak kodlar (`src/*.py`) Clean Architecture prensiplerine göre düzenlenmiştir.
- **Esneklik:** Google Colab (`main.ipynb`) veya CLI (`main.py`) üzerinden çalıştırılabilir.

---

## 📚 Dokümantasyon

- **[☁️ Google Colab Kurulum ve Kullanım Kılavuzu](COLAB.md)** 👈 *(Colab kullanıcıları buradan başlamalı)*
- **[🏗️ Mimari ve Teknik Detaylar](WALKTHROUGH.md)** *(Yakında)*

---

## 📁 Proje Yapısı

```
credit-approval/
│
├── main.ipynb                    # 📓 COLAB GİRİŞ NOKTASI (İnteraktif)
├── main.py                       # 💻 CLI GİRİŞ NOKTASI (Terminal)
├── COLAB.md                      # 📖 Colab Kullanım Kılavuzu
│
├── configs/                      # ⚙️ Konfigürasyonlar (YAML)
│   ├── base.yaml                 #    Genel ayarlar
│   ├── training.yaml             #    Model hiperparametreleri
│   └── deployment.yaml           #    İş kuralları ve limitler
│
├── src/                          # 📦 Kaynak Kodlar (Python Modülleri)
│   ├── core/                     #    ConfigLoader, Logger, Exceptions
│   ├── data/                     #    DataLoader, DataValidator
│   ├── features/                 #    FeatureEngineer, Preprocessor
│   ├── models/                   #    ModelFactory (GPU/CPU), Registry
│   ├── training/                 #    Trainer, Optuna Optimizer
│   ├── evaluation/               #    Evaluator, BusinessMetrics
│   ├── pipelines/                #    Training & Inference Pipelines
│   └── serving/                  #    ModelPredictor API Handler
│
├── scripts/                      # 🛠️ Yardımcı Scriptler
│   ├── train.py                  #    Eğitim scripti
│   └── predict.py                #    Tahmin scripti
│
├── tests/                        # 🧪 Unit Testler
├── docker/                       # 🐳 Docker Dosyaları
├── requirements.txt              # 📋 Bağımlılıklar
└── setup.py                      # 📦 Paket Kurulum Dosyası
```

---

## 🚀 Hızlı Başlangıç (Local)

Kendi bilgisayarınızda çalıştırmak için:

```bash
# 1. Projeyi klonlayın
git clone https://github.com/example/credit-approval.git
cd credit-approval

# 2. Sanal ortam oluşturun
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 4. Eğitimi başlatın
python main.py
```

---

## 📊 Özellikler

- **Multi-Environment:** Local, Colab, Kaggle ve Docker ortamlarını otomatik algılar.
- **Model Factory:** XGBoost, LightGBM, CatBoost (GPU destekli) ve Sklearn modelleri.
- **Advanced MLOps:**
  - **Experiment Tracking:** Tüm parametreler YAML ile yönetilir.
  - **Model Registry:** Modeller versiyonlanır.
  - **Logging:** Renkli ve detaylı loglama.
- **Business Focus:** Sadece Accuracy değil, ROI (Yatırım Getirisi) analizi yapar.

---

## 🧪 Testler

```bash
pytest tests/ -v
```

---

## 📄 Lisans

MIT License