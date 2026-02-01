# Credit Approval ML Pipeline

> **Clean Architecture + MLOps-Ready Hybrid Structure**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Colab Ready](https://img.shields.io/badge/Main-Notebook-orange.svg)](main.ipynb)

---

## 🏗️ Architecture Overview

Bu proje **Hibrit Yapı** (Hybrid Structure) kullanır:
1. **Modüler Python Dosyaları (`src/*.py`):** MLOps, test edilebilirlik ve düzen için.
2. **Jupyter Notebook (`main.ipynb`):** Google Colab ve interaktif deneyler için.

Bu sayede **"Import" sorunları yaşamazsınız** hem de notebook rahatlığını kullanırsınız.

```
credit-approval/
├── main.ipynb                    # 📓 COLAB GİRİŞ NOKTASI
├── main.py                       # 💻 CLI GİRİŞ NOKTASI
├── configs/                      # 📋 YAML Konfigürasyonlar
├── src/                          # 📦 Modüler Kaynak Kod (Python)
│   ├── core/                     #    Config, Logger
│   ├── data/                     #    Loader, Validator
│   ├── features/                 #    Feature Engineering
│   ├── models/                   #    Model Factory
│   ├── pipelines/                #    Training/Inference Pipelines
│   └── ...
├── scripts/                      # 🛠️ Yardımcı Scriptler (.py)
├── tests/                        # 🧪 Testler (.py)
└── requirements.txt
```

---

## 🚀 Google Colab'da Nasıl Çalıştırılır?

1. **Projeyi Drive'a Yükleyin:** Tüm klasörü Google Drive'ınıza yükleyin.
2. **Setup:** `main.ipynb` dosyasını Colab ile açın.
3. **Drive Bağlantısı:** İlk hücredeki `PROJECT_PATH` değişkenini projenizin olduğu yol ile güncelleyin (örn: `/content/drive/MyDrive/credit-approval`).
4. **Çalıştırın:** Notebook hücrelerini sırasıyla çalıştırın.

---

## 💻 Local Kurulum

```bash
# Kurulum
pip install -r requirements.txt

# Çalıştırma (Python)
python main.py

# Çalıştırma (Notebook)
jupyter notebook main.ipynb
```