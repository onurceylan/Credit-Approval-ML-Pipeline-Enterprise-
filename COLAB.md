# ☁️ Google Colab Kullanım Kılavuzu

Bu proje, Google Colab üzerinde sorunsuz çalışacak şekilde tasarlanmıştır. Aşağıdaki adımları takip ederek modeli eğitebilirsiniz.

## 🚀 Hızlı Başlangıç

1. **Projeyi Drive'a Yükleyin:** Tüm proje klasörünü `credit-approval` adıyla Google Drive'ınıza (tercihen "MyDrive" altına) yükleyin.
2. **Notebook'u Açın:** Drive içinde `main.ipynb` dosyasını bulun ve çift tıklayarak Colab ile açın.
3. **GPU Aktifleştirme:** Üst menüden `Runtime` -> `Change runtime type` seçin ve **T4 GPU**'yu seçin.
4. **Çalıştırın:** Hücreleri sırasıyla çalıştırın.

---

## 🛠️ Detaylı Adımlar

### 1. Dosya Yapısının Doğruluğu
Drive'a yüklediğiniz klasörün şu yapıda olduğundan emin olun:
```
credit-approval/
├── main.ipynb          <-- Çalıştıracağınız dosya
├── configs/
├── src/                <-- Python modülleri
├── requirements.txt
└── ...
```

### 2. Dosya Yolu Ayarı
`main.ipynb` içindeki ilk hücrede `PROJECT_PATH` değişkeni projenizin Drive'daki yoluyla eşleşmelidir:
```python
PROJECT_PATH = '/content/drive/MyDrive/credit-approval'
```

### 3. Çıktıları Yorumlama

Eğitim sonrası oluşan dosyaların anlamları:

#### 📊 Grafikler (`plots/`)
- **training_results_dashboard.png**: 2x2 Model Performans özeti.
- **business_impact_extended.png**: [YENİ] 12 Panelli Kurumsal İş Etkisi Dashboard'u (ROI, NPV, Risk, Operasyonel Hız vb.).
- **model_selection_dashboard.png**: [YENİ] 6 Panelli Model Seçim ve Hazırlık Dashboard'u.
- **roc_curves.png** & **confusion_matrices.png**: Standart model başarı grafikleri.
- **feature_importance_[Model].png**: Seçilen model için en önemli karar verici öznitelikler.

#### 📝 Raporlar (`results/`)
- **evaluation_report.json**: Tüm modellerin detaylı test metrikleri.
- **business_case.txt**: ROI, Amortisman ve Finansal senaryo analizi.
- **implementation_guide.txt**: Canlıya geçiş yol haritası ve izleme önerileri.

---

## ❓ Sık Karşılaşılan Sorunlar
- **Path hatası:** `PROJECT_PATH` değişkenini kontrol edin.
- **Import hatası:** Drive'ın doğru mount edildiğinden ve `src` klasörünün yerinde olduğundan emin olun.
