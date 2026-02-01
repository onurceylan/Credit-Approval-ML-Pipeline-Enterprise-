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
# Eğer klasörü doğrudan MyDrive içine attıysanız bu yol doğrudur:
PROJECT_PATH = '/content/drive/MyDrive/credit-approval'
```

### 3. Drive Bağlantısı (Mount)
Notebook'u çalıştırdığınızda Google Drive'a erişim izni isteyecektir. "Connect to Google Drive" butonuna tıklayıp izin verin.

> **Not:** Script `force_remount=True` kullandığı için bağlantı koparsa otomatik tekrar dener.

### 4. Eğitim Süreci
Notebook sırasıyla şunları yapar:
1.  Gerekli kütüphaneleri (`requirements.txt`) kurar.
2.  Python modüllerini (`src/`) içe aktarır.
3.  Veriyi yükler, temizler ve özellik mühendisliği yapar.
4.  Seçilen modelleri (XGBoost, LightGBM vb.) Optuna ile optimize eder ve eğitir.
5.  Sonuçları `ml_pipeline_output/` klasörüne kaydeder.

### 5. Sonuçları Görüntüleme
Eğitim bittikten sonra Drive'ınızdaki `ml_pipeline_output` klasöründe şunları bulacaksınız:
- `models/`: Kaydedilmiş modeller (.joblib)
- `plots/`: Başarı grafikleri (.png)
- `results/`: Detaylı raporlar (.json)

### 6. Çıktıları Yorumlama

Eğitim sonrası oluşan dosyaların anlamları:

#### 📊 Grafikler (`plots/`)
- **model_comparison.png**: Hangi modelin daha başarılı olduğunu gösterir (Accuracy, AUC).
- **business_impact.png**: Modellerin finansal etkisini (Net Kâr ve ROI) kıyaslar. En yüksek ROI'ye sahip model iş açısından en iyisidir.
- **roc_curves.png**: Eğri sol üst köşeye ne kadar yakınsa model o kadar iyidir.
- **feature_importance.png**: Modelin hangi müşteri özelliklerine (Gelir, Yaş vb.) daha çok önem verdiğini gösterir.

#### 📝 Raporlar (`results/`)
- **Friedman Test**: Modeller arası farkın "şans eseri" olup olmadığını söyler.
- **Business Case**: "Bu modeli kullanırsak yılda X dolar kâr ederiz" şeklindeki yönetici özetidir.

---

## ❓ Sık Karşılaşılan Sorunlar

**Soru:** `ModuleNotFoundError: No module named 'src'` hatası alıyorum.
**Çözüm:** `PROJECT_PATH` değişkeninin doğru olduğundan emin olun. Klasör ismini değiştirdiyseniz kodda da güncelleyin.

**Soru:** Eğitim çok yavaş.
**Çözüm:** `Runtime` -> `Change runtime type` menüsünden **GPU** seçili olduğundan emin olun.

**Soru:** `Drive Mount` hatası alıyorum.
**Çözüm:** Sol menüdeki "Dosyalar" simgesine tıklayıp `drive` klasörünün orada olup olmadığını kontrol edin. Gerekirse "Mount Drive" butonuna manuel basın.
