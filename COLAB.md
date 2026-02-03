# ☁️ Google Colab User Guide

This project is designed to run seamlessly on Google Colab. Follow the steps below to train your model and analyze results.

## 🚀 Quick Start

1.  **Upload Project to Drive:** Upload the entire `credit-approval` project folder to your Google Drive (preferably under "MyDrive").
2.  **Open the Notebook:** Locate `main.ipynb` within your Drive and double-click to open it with Google Colab.
3.  **Activate GPU:** From the top menu, select `Runtime` -> `Change runtime type` and choose **T4 GPU**.
4.  **Run:** Execute the cells sequentially.

---

## 🛠️ Detailed Steps

### 1. Verification of File Structure
Ensure that the folder uploaded to Drive follows this structure:
```
credit-approval/
├── main.ipynb          <-- The entry point
├── configs/
├── src/                <-- Python modules
├── requirements.txt
└── ...
```

### 2. Path Configuration
In the first cell of `main.ipynb`, the `PROJECT_PATH` variable must match your project's path on Drive:
```python
PROJECT_PATH = '/content/drive/MyDrive/credit-approval'
```

### 3. Interpreting Outputs

After execution, follow these generated artifacts for insights:

#### 📊 Dashboards (`plots/`)
- **training_results_dashboard.png**: 2x2 Model Performance summary.
- **business_impact_extended.png**: [NEW] 12-Panel Enterprise Business Impact Dashboard (ROI, NPV, Risk, Ops Speed, etc.).
- **model_selection_dashboard.png**: [NEW] 6-Panel Model Selection and Readiness Dashboard.
- **roc_curves.png** & **confusion_matrices.png**: Standard model performance visualizations.
- **feature_importance_[Model].png**: Top decision-making features for the selected model.

#### 📝 Reports (`results/`)
- **evaluation_report.json**: Detailed test metrics for all algorithms.
- **business_case.txt**: ROI, Amortization, and Financial Scenario analysis.
- **implementation_guide.txt**: Roadmap for production deployment and monitoring.

---

## ❓ Troubleshooting
- **Path Error:** Check the `PROJECT_PATH` variable and Drive folder name.
- **Import Error:** Ensure Drive is mounted correctly and the `src` folder is present in the working directory.
