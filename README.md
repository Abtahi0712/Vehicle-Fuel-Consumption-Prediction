# Vehicle Fuel Consumption Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)

A multi-class machine learning classification system that predicts a vehicle's annual fuel consumption category — **Low**, **Medium**, or **High** — using 34 years of US EPA fuel economy data.

> **Best result: KNN (k=3) — 97.9% test accuracy · F1-macro 0.98**

---

## 📋 Problem Statement

Fuel consumption is a critical factor for consumers, fleet managers, and emissions regulators. This project builds a supervised classification pipeline that labels vehicles into three consumption tiers based on their engine and design characteristics, enabling early-stage product comparisons and policy analysis without requiring physical testing.

---

## 📊 Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | [US EPA Fuel Economy Data](https://www.fueleconomy.gov/feg/download.shtml) |
| **Years covered** | 1984 – 2017 |
| **Raw records** | 38,113 rows × 81 columns |
| **After cleaning** | 36,794 rows × 54 columns |
| **After balancing** | 8,046 records (2,682 per class) |

**Target variable:** `annual_consumption_in_barrels_ft1`

| Class | Range |
|-------|-------|
| Low | < 12 barrels / year |
| Medium | 12 – 24 barrels / year |
| High | > 24 barrels / year |

---

## 🔧 Methodology

### 1. Data Cleaning
- Dropped 13 columns with > 10,000 missing values
- Removed rows with nulls across 4 critical feature columns → **36,794 rows retained**
- Dropped 7 EV-specific and irrelevant columns

### 2. Feature Encoding & Scaling
- **Label Encoding** on 9 categorical features (make, model, drive type, fuel type, etc.)
- **MinMaxScaler** normalisation applied to all numeric features

### 3. Class Balancing
- Original dataset was imbalanced — Medium class dominated
- Downsampled Medium to 2,682 records → **balanced dataset of 8,046 samples**

### 4. Train / Test Split
- **70 / 30** stratified split → 5,632 training · 2,414 test records

---

## 🤖 Results

| Model | Train Accuracy | Test Accuracy | F1-Macro |
|-------|:--------------:|:-------------:|:--------:|
| **KNN (k=3)** | 99.1% | **97.9%** | **0.98** |
| SVM | 93.8% | 92.3% | 0.92 |
| Gaussian Naive Bayes | 93.9% | 91.6% | 0.92 |

**KNN (k=3)** was selected as the final model — it achieved the highest test accuracy (97.9%) and near-perfect F1-macro (0.98) with balanced performance across all three classes.

---

## 📁 Project Structure

```
Vehicle-Fuel-Consumption-Prediction/
├── Vehicle_Fuel_Consumption_Prediction.ipynb   # Main notebook (EDA + models)
├── requirements.txt                             # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Notebook

This project was developed in **Google Colab**. Recommended workflow:

1. Upload `Vehicle_Fuel_Consumption_Prediction.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Download the dataset from [fueleconomy.gov](https://www.fueleconomy.gov/feg/download.shtml)
3. Upload the data file to your Google Drive and mount it:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Update the data path in the first cell, then **Run All**

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | KNN, SVM, Naive Bayes, MinMaxScaler, LabelEncoder, metrics |
| `matplotlib` | Plots and charts |
| `seaborn` | Statistical visualisations |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

Dataset provided by the [US Department of Energy — Fuel Economy Guide](https://www.fueleconomy.gov/feg/download.shtml).
