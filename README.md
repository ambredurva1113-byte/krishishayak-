---

# 🌾 KrishiSahayak — Agricultural Risk Prediction System
**Crop Distress Early-Warning System for Maharashtra Districts**

Built for farmers and agricultural officers who need to know — which district, which crop, which year is heading toward distress — before the season ends badly.

---

## Problem Statement
Maharashtra farmers face crop failures every year but get warnings too late.
KrishiSahayak predicts it early —

**Rainfall + Temperature + Yield History → Distress Score → Risk Level → Action**

---

## Modules

| Module | Description |
|---|---|
| Risk Label Engine | Builds Low/Medium/High label from real yield shortfall + rainfall deviation |
| ML Predictor | XGBoost classifier — predicts crop distress per district per year |
| Baseline Comparison | Logistic Regression vs XGBoost — honest accuracy reporting |
| Bilingual Dashboard | English/Marathi Streamlit UI with district map |

---

## Model Results

| Model | Accuracy | F1 (Macro) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 57.5% | 0.570 | 0.761 |
| XGBoost | 63.9% | 0.638 | 0.816 |

---

## Dataset
- Real Maharashtra government agricultural data
- 25 districts · 14 crops · 1999–2017
- 4,684 model-ready rows
- Source: Maharashtra district-wise crop + rainfall records

---

## Tech Stack
- **Language:** Python 3.12
- **Dashboard:** Streamlit (English + Marathi)
- **ML:** XGBoost · Scikit-learn
- **Charts:** Plotly
- **Data:** Pandas + CSV

---

## Setup
```bash
pip install streamlit xgboost scikit-learn pandas numpy plotly joblib
python 01_generate_data.py
python 02_train_model.py
streamlit run app.py
```

---

## Folder Structure
```
KrishiSahayak/
├── raw_final_crop.csv          ← real source dataset
├── krishisahayak_dataset.csv   ← cleaned model-ready data
├── 01_generate_data.py         ← cleaning + feature engineering
├── 02_train_model.py           ← model training + evaluation
├── reference_data.py           ← district coordinates for map
├── app.py                      ← Streamlit dashboard
├── xgb_model.pkl               ← saved XGBoost model
└── README.md
```

---

**Developed by Durva Sagar Ambre · TY Project · 2026**

---
