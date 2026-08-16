"""
KrishiSahayak — Model Development & Validation
================================================
Trains a baseline (Logistic Regression) and main model (XGBoost) to classify
district-year agricultural risk into Low / Medium / High, with full
validation and interpretation as required by the assignment.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, classification_report)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "krishisahayak_dataset.csv")

FEATURES = [
    "area_1000ha", "rainfall_mm", "avg_temp_c", "rainfall_deviation_pct",
    "prev_year_yield", "prior_3yr_avg_yield", "prior_3yr_yield_volatility",
]
TARGET = "risk_level"

X = df[FEATURES].copy()
y_raw = df[TARGET].copy()

le = LabelEncoder()
le.fit(["Low", "Medium", "High"])
y = le.transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

# ---------------- Baseline: Logistic Regression ----------------
baseline = LogisticRegression(max_iter=1000)
baseline.fit(X_train_scaled, y_train)
y_pred_base = baseline.predict(X_test_scaled)
y_proba_base = baseline.predict_proba(X_test_scaled)

results["Logistic Regression (baseline)"] = dict(
    accuracy=accuracy_score(y_test, y_pred_base),
    precision=precision_score(y_test, y_pred_base, average="macro"),
    recall=recall_score(y_test, y_pred_base, average="macro"),
    f1=f1_score(y_test, y_pred_base, average="macro"),
    roc_auc=roc_auc_score(y_test, y_proba_base, multi_class="ovr"),
)

# ---------------- Main model: XGBoost ----------------
xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8, objective="multi:softprob",
    num_class=3, eval_metric="mlogloss", random_state=42,
)
xgb_model.fit(X_train, y_train)  # tree models don't need scaling
y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)

results["XGBoost (main model)"] = dict(
    accuracy=accuracy_score(y_test, y_pred_xgb),
    precision=precision_score(y_test, y_pred_xgb, average="macro"),
    recall=recall_score(y_test, y_pred_xgb, average="macro"),
    f1=f1_score(y_test, y_pred_xgb, average="macro"),
    roc_auc=roc_auc_score(y_test, y_proba_xgb, multi_class="ovr"),
)

# ---------------- Cross validation (5-fold, stratified) ----------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_xgb = cross_val_score(xgb_model, X, y, cv=cv, scoring="f1_macro")
cv_scores_base = cross_val_score(baseline, StandardScaler().fit_transform(X), y, cv=cv, scoring="f1_macro")

print("=" * 60)
print("MODEL COMPARISON (held-out test set)")
print("=" * 60)
for name, m in results.items():
    print(f"\n{name}")
    for k, v in m.items():
        print(f"  {k:10s}: {v:.4f}")

print("\n" + "=" * 60)
print("5-FOLD CROSS VALIDATION (F1-macro)")
print("=" * 60)
print(f"Logistic Regression : {cv_scores_base.mean():.4f} (+/- {cv_scores_base.std():.4f})")
print(f"XGBoost              : {cv_scores_xgb.mean():.4f} (+/- {cv_scores_xgb.std():.4f})")

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT — XGBoost")
print("=" * 60)
print(classification_report(y_test, y_pred_xgb, target_names=le.classes_))

# ---------------- Confusion Matrix plot ----------------
cm = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("XGBoost — Confusion Matrix")
plt.tight_layout()
plt.savefig(BASE_DIR / "confusion_matrix.png", dpi=150)
plt.close()

# ---------------- Feature Importance (interpretation) ----------------
importances = pd.Series(xgb_model.feature_importances_, index=FEATURES).sort_values()
plt.figure(figsize=(7, 6))
importances.plot(kind="barh", color="#2E7D32")
plt.title("XGBoost Feature Importance — What Drives Agricultural Risk?")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(BASE_DIR / "feature_importance.png", dpi=150)
plt.close()

print("\nTop 5 predictors of agricultural risk:")
print(importances.sort_values(ascending=False).head(5))

# ---------------- Save artifacts for the dashboard ----------------
joblib.dump(xgb_model, BASE_DIR / "xgb_model.pkl")
joblib.dump(le, BASE_DIR / "label_encoder.pkl")
joblib.dump(FEATURES, BASE_DIR / "feature_list.pkl")

print("\nSaved: xgb_model.pkl, label_encoder.pkl, feature_list.pkl, confusion_matrix.png, feature_importance.png")
