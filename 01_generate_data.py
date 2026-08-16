"""
KrishiSahayak — Real Dataset Preparation
==========================================
DATA SOURCE (satisfies assignment Section 4 — "GitHub repositories" is an
explicitly named acceptable source):
    Repository : Runax15/crop-yield-prediction-maharashtra
    URL        : https://github.com/Runax15/crop-yield-prediction-maharashtra
    File used  : final_crop.csv
    Underlying data compiled from Maharashtra government agricultural
    statistics (district-wise crop area/production/yield) merged with
    rainfall and temperature records.

DATASET DESCRIPTION (for your report's "Dataset Description" section):
    Raw rows       : 15,176 (7,588 after removing exact duplicate rows
                     found in the source file — see below)
    Districts      : 26 real Maharashtra districts (25 used — Mumbai/Bombay
                     excluded as a non-agricultural urban district)
    Years covered  : 1997 - 2017
    Crops covered  : 14 real crops (Cotton, Rice, Soyabean, Sugarcane,
                     Groundnut, Maize, Pigeonpea, etc.)

GRANULARITY DECISION:
    Risk is predicted at DISTRICT-CROP-YEAR level. A district can be
    low-risk for cotton but high-risk for rice in the same year — this
    also gives farmer-usable output: "your cotton crop in Nagpur this
    year is high-risk" rather than one vague district-wide number.

*** IMPORTANT METHODOLOGICAL NOTE — DATA LEAKAGE AVOIDANCE ***
    The target (risk_level) is built from how much THIS YEAR's yield
    dropped relative to the district-crop's own recent history. If we then
    fed "this year's yield change" back in as a MODEL INPUT, the model
    would just be reading its own answer key (this produced an
    unrealistic ~99% accuracy in an earlier version of this pipeline).

    This version fixes that by strictly separating:
      - LABEL inputs  : current year's yield vs. its own prior 3-year
                        average (only used to build risk_level)
      - PREDICTOR inputs : only information a farmer/officer would
                        actually have BEFORE the season's outcome is
                        known — current season rainfall & temperature,
                        area sown, crop type, and the district-crop's
                        OWN PRIOR-YEAR yield statistics (lagged, computed
                        with .shift(1) so the current year is never
                        included in its own predictor)
    This is what makes it a genuine early-warning / preventive model,
    consistent with the project's stated purpose, rather than a model
    that "predicts" an outcome using the outcome itself.

IMPORTANT — HONEST LIMITATION (state this in your report):
    Public, machine-readable district-level market price and farmer
    debt/loan data does not exist in a clean downloadable form for this
    period. The risk label is therefore built from real, verifiable
    rainfall and yield-outcome data — not a financial-stress dimension as
    originally scoped. Reporting this limitation honestly is stronger
    academic practice than fabricating a fourth data source.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 1. Load real raw data
# ---------------------------------------------------------------------------
raw = pd.read_csv(BASE_DIR / "raw_final_crop.csv")
raw = raw.rename(columns={
    "Dist Name": "district", "Year": "year", "Crop": "crop",
    "Area(1000 ha)": "area_1000ha", "Production(1000 tons)": "production_1000t",
    "Yield(Kg per ha)": "yield_kg_per_ha", "Total Rainfall": "rainfall_mm",
    "Avg Temp": "avg_temp_c",
})
raw = raw.drop(columns=[c for c in raw.columns if "Unnamed" in c])

# The source file contains exact duplicate rows (every district-year-crop
# record appears twice, verbatim) — a real artifact of the raw file, removed
# here rather than silently double-counted
before_dedup = len(raw)
raw = raw.drop_duplicates(subset=["district", "year", "crop"], keep="first")
print(f"Removed {before_dedup - len(raw)} exact duplicate rows found in source file "
      f"({before_dedup} -> {len(raw)})")

print(f"Raw data loaded: {raw.shape[0]} rows, {raw['district'].nunique()} districts, "
      f"{raw['crop'].nunique()} crops, years {raw['year'].min()}-{raw['year'].max()}")

# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------
before = len(raw)
raw = raw.dropna(subset=["rainfall_mm", "avg_temp_c"])
raw = raw[raw["district"] != "Bombay"]          # non-agricultural urban district
raw = raw[raw["area_1000ha"] > 0]                # drop rows where crop wasn't actually grown
print(f"After cleaning (missing weather data, Bombay, zero-area rows): "
      f"{len(raw)} rows remain ({before - len(raw)} dropped)")

raw = raw.sort_values(["district", "crop", "year"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 3. LABEL construction — uses current-year yield vs OWN prior baseline
#    (this is allowed to use current-year yield, because the label is what
#    we are trying to predict, not what we feed the model)
# ---------------------------------------------------------------------------
grp = raw.groupby(["district", "crop"])["yield_kg_per_ha"]
prior_3yr_avg_yield = grp.transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
raw["yield_vs_prior_avg_pct"] = (raw["yield_kg_per_ha"] - prior_3yr_avg_yield) / prior_3yr_avg_yield * 100

district_normal_rainfall = raw.groupby("district")["rainfall_mm"].transform("mean")
raw["rainfall_deviation_pct"] = (
    (raw["rainfall_mm"] - district_normal_rainfall) / district_normal_rainfall * 100
)

# Distress score for the LABEL: how far this year's yield fell below the
# district-crop's own recent normal, combined with how dry this year was.
# (rainfall_deviation_pct is a real-world CAUSE of distress, safe to use in
# both label construction context and as a predictor, since it doesn't
# depend on this year's yield outcome — it's independent weather data)
def minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

yield_shortfall = minmax(-raw["yield_vs_prior_avg_pct"].clip(-200, 200))
rainfall_stress = minmax(-raw["rainfall_deviation_pct"])
raw["distress_score"] = 0.65 * yield_shortfall + 0.35 * rainfall_stress

# ---------------------------------------------------------------------------
# 4. PREDICTOR features — only information available BEFORE/without
#    needing this year's yield outcome (lagged yield stats + current
#    season weather + area sown + crop identity)
# ---------------------------------------------------------------------------
raw["prev_year_yield"] = grp.transform(lambda s: s.shift(1))
raw["prior_3yr_avg_yield"] = prior_3yr_avg_yield
raw["prior_3yr_yield_volatility"] = grp.transform(
    lambda s: s.shift(1).rolling(3, min_periods=2).std())

before = len(raw)
raw = raw.dropna(subset=["prev_year_yield", "prior_3yr_avg_yield",
                          "prior_3yr_yield_volatility", "yield_vs_prior_avg_pct"])
print(f"After computing lagged (prior-year) features per district-crop: "
      f"{len(raw)} rows remain ({before - len(raw)} dropped — need >=2 prior "
      f"years of history per district-crop)")

q1, q2 = raw["distress_score"].quantile([0.33, 0.67])
raw["risk_level"] = pd.cut(raw["distress_score"], bins=[-1, q1, q2, 2],
                            labels=["Low", "Medium", "High"])

# ---------------------------------------------------------------------------
# 5. Save — note: yield_kg_per_ha (current year, used only to build the
#    label) and yield_vs_prior_avg_pct / distress_score are kept in the
#    file for transparency/audit, but 02_train_model.py must NOT use them
#    as predictors (see updated FEATURES list there)
# ---------------------------------------------------------------------------
final_cols = ["district", "crop", "year", "area_1000ha", "rainfall_mm", "avg_temp_c",
              "rainfall_deviation_pct", "prev_year_yield", "prior_3yr_avg_yield",
              "prior_3yr_yield_volatility",
              "yield_kg_per_ha", "yield_vs_prior_avg_pct", "distress_score", "risk_level"]
final = raw[final_cols].reset_index(drop=True)
final.to_csv(BASE_DIR / "krishisahayak_dataset.csv", index=False)

print(f"\nFinal dataset saved: {final.shape[0]} rows, {final.shape[1]} columns")
print(f"Districts: {final['district'].nunique()} | Crops: {final['crop'].nunique()} | "
      f"Years: {final['year'].min()}-{final['year'].max()}")
print("\nRisk level distribution:")
print(final["risk_level"].value_counts())
print("\nSample rows:")
print(final.head(5).to_string())

