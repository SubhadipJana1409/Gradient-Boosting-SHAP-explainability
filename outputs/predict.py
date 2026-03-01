"""
predict.py — Use the trained GBM (Gradient Boosting) AMR Models
================================================================
Usage:
    python predict.py

Or import predict_resistance() in your own code.
================================================================
"""

import pickle
import numpy as np
import pandas as pd

with open("gbm_models.pkl",    "rb") as f: models = pickle.load(f)
with open("gbm_metadata.pkl",  "rb") as f: meta   = pickle.load(f)

FEATURE_NAMES = meta["feature_names"]
FRIENDLY      = meta["friendly_names"]
FOCUS_ABS     = meta["focus_abs"]
TOP_SPECIES   = meta["top_species"]

AB_FULLNAMES  = {
    "VAN":"Vancomycin","CAZ":"Ceftazidime",
    "CIP":"Ciprofloxacin","SXT":"Trimethoprim-sulfa",
}


def build_features(species, ward, age, gender, year=2015):
    sp_grp = species if species in TOP_SPECIES else "Other"
    row    = {f: 0.0 for f in FEATURE_NAMES}
    sp_col   = f"sp_{sp_grp}"
    ward_col = f"ward_{ward}"
    if sp_col   in row: row[sp_col]   = 1.0
    if ward_col in row: row[ward_col] = 1.0
    row["gender_M"] = 1.0 if gender == "M" else 0.0
    row["age"]  = (age  - meta["age_mean"])  / meta["age_std"]
    row["year"] = (year - meta["year_mean"]) / meta["year_std"]
    return np.array([row[f] for f in FEATURE_NAMES]).reshape(1, -1)


def predict_resistance(species, ward, age, gender, year=2015, threshold=0.5):
    X = build_features(species, ward, age, gender, year)
    results = {}
    for ab, model in models.items():
        prob = model.predict_proba(X)[0][1]
        results[ab] = {
            "probability": round(prob, 3),
            "prediction" : "R" if prob >= threshold else "S",
            "antibiotic" : AB_FULLNAMES[ab],
        }
    resistant = [AB_FULLNAMES[ab] for ab, r in results.items()
                 if r["prediction"] == "R"]
    return {"per_antibiotic": results, "resistant_to": resistant,
            "n_resistant": len(resistant)}


if __name__ == "__main__":
    print("=" * 55)
    print("GBM AMR Resistance Predictor — Day 11")
    print("=" * 55)

    cases = [
        ("B_ESCHR_COLI", "ICU",        75, "M", 2015),
        ("B_STPHY_AURS", "Clinical",   45, "F", 2012),
        ("B_KLBSL_PNMN", "ICU",        60, "F", 2016),
        ("B_STRPT_PNMN", "Outpatient", 30, "M", 2010),
    ]
    for species, ward, age, gender, year in cases:
        r = predict_resistance(species, ward, age, gender, year)
        print(f"\n{species} | {ward} | Age={age} | {gender} | {year}")
        for ab, res in r["per_antibiotic"].items():
            flag = "🔴 R" if res["prediction"] == "R" else "🟢 S"
            print(f"  {ab:5s} {flag}  (p={res['probability']:.3f})")
        print(f"  → Resistant to: {r['resistant_to']}")
