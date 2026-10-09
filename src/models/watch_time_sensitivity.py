from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("data/processed/netflix_churn_model_input.csv")
MODEL = Path("models/churn_model.joblib")
OUTPUT = Path("reports/watch_time_treatment_sensitivity.csv")
TARGET = "churned"

NUMERIC = ["watch_hours","last_login_days","monthly_fee","number_of_profiles","avg_watch_time_per_day"]
CATEGORICAL = ["subscription_type","region","device","payment_method","favorite_genre"]
FEATURES = NUMERIC + CATEGORICAL

def make_pipeline(params):
    num = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    prep = ColumnTransformer([("num", num, NUMERIC),("cat", cat, CATEGORICAL)])
    return Pipeline([("preprocess", prep),("model", GradientBoostingClassifier(**params))])

def score(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:,1]
    return {
        "accuracy": accuracy_score(y,pred),
        "precision": precision_score(y,pred,zero_division=0),
        "recall": recall_score(y,pred,zero_division=0),
        "f1": f1_score(y,pred,zero_division=0),
        "roc_auc": roc_auc_score(y,prob),
    }

def main():
    df = pd.read_csv(DATA)
    saved = joblib.load(MODEL)
    params = saved.named_steps["model"].get_params()
    y = df[TARGET].astype(int)
    train_idx, test_idx = train_test_split(df.index, test_size=0.20, stratify=y, random_state=42)

    capped = df.copy()
    capped["avg_watch_time_per_day"] = capped["avg_watch_time_per_day"].clip(upper=24)

    scenarios = {
        "original": df.copy(),
        "exclude_over_24h": df[df["avg_watch_time_per_day"] <= 24].copy(),
        "cap_at_24h": capped,
    }

    rows = []
    for name, work in scenarios.items():
        tr = train_idx.intersection(work.index)
        te = test_idx.intersection(work.index)
        model = make_pipeline(params)
        model.fit(work.loc[tr, FEATURES], work.loc[tr, TARGET].astype(int))
        result = score(model, work.loc[te, FEATURES], work.loc[te, TARGET].astype(int))
        rows.append({
            "scenario": name,
            "train_rows": len(tr),
            "test_rows": len(te),
            "flagged_rows_remaining": int((work["avg_watch_time_per_day"] > 24).sum()),
            **result
        })

    out = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    print(out.to_string(index=False))
    print(f"Saved: {OUTPUT}")

if __name__ == "__main__":
    main()
