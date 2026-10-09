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
OUTPUT = Path("reports/feature_ablation_results.csv")
TARGET = "churned"

NUMERIC = ["watch_hours","last_login_days","monthly_fee","number_of_profiles","avg_watch_time_per_day"]
CATEGORICAL = ["subscription_type","region","device","payment_method","favorite_genre"]

ABLATIONS = {
    "all_features": [],
    "without_avg_watch_time_per_day": ["avg_watch_time_per_day"],
    "without_last_login_days": ["last_login_days"],
    "without_watch_hours": ["watch_hours"],
    "without_top_3_behavior_features": ["avg_watch_time_per_day","last_login_days","watch_hours"],
}

def preprocessor(numeric, categorical):
    num = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, numeric),("cat", cat, categorical)])

def main():
    df = pd.read_csv(DATA)
    saved = joblib.load(MODEL)
    params = saved.named_steps["model"].get_params()
    y = df[TARGET].astype(int)
    train_idx, test_idx = train_test_split(df.index, test_size=0.20, stratify=y, random_state=42)

    rows = []
    for name, removed in ABLATIONS.items():
        nums = [c for c in NUMERIC if c not in removed]
        cats = [c for c in CATEGORICAL if c not in removed]
        features = nums + cats

        pipe = Pipeline([
            ("preprocess", preprocessor(nums, cats)),
            ("model", GradientBoostingClassifier(**params)),
        ])
        pipe.fit(df.loc[train_idx, features], y.loc[train_idx])
        pred = pipe.predict(df.loc[test_idx, features])
        prob = pipe.predict_proba(df.loc[test_idx, features])[:,1]

        rows.append({
            "scenario": name,
            "removed_features": ", ".join(removed) if removed else "None",
            "accuracy": accuracy_score(y.loc[test_idx], pred),
            "precision": precision_score(y.loc[test_idx], pred, zero_division=0),
            "recall": recall_score(y.loc[test_idx], pred, zero_division=0),
            "f1": f1_score(y.loc[test_idx], pred, zero_division=0),
            "roc_auc": roc_auc_score(y.loc[test_idx], prob),
        })

    out = pd.DataFrame(rows)
    base_auc = out.loc[out["scenario"]=="all_features","roc_auc"].iloc[0]
    out["roc_auc_drop_vs_all_features"] = base_auc - out["roc_auc"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    print(out.to_string(index=False))
    print(f"Saved: {OUTPUT}")

if __name__ == "__main__":
    main()
