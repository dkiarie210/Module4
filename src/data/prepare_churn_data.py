from pathlib import Path
import json
import pandas as pd

INPUT = Path("data/raw/netflix_customer_churn.csv")
OUTPUT = Path("data/processed/netflix_churn_model_input.csv")
SUMMARY = Path("reports/data_readiness_summary.json")

REQUIRED = [
    "customer_id","age","gender","subscription_type","watch_hours",
    "last_login_days","region","device","monthly_fee","churned",
    "payment_method","number_of_profiles","avg_watch_time_per_day",
    "favorite_genre"
]
NUMERIC = ["age","watch_hours","last_login_days","monthly_fee",
           "number_of_profiles","avg_watch_time_per_day","churned"]

df = pd.read_csv(INPUT)
df.columns = df.columns.str.strip().str.lower()
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

before = len(df)
duplicates = int(df.duplicated().sum())
df = df.drop_duplicates().copy()
for c in NUMERIC:
    df[c] = pd.to_numeric(df[c], errors="coerce")

invalid_target = ~df["churned"].isin([0, 1]) & df["churned"].notna()
if invalid_target.any():
    raise ValueError("Target contains values other than 0/1")

df = df[df["churned"].notna()].copy()
df["flag_avg_watch_over_24h"] = df["avg_watch_time_per_day"].gt(24)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
SUMMARY.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

summary = {
    "input_rows": before,
    "output_rows": len(df),
    "duplicates_removed": duplicates,
    "missing_values": {k: int(v) for k, v in df[REQUIRED].isna().sum().to_dict().items()},
    "target_distribution": {str(k): int(v) for k, v in df["churned"].value_counts().to_dict().items()},
    "avg_watch_over_24h": int(df["flag_avg_watch_over_24h"].sum()),
}
SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"Saved model-ready data to {OUTPUT}")
