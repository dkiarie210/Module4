from pathlib import Path
import kagglehub
import pandas as pd

DATASET = "abdulwadood11220/netflix-customer-churn-dataset"
OUT = Path("data/raw/netflix_customer_churn.csv")

EXPECTED = {
    "customer_id","age","gender","subscription_type","watch_hours",
    "last_login_days","region","device","monthly_fee","churned",
    "payment_method","number_of_profiles","avg_watch_time_per_day",
    "favorite_genre"
}

path = Path(kagglehub.dataset_download(DATASET))
csvs = list(path.rglob("*.csv"))
if not csvs:
    raise FileNotFoundError(f"No CSV found under {path}")

src = next((p for p in csvs if p.name.lower() == "netflix_customer_churn.csv"), csvs[0])
df = pd.read_csv(src)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
missing = EXPECTED - set(df.columns)
if missing:
    raise ValueError(f"Missing expected columns: {sorted(missing)}")

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"Saved {len(df):,} rows x {len(df.columns)} columns to {OUT}")
