from pathlib import Path
import pandas as pd

REQUIRED = {"customer_id","age","gender","subscription_type","watch_hours","last_login_days","region","device","monthly_fee","churned","payment_method","number_of_profiles","avg_watch_time_per_day","favorite_genre"}

def test_processed_dataset_contract():
    path = Path("data/processed/netflix_churn_model_input.csv")
    assert path.exists(), "Run the preparation pipeline first."
    df = pd.read_csv(path)
    assert REQUIRED.issubset(df.columns)
    assert len(df) >= 5000
    assert set(df["churned"].dropna().unique()).issubset({0, 1})
    assert df["customer_id"].notna().all()
