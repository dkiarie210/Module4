import subprocess
import sys

STEPS = [
    ("Ingest Kaggle churn dataset", "src/data/ingest_churn.py"),
    ("Prepare and validate model data", "src/data/prepare_churn_data.py"),
    ("Train/tune/evaluate models with MLflow", "src/models/train.py"),
    ("Run fairness analysis", "src/models/fairness.py"),
    ("Generate SHAP explanations", "src/models/explain.py"),
    ("Run sensitivity analysis", "src/models/sensitivity.py"),
]

for label, script in STEPS:
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    subprocess.run([sys.executable, script], check=True)

print("\nPipeline complete.")
print("Model: models/churn_model.joblib")
print("Metrics: reports/model_comparison.csv")
print("Fairness: reports/fairness_metrics.csv")
print("SHAP: reports/figures/shap_summary.png")
