from __future__ import annotations

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

DATA = Path("data/processed/netflix_churn_model_input.csv")
MODEL = Path("models/churn_model.joblib")
OUTPUT = Path("reports/probability_calibration_results.csv")
FIGURE = Path("reports/figures/calibration_curve.png")
CAL_MODEL = Path("models/churn_model_calibrated.joblib")
TARGET = "churned"
FEATURES = ["watch_hours","last_login_days","monthly_fee","number_of_profiles","avg_watch_time_per_day","subscription_type","region","device","payment_method","favorite_genre"]

def evaluate(name, model, X, y):
    prob = model.predict_proba(X)[:,1]
    pred = (prob >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y,pred),
        "precision": precision_score(y,pred,zero_division=0),
        "recall": recall_score(y,pred,zero_division=0),
        "f1": f1_score(y,pred,zero_division=0),
        "roc_auc": roc_auc_score(y,prob),
        "brier_score": brier_score_loss(y,prob),
        "log_loss": log_loss(y,prob),
    }, prob

def main():
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,stratify=y,random_state=42)

    original = joblib.load(MODEL)
    calibrated = CalibratedClassifierCV(estimator=clone(original), method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)

    original_result, original_prob = evaluate("original_uncalibrated", original, X_test, y_test)
    calibrated_result, calibrated_prob = evaluate("sigmoid_calibrated", calibrated, X_test, y_test)

    out = pd.DataFrame([original_result, calibrated_result])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    CAL_MODEL.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    joblib.dump(calibrated, CAL_MODEL)

    f1o, mpo = calibration_curve(y_test, original_prob, n_bins=10, strategy="quantile")
    f1c, mpc = calibration_curve(y_test, calibrated_prob, n_bins=10, strategy="quantile")

    plt.figure(figsize=(7,6))
    plt.plot([0,1],[0,1],"--",label="Perfect calibration")
    plt.plot(mpo,f1o,marker="o",label="Original Gradient Boosting")
    plt.plot(mpc,f1c,marker="o",label="Sigmoid calibrated")
    plt.xlabel("Mean predicted churn probability")
    plt.ylabel("Observed churn rate")
    plt.title("Probability Calibration — Netflix Churn Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE,dpi=180)
    plt.close()

    print(out.to_string(index=False))
    print(f"Saved metrics: {OUTPUT}")
    print(f"Saved curve: {FIGURE}")
    print(f"Saved calibrated model: {CAL_MODEL}")

if __name__ == "__main__":
    main()
