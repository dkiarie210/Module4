from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

MODEL = Path("models/churn_model.joblib")
DATA = Path("data/processed/netflix_churn_model_input.csv")
FIGS = Path("reports/figures")
NUMERIC = ["watch_hours","last_login_days","monthly_fee","number_of_profiles","avg_watch_time_per_day"]
CATEGORICAL = ["subscription_type","region","device","payment_method","favorite_genre"]


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    pipe = joblib.load(MODEL)
    df = pd.read_csv(DATA)
    X = df[NUMERIC + CATEGORICAL].sample(n=min(200, len(df)), random_state=42)
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["model"]
    Xt = prep.transform(X)
    names = prep.get_feature_names_out()
    Xt = pd.DataFrame(Xt, columns=names, index=X.index)

    if hasattr(clf, "feature_importances_"):
        explainer = shap.TreeExplainer(clf)
        sv = explainer(Xt)
    elif clf.__class__.__name__ == "LogisticRegression":
        explainer = shap.LinearExplainer(clf, Xt)
        sv = explainer(Xt)
    else:
        bg = Xt.iloc[:min(50, len(Xt))]
        explainer = shap.Explainer(clf.predict_proba, bg)
        sv = explainer(Xt)

    if getattr(sv.values, "ndim", 0) == 3:
        sv = sv[:, :, 1]

    shap.summary_plot(sv, Xt, show=False, max_display=15)
    plt.tight_layout(); plt.savefig(FIGS / "shap_summary.png", dpi=180, bbox_inches="tight"); plt.close()
    shap.plots.waterfall(sv[0], max_display=12, show=False)
    plt.tight_layout(); plt.savefig(FIGS / "shap_local.png", dpi=180, bbox_inches="tight"); plt.close()
    print("Saved SHAP summary and local explanation.")


if __name__ == "__main__":
    main()
