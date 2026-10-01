from pathlib import Path
import pandas as pd
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference, selection_rate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

INPUT = Path("reports/test_predictions.csv")
OUTPUT = Path("reports/fairness_metrics.csv")


def audit(df, sensitive):
    frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "precision": lambda y, p: precision_score(y, p, zero_division=0),
            "recall": lambda y, p: recall_score(y, p, zero_division=0),
            "f1": lambda y, p: f1_score(y, p, zero_division=0),
            "selection_rate": selection_rate,
        },
        y_true=df.y_true,
        y_pred=df.y_pred,
        sensitive_features=df[sensitive],
    )
    out = frame.by_group.reset_index()
    out.insert(0, "protected_attribute", sensitive)
    out["demographic_parity_difference_overall"] = demographic_parity_difference(df.y_true, df.y_pred, sensitive_features=df[sensitive])
    out["equalized_odds_difference_overall"] = equalized_odds_difference(df.y_true, df.y_pred, sensitive_features=df[sensitive])
    return out


def main():
    df = pd.read_csv(INPUT)
    df["age_group"] = pd.cut(df.age, bins=[17,29,44,59,float("inf")], labels=["18-29","30-44","45-59","60+"], include_lowest=True).astype(str)
    result = pd.concat([audit(df, "gender"), audit(df, "age_group")], ignore_index=True)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
