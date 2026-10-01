from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

INPUT = Path("reports/test_predictions.csv")
OUTPUT = Path("reports/sensitivity_thresholds.csv")

df = pd.read_csv(INPUT)
rows = []
for t in np.arange(0.30, 0.71, 0.05):
    pred = (df.y_prob >= t).astype(int)
    rows.append({"threshold": round(float(t),2), "accuracy": accuracy_score(df.y_true,pred), "precision": precision_score(df.y_true,pred,zero_division=0), "recall": recall_score(df.y_true,pred,zero_division=0), "f1": f1_score(df.y_true,pred,zero_division=0)})
pd.DataFrame(rows).to_csv(OUTPUT, index=False)
print(pd.DataFrame(rows).to_string(index=False))
