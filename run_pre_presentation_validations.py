from __future__ import annotations
import subprocess, sys

steps = [
    ("Feature ablation / leakage stress test", "src/models/feature_ablation.py"),
    ("Watch-time anomaly sensitivity", "src/models/watch_time_sensitivity.py"),
    ("Probability calibration", "src/models/probability_calibration.py"),
]

for label, script in steps:
    print("\n" + "="*80)
    print(label)
    print("="*80)
    subprocess.run([sys.executable, script], check=True)

print("\nAll pre-presentation validations completed.")
