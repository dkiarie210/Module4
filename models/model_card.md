# Model Card — Netflix Customer Churn Model

## Purpose
Predict the dataset's binary `churned` label for academic model development and validation.

## Unit of analysis
One customer record.

## Training features
`watch_hours`, `last_login_days`, `monthly_fee`, `number_of_profiles`, `avg_watch_time_per_day`, `subscription_type`, `region`, `device`, `payment_method`, `favorite_genre`.

## Excluded from prediction
- `customer_id`: identifier only
- `age`: retained for fairness audit
- `gender`: retained for fairness audit

## Algorithms evaluated
Dummy baseline, Logistic Regression, Random Forest, Gradient Boosting.

## Validation
80/20 stratified hold-out, 5-fold stratified cross-validation, accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and ROC curve.

## Explainability
Global and local SHAP explanations are generated for the selected model.

## Fairness
Audited by gender and age group using group performance, selection rate, demographic parity difference, and equalized odds difference.

## Limitations
This public Kaggle dataset should not be represented as Netflix production customer data. The dataset does not define an explicit future churn horizon, so the task is churn-status classification rather than a validated future-event forecast.
