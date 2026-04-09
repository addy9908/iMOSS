# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone

# ===============================
# 1. Load data
# ===============================
excel_file = r"C:\Users\yez4\Box\NIDA works\Projects\Manuscript for tail suspension system\202500328\ML_threshold\immobility_threshold_data.xlsx"
sheet_name = "Sheet1"
file_path = os.path.dirname(excel_file)
output_path = os.path.join(file_path,'Output')

df = pd.read_excel(excel_file, sheet_name=sheet_name)

# Features and target
X = df[["Threshold", "Auto_Time"]]
y = df["Manual_Time"]
groups = df["Mouse"]

# ===============================
# 2. Train-test split by mouse
# ===============================
mice = df["Mouse"].unique()
train_mice, test_mice = train_test_split(mice, test_size=0.25, random_state=42)

train_df = df[df["Mouse"].isin(train_mice)].copy()
test_df  = df[df["Mouse"].isin(test_mice)].copy()

X_train = train_df[["Threshold", "Auto_Time"]]
y_train = train_df["Manual_Time"]
g_train = train_df["Mouse"]

X_test = test_df[["Threshold", "Auto_Time"]]
y_test = test_df["Manual_Time"]

print(f"Train mice: {len(train_mice)}, Test mice: {len(test_mice)}")
print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

# ===============================
# 3. Baseline model
# ===============================
baseline_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
baseline_model.fit(X_train, y_train)

baseline_test_pred = baseline_model.predict(X_test)
baseline_test_mae = mean_absolute_error(y_test, baseline_test_pred)

# Grouped CV on training mice only
n_splits = min(5, len(np.unique(g_train)))
if n_splits < 2:
    raise ValueError("Not enough training mice for cross-validation.")

gkf = GroupKFold(n_splits=n_splits)
baseline_cv_maes = []

for tr_idx, val_idx in gkf.split(X_train, y_train, groups=g_train):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    m = clone(baseline_model)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_val)
    baseline_cv_maes.append(mean_absolute_error(y_val, pred))

baseline_cv_mae = float(np.mean(baseline_cv_maes))

print(f"Baseline RF | CV MAE (train only): {baseline_cv_mae:.3f} | Test MAE: {baseline_test_mae:.3f}")

# ===============================
# 4. Tuned model
#    Tune ONLY on training mice
# ===============================
param_dist = {
    "n_estimators": [100, 150, 200, 250, 300, 400],
    "max_depth": [None, 4, 6, 8, 10, 12, 16, 20],
    "min_samples_split": [2, 4, 6, 8, 10],
    "min_samples_leaf": [1, 2, 3, 4, 5],
    "max_features": ["sqrt", "log2", 1.0, 0.7, 0.5],
    "bootstrap": [True],
    "max_samples": [0.7, 0.8, 0.9, 1.0],
}

search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=25,
    scoring="neg_mean_absolute_error",
    cv=gkf,
    random_state=42,
    n_jobs=-1,
    refit=True
)

search.fit(X_train, y_train, groups=g_train)

tuned_model = search.best_estimator_
tuned_cv_mae = -search.best_score_

tuned_test_pred = tuned_model.predict(X_test)
tuned_test_mae = mean_absolute_error(y_test, tuned_test_pred)

print(f"Tuned RF    | CV MAE (train only): {tuned_cv_mae:.3f} | Test MAE: {tuned_test_mae:.3f}")
print("Best tuned params:", search.best_params_)

# ===============================
# 5. Choose final model using CV only
#    Do NOT use test MAE for selection
# ===============================
if tuned_cv_mae < baseline_cv_mae:
    best_model = tuned_model
    best_name = "Tuned RF"
    best_cv_mae = tuned_cv_mae
else:
    best_model = baseline_model
    best_name = "Baseline RF"
    best_cv_mae = baseline_cv_mae

print(f"Selected model by training CV: {best_name} (CV MAE = {best_cv_mae:.3f})")

# ===============================
# 6. Final test evaluation once
# ===============================
final_pred = best_model.predict(X_test)
final_test_mae = mean_absolute_error(y_test, final_pred)

print(f"Final model: {best_name} | Final test MAE: {final_test_mae:.3f}")

# Optional: save chosen model
# import joblib
# joblib.dump(best_model, os.path.join(os.path.dirname(excel_file), "best_model.joblib"))

# ===============================
# 7. Individual best threshold
# ===============================
search_space = np.linspace(df['Threshold'].min(), df['Threshold'].max(), 1000)
results = []
for mouse in df['Mouse'].unique():
    sub = df[df['Mouse'] == mouse]
    manual = sub['Manual_Time'].iloc[0]
    preds = []
    for t in search_space:
        auto_est = np.interp(t, sub['Threshold'], sub['Auto_Time'])
        X_new = pd.DataFrame([[t, auto_est]], columns=['Threshold', 'Auto_Time'])
        pred = best_model.predict(X_new)[0]
        preds.append(pred)
    preds = np.array(preds)
    best_idx = np.argmin(np.abs(preds - manual))
    best_thresh = search_space[best_idx]
    best_pred = preds[best_idx]
    results.append([mouse, manual, best_thresh, best_pred])

results_df = pd.DataFrame(results, columns=['Mouse', 'Manual', 'Best_Threshold', 'Predicted_Time'])
print("\nIndividual best thresholds:")
print(results_df)

# ===============================
# 8. Global best threshold
# ===============================
global_errors = []
for t in search_space:
    preds, manuals = [], []
    for mouse in df['Mouse'].unique():
        sub = df[df['Mouse'] == mouse]
        manual = sub['Manual_Time'].iloc[0]
        auto_est = np.interp(t, sub['Threshold'], sub['Auto_Time'])
        X_new = pd.DataFrame([[t, auto_est]], columns=['Threshold', 'Auto_Time'])
        pred = best_model.predict(X_new)[0]
        preds.append(pred)
        manuals.append(manual)
    error = mean_absolute_error(manuals, preds)
    global_errors.append(error)

best_idx = np.argmin(global_errors)
global_best_thresh = search_space[best_idx]
print(f"\nGlobal best threshold: {global_best_thresh:.3f}, MAE={global_errors[best_idx]:.3f}")

# save results_df as csv
csv_name = 'ML_result.csv'
csv_path = os.path.join(output_path, csv_name)
results_df.to_csv(csv_path, index=False)