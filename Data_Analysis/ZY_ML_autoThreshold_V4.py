# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 16:15:26 2025
v4: keep re-tuning with progressively more regularized search spaces until (a) the overfitting gap is acceptable or (b) 5 minutes have elapsed—whichever comes first.

use ML to decide the detection threshold for auto detection based on total_immobility_time

immobility_threshold_data.xlsx: Mouse, Threshold, Auto_Time, Manual_Time
test_size=0.25, random_state=42, kFold = 5

Threshold Optimization Using Machine Learning

To determine the optimal immobility detection threshold for automated scoring, we employed a machine learning–based prediction strategy. For each mouse, automated total immobility time was computed across a range of candidate thresholds, while manual immobility time was independently quantified by blinded observers. The resulting dataset consisted of paired values of threshold, automated immobility time, and the corresponding manual immobility time.

We trained a regression model (Random Forest Regressor, scikit-learn v1.5.1) to learn the relationship between threshold–automated immobility measures and manual immobility times. Model performance was evaluated using mean absolute error (MAE), which quantifies the average absolute difference between predicted and manual values. Both internal validation (5-fold cross-validation) and external validation (hold-out mice not seen during training) were performed to assess robustness and generalizability.

To identify the individual best threshold for each mouse, we searched across a dense range of threshold values (interpolated between the tested thresholds). At each candidate threshold, the model predicted automated immobility time, and the threshold yielding the smallest absolute difference from the manual reference was selected. To identify a global best threshold applicable across all mice, we minimized the average MAE between predicted automated times and manual times over the entire cohort.

This framework provides both per-animal optimal thresholds and a population-level optimal threshold, thereby enabling flexible comparison of automated and manual scoring and identifying parameter settings that best approximate manual annotations.

@author: yez4
"""
import os, time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
from scipy.stats import sem
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
# from copy import deepcopy
import random, math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional #Tuple, 

# ===============================
# 0. Set plot fontsize
# ===============================
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 10 #14
mpl.rcParams['axes.titlesize'] = 16 
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
plt.close('all')

# ===============================
# 1. Load data
# ===============================
# change this full file name to your own file
excel_file = r"C:\Users\yez4\Box\NIDA works\Projects\Manuscript for tail suspension system\202500328\ML_threshold\immobility_threshold_data.xlsx"
sheet_name = 'Sheet1'

file_path = os.path.dirname(excel_file)
output_path = os.path.join(file_path,'Output')
os.makedirs(output_path, exist_ok=True)

df = pd.read_excel(excel_file, sheet_name=sheet_name)

print(df.head())  # preview first few rows

# Features and target
X = df[['Threshold', 'Auto_Time']]
y = df['Manual_Time']

# ===============================
# 2. Train-test split by mouse (external validation)
# ===============================
mice = df['Mouse'].unique()
train_mice, test_mice = train_test_split(mice, test_size=0.25, random_state=42) #12 mice for training and 4 for testing

train_df = df[df['Mouse'].isin(train_mice)]
test_df = df[df['Mouse'].isin(test_mice)]

X_train, y_train = train_df[['Threshold','Auto_Time']], train_df['Manual_Time']
X_test, y_test   = test_df[['Threshold','Auto_Time']], test_df['Manual_Time']

# ===============================
# 3.1. Baseline RF
# ===============================
baseline_model = RandomForestRegressor(n_estimators=200, random_state=42)
baseline_model.fit(X_train, y_train)

y_pred_baseline = baseline_model.predict(X_test)
baseline_external_mae = mean_absolute_error(y_test, y_pred_baseline)

# Internal CV MAE
kf = KFold(n_splits=5, shuffle=True, random_state=42)
baseline_internal_maes = []
for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    m = clone(baseline_model)  # fresh copy
    m.fit(X_tr, y_tr)
    baseline_internal_maes.append(mean_absolute_error(y_val, m.predict(X_val)))
baseline_internal_mae_avg = np.mean(baseline_internal_maes)

print(f"Baseline RF: External MAE={baseline_external_mae:.3f}, Internal MAE={baseline_internal_mae_avg:.3f}")


# ===============================
# 3.2. Hyperparameter tuning with RandomizedSearchCV
# ===============================
'''
We performed hyperparameter tuning using RandomizedSearchCV with restricted parameter ranges. 
However, the tuned models did not improve upon the baseline Random Forest (200 estimators, 
default parameters), which already achieved the lowest validation error and minimal overfitting. 
Therefore, we report results from the baseline model as the final model.
'''
# code for re-tuning
# ===========================================
# Utilities
# ===========================================

def _unique_sorted_ints(values, min_val=None, max_val=None):
    vals = sorted(set(int(v) for v in values))
    if min_val is not None:
        vals = [max(min_val, v) for v in vals]
    if max_val is not None:
        vals = [min(max_val, v) for v in vals]
    return sorted(set(vals))

def _scale_int_list(base_list, scale, *, lo=None, hi=None):
    """Scale a list of ints by 'scale' and clamp to [lo, hi] if provided."""
    scaled = [int(round(v * scale)) for v in base_list]
    return _unique_sorted_ints(scaled, lo, hi)

def _expand_int_range(center_vals: List[int], widen: int, lo: int, hi: int):
    """Given some center values, expand around them up to 'widen' and clamp."""
    expanded = set(center_vals)
    for v in center_vals:
        for d in range(1, widen + 1):
            expanded.add(v + d)
            expanded.add(v - d)
    expanded = [v for v in expanded if lo <= v <= hi]
    return sorted(set(expanded))

def _choose_max_features(scale: float):
    """
    For RF, higher scale → more features per split; lower scale → fewer.
    Return a diverse set to let the search pick.
    """
    if scale >= 1.15:
        return [1.0, 'sqrt', 'log2', 0.7]  # allow full, plus common heuristics
    elif scale <= 0.85:
        return ['sqrt', 'log2', 0.7, 0.5, 0.3]
    else:
        return ['sqrt', 'log2', 1.0, 0.5]

def _choose_max_samples(scale: float):
    """Higher scale → closer to full dataset; lower scale → more subsampling."""
    if scale >= 1.15:
        return [0.9, 1.0]
    elif scale <= 0.85:
        return [0.5, 0.6, 0.7, 0.8]
    else:
        return [0.7, 0.8, 0.9]

@dataclass
class SearchResult:
    model: Any
    best_params: Dict[str, Any]
    internal_mae_avg: float
    external_mae: float
    overfit_gap: float
    is_overfit: bool
    tier_scale: float
    n_iter_used: int
    elapsed_sec: float

# ===========================================
# One pass of tuning on a given param_dist
# ===========================================

def run_search_once(
    X_train, y_train, X_test, y_test, X_full, y_full,
    param_dist: Dict[str, Any],
    n_iter=20, cv=5, rnd=42, n_jobs=-1,
    overfit_tol=0.10
) -> SearchResult:
    t0 = time.time()
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=rnd),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='neg_mean_absolute_error',
        cv=cv,
        random_state=rnd,
        n_jobs=n_jobs
    )
    search.fit(X_train, y_train)
    tuned_model = search.best_estimator_

    # External (holdout) score
    y_pred_tuned = tuned_model.predict(X_test)
    external_mae = mean_absolute_error(y_test, y_pred_tuned)

    # Internal CV score (retrain per fold on full train set)
    kf = KFold(n_splits=cv, shuffle=True, random_state=rnd)
    internal_maes = []
    for tr_idx, val_idx in kf.split(X_full):
        X_tr, X_val = X_full.iloc[tr_idx], X_full.iloc[val_idx]
        y_tr, y_val = y_full.iloc[tr_idx], y_full.iloc[val_idx]
        model_fold = clone(tuned_model)
        model_fold.fit(X_tr, y_tr)
        y_val_pred = model_fold.predict(X_val)
        internal_maes.append(mean_absolute_error(y_val, y_val_pred))
    internal_mae_avg = float(np.mean(internal_maes))

    gap = external_mae - internal_mae_avg
    is_overfit = gap > overfit_tol * internal_mae_avg

    return SearchResult(
        model=tuned_model,
        best_params=search.best_params_,
        internal_mae_avg=internal_mae_avg,
        external_mae=external_mae,
        overfit_gap=gap,
        is_overfit=is_overfit,
        tier_scale=1.0,   # filled by caller
        n_iter_used=n_iter,
        elapsed_sec=time.time() - t0
    )

# ===========================================
# Dynamic param distribution generator
# ===========================================

def generate_param_dist(base: Dict[str, Any], scale: float) -> Dict[str, Any]:
    """
    scale > 1  → allow more complexity (deeper trees, smaller leaves/splits, more features/samples/estimators)
    scale < 1  → regularize (shallower, bigger leaves/splits, fewer features/samples/estimators)
    """
    # --- n_estimators ---
    base_estimators = base.get('n_estimators', [150, 200, 250])
    # scale and also widen around scaled values a bit for diversity
    scaled_estimators = _scale_int_list(base_estimators, scale, lo=60, hi=600)
    scaled_estimators = _expand_int_range(scaled_estimators, widen=20, lo=60, hi=600)

    # --- max_depth ---
    # Always include some finite depths; include None only when exploring more complexity
    base_depths = [d for d in base.get('max_depth', [None, 8, 12, 16])]
    finite_depths = [d for d in base_depths if isinstance(d, int)]
    finite_scaled = _scale_int_list(finite_depths, scale, lo=3, hi=40)
    finite_scaled = _expand_int_range(finite_scaled, widen=2, lo=3, hi=40)
    if scale >= 1.10 and (None not in base_depths):
        depth_candidates = sorted(set([*finite_scaled, 20, 24, 28])) + [None]
    elif scale >= 1.10:
        depth_candidates = sorted(set([*finite_scaled])) + [None]
    else:
        # when regularizing, drop None to force shallower trees
        depth_candidates = sorted(set(finite_scaled))

    # --- min_samples_split / leaf ---
    base_split = base.get('min_samples_split', [2, 4, 6, 8])
    base_leaf = base.get('min_samples_leaf', [1, 2, 3, 4, 5])

    # For higher complexity: *smaller* splits/leaves; for regularization: *larger*
    split_scaled = _scale_int_list(base_split, 1.0 / max(scale, 1e-9), lo=2, hi=64)
    leaf_scaled  = _scale_int_list(base_leaf,  1.0 / max(scale, 1e-9), lo=1, hi=64)
    # Widen a bit for diversity
    split_scaled = _expand_int_range(split_scaled, widen=2, lo=2, hi=64)
    leaf_scaled  = _expand_int_range(leaf_scaled,  widen=2, lo=1, hi=64)

    # --- max_features & max_samples ---
    max_features = _choose_max_features(scale)
    max_samples  = _choose_max_samples(scale)

    param_dist = {
        'n_estimators': sorted(set(scaled_estimators)),
        'max_depth': depth_candidates,
        'min_samples_split': sorted(set(split_scaled)),
        'min_samples_leaf': sorted(set(leaf_scaled)),
        'max_features': max_features,
        'bootstrap': [True],
        'max_samples': max_samples,
    }
    return param_dist

# ===========================================
# Adaptive controller
# ===========================================
def adaptive_rf_search_collect(
    X_train, y_train, X_test, y_test, X_full, y_full,
    *,
    max_minutes: int = 5, # run out time in min
    cv: int = 5,
    rnd: int = 42,  # master seed
    n_jobs: int = -1,
    overfit_tol: float = 0.10,
    # define "lowfitting" (uniformly weak) relative to running best:
    underfit_tol_ext: float = 0.01,   # 1% worse external than best so far
    underfit_tol_int: float = 0.01,   # 1% worse internal than best so far
    gap_small_frac: float  = 0.03,    # |gap| <= 3% of internal → "small"
    # scaling dynamics:
    shrink_factor: float = 0.85,
    expand_factor: float = 1.15,
    base_param_dist: Optional[Dict[str, Any]] = None,
    initial_scale: float = 1.0,
    base_n_iter: int = 20,
    # ✅ NEW: baseline gate for eligibility
    baseline_external_mae: Optional[float] = None
) -> Dict[str, Any]:
    """
    Run for the full time budget, collect all models that are:
      - not overfit
      - not lowfit (uniformly weak)
      - (if provided) external_mae <= baseline_external_mae
    and return the best eligible by external MAE at the end.
    If none eligible, fall back to global best external MAE (even if above baseline),
    so you still get a model back and can inspect history.
    """
    # ✅ Seed all RNGs for reproducibility
    random.seed(rnd)
    np.random.seed(rnd)
    
    if base_param_dist is None:
        base_param_dist = {
            'n_estimators': [150, 200, 250],
            'max_depth': [None, 8, 12, 16],
            'min_samples_split': [2, 4, 6, 8],
            'min_samples_leaf': [1, 2, 3, 4, 5],
            'max_features': ['sqrt', 'log2', 1.0, 0.5],
            'bootstrap': [True],
            'max_samples': [0.7, 0.8, 0.9, 1.0],
        }

    start = time.time()
    scale = float(initial_scale)

    history: List[Dict[str, Any]] = []
    eligible: List[Dict[str, Any]] = []

    best_ext = math.inf
    best_int = math.inf
    best_global = None  # track overall best by external MAE, for fallback

    round_idx = 0

    while True:
        elapsed = time.time() - start
        if elapsed >= max_minutes * 60:
            print("⏱️ Time budget reached.")
            break

        round_idx += 1
        time_ratio = elapsed / (max_minutes * 60)
        n_iter = max(5, int(base_n_iter * (1.0 - 0.4 * time_ratio)))

        param_dist = generate_param_dist(base_param_dist, scale)
        result = run_search_once(
            X_train, y_train, X_test, y_test, X_full, y_full,
            param_dist=param_dist,
            n_iter=n_iter, cv=cv, rnd=rnd, n_jobs=n_jobs,
            overfit_tol=overfit_tol
        )
        result.scale_used = scale
        result.n_iter_used = n_iter

        # Update running best (for underfit detection and fallback)
        if result.external_mae < best_ext:
            best_ext = result.external_mae
            best_int = result.internal_mae_avg
            best_global = {
                "model": result.model,
                "external_mae": result.external_mae,
                "internal_mae": result.internal_mae_avg,
                "params": result.best_params,
                "scale": scale,
                "n_iter": n_iter,
            }

        gap = result.overfit_gap
        is_overfit = result.is_overfit
        is_gap_small = abs(gap) <= gap_small_frac * result.internal_mae_avg

        # Underfit (lowfitting) heuristic: uniformly weak & consistent
        is_underfit = (
            (result.external_mae >= best_ext * (1 + underfit_tol_ext)) and
            (result.internal_mae_avg >= best_int * (1 + underfit_tol_int)) and
            is_gap_small
        )

        # ✅ NEW: baseline gate
        passes_baseline = (baseline_external_mae is None) or (result.external_mae <= baseline_external_mae)

        # Record round
        status = (
            "OVERFIT" if is_overfit else
            "LOWFIT" if is_underfit else
            ("ABOVE_BASE" if not passes_baseline else "ELIGIBLE")
        )
        
        rec = {
            "round": round_idx,
            "scale": scale,
            "n_iter": n_iter,
            "best_params": result.best_params,
            "internal_mae": result.internal_mae_avg,
            "external_mae": result.external_mae,
            "gap": gap,
            "is_overfit": is_overfit,
            "is_underfit": is_underfit,
            "passes_baseline": passes_baseline,
            "status": status,
            "elapsed_round_sec": result.elapsed_sec,
            "elapsed_total_sec": elapsed + result.elapsed_sec
        }
        history.append(rec)

        print(
            f"[Round {round_idx}] scale={scale:.3f} | n_iter={n_iter} | "
            f"Int={result.internal_mae_avg:.4f} | Ext={result.external_mae:.4f} | "
            f"Gap={gap:.4f} | {status}"
        )

        # Collect eligible models (neither overfit nor underfit)
        if (not is_overfit) and (not is_underfit):
            eligible.append({
                "model": result.model,
                "external_mae": result.external_mae,
                "internal_mae": result.internal_mae_avg,
                "params": result.best_params,
                "scale": scale,
                "n_iter": n_iter,
            })

        # Adapt scale for next round (no early stopping)
        if is_overfit:
            scale *= (shrink_factor + random.uniform(-0.03, 0.03))
        elif is_underfit or (not passes_baseline and baseline_external_mae is not None):
            scale *= (expand_factor + random.uniform(-0.03, 0.03))
        else:
            # eligible (good-ish): explore locally
            scale *= (1.0 + random.uniform(-0.02, 0.02))

        scale = float(np.clip(scale, 0.55, 1.8))

    # Final selection
    if eligible:
        final = min(eligible, key=lambda e: e["external_mae"])
        selection_reason = "best among eligible (not overfit or lowfit)"
    else:
        final = best_global
        selection_reason = "fallback to global best external MAE (no eligible models found)"

    return {
        "final_model": final["model"] if final else None,
        "final_params": final["params"] if final else None,
        "final_external_mae": final["external_mae"] if final else None,
        "final_internal_mae": final["internal_mae"] if final else None,
        "selection_reason": selection_reason,
        "eligible_count": len(eligible),
        "history": history,
        "eligible_models": eligible,  # full list if you want to inspect later
        "best_overall": best_global
    }

# ===========================================
# Hyperparameter tuning with RandomizedSearchCV on the data
# ===========================================
result = adaptive_rf_search_collect(
    X_train, y_train, X_test, y_test, X, y,
    max_minutes=5, # time out in min
    cv=5,
    rnd=42,
    n_jobs=-1,
    overfit_tol=0.10,
    underfit_tol_ext=0.01,
    underfit_tol_int=0.01,
    gap_small_frac=0.03,
    shrink_factor=0.85,
    expand_factor=1.15,
    base_param_dist=None,
    initial_scale=1.0,
    base_n_iter=20,
    baseline_external_mae = baseline_external_mae  # ✅ only keep candidates at or below this
)
print("Chosen:", result["final_external_mae"], result["final_params"], result["selection_reason"])
tuned_model = result["final_model"]
tuned_external_mae = result["final_external_mae"]
tuned_internal_mae_avg = result["final_internal_mae"]

if tuned_external_mae - tuned_internal_mae_avg > 0.1 * tuned_internal_mae_avg:  # e.g., >10% difference
    print("⚠️ Warning: Possible overfitting detected for tuned_model (>10% difference)! Consider reducing model complexity.")

    
# ===============================
# 3.3. Decide which model to use as best_model
# ===============================
if tuned_external_mae < baseline_external_mae:
    best_model = tuned_model
    internal_mae_avg = tuned_internal_mae_avg
    external_mae = tuned_external_mae
    print(">>>Decision: Using Tuned RF as best_model")
else:
    best_model = baseline_model
    internal_mae_avg = baseline_internal_mae_avg
    external_mae = baseline_external_mae
    print(">>>Decision: Using Baseline RF as best_model")

# Overfitting check
if external_mae - internal_mae_avg > 0.1 * internal_mae_avg:  # >10% difference
    print("⚠️ Warning: Possible overfitting detected for final best_model (>10% difference)! Consider reducing model complexity.")
else:
    print("🍻 Cheers: No overfitting detected for best_model (<10% difference)!")    
    
# ===============================
# 4. Plot the Learning curve for best_model
# ===============================
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X, y, cv=5, scoring='neg_mean_absolute_error',
    train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
)

train_scores_mean = -np.mean(train_scores, axis=1)
val_scores_mean   = -np.mean(val_scores, axis=1)

fig0 = plt.figure(figsize=(3,3))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training MAE')
plt.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Validation MAE')
plt.axhline(internal_mae_avg, color='green', linestyle='--', label='K-Fold CV Internal MAE avg')
plt.axhline(external_mae, color='magenta', linestyle='--', label='K-Fold CV External MAE')
plt.xlabel('Training size')
plt.ylabel('MAE (s)')
plt.title('Learning Curve for Best Model', fontsize = 12)
plt.legend(framealpha=0, edgecolor='none',fontsize = 10)
plt.grid(True)
plt.tight_layout(pad=0)
plt.show()

fig0_name = 'learning curve1'
fig0.savefig(os.path.join(output_path, fig0_name +".pdf")) 
fig0.savefig(os.path.join(output_path, fig0_name +".png"), dpi=600) 

# ===============================
# 5. Individual best threshold
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
# 6. Global best threshold
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

'''
Result:
# test_size = 0.3: Global best threshold: 0.792 with MAE = 4.352
    External validation MAE: 15.686644083575034
    Internal validation MAE (avg): 16.128910043346856
    Each fold MAE: [20.574084814635995, 12.021464924677423, 12.843420452034943, 14.176111260179306, 21.029468765206616]
    
# test_size = 0.25: Global best threshold: 0.792 with MAE=4.115
    External MAE=14.346, Internal MAE=16.129

# test_size = 0.5: Global best threshold: 0.792, MAE=6.612
    External validation MAE: 16.27770587832504
    Internal validation MAE (avg): 16.128910043346856
    Each fold MAE: [20.574084814635995, 12.021464924677423, 12.843420452034943, 14.176111260179306, 21.029468765206616]
'''

# ===============================
# 7. 4-panel plots for publication
# ===============================
search_thresholds = df['Threshold'].unique()  # my 5 tested thresholds

from zy_preset_mpl_v2 import preset_mpl
preset_mpl()

# Panel A – scatter plots for each threshold
fig1, axes = plt.subplots(1, len(search_thresholds), figsize=(6, 2))
for i, thresh in enumerate(search_thresholds):
    ax = axes[i]
    ax.set_aspect('equal', adjustable='box')  # square axes
    sub = df[df['Threshold'] == thresh]
    num_points = len(sub)
    # Use a colormap (tab20 has 20 distinct colors)
    cmap = cm.get_cmap('tab20', num_points)  # 'tab20' or 'tab10', 'tab20b', etc.
    
    # Generate colors for each point
    colors = [cmap(i) for i in range(num_points)]
    
    ax.scatter(
        sub['Manual_Time'],
        sub['Auto_Time'],
        color='tab:blue',
        marker='o',          # circle
        facecolors='none',   # open (no fill)
        edgecolors=colors  # outline color
    )
    ax.plot([sub['Manual_Time'].min(), sub['Manual_Time'].max()],
            [sub['Manual_Time'].min(), sub['Manual_Time'].max()], 'r--')  # unity line
    if i ==2:
        ax.set_xlabel('Manual Time (s)')
    else:
        ax.set_xlabel('')
    if i==0:
        ax.set_ylabel('Auto Time (s)')
    else:
        ax.set_ylabel('')
        ax.tick_params(labelleft=False)   # remove tick labels
        
    ax.set_title(thresh)
plt.suptitle('Manual vs Automated Scoring at Candidate Thresholds', fontsize=12, fontweight='bold')
plt.tight_layout(pad = 0)
plt.show()

fig1_name = 'Manual vs Automated Scoring at Candidate Thresholds2'
fig1.savefig(os.path.join(output_path, fig1_name +".pdf"))          # Save as PDF (vector graphics)
fig1.savefig(os.path.join(output_path, fig1_name +".png"), dpi=600) 

# Panel B – individual optimal thresholds
fig2, ax = plt.subplots(figsize=(3,3))
ax.scatter(
    results_df['Mouse'], 
    results_df['Best_Threshold'], 
    color='tab:green', 
    s=100,
    marker='o',          # circle
    facecolors='none',   # open (no fill)
    edgecolors='tab:green'  # outline color
    )
ax.axhline(global_best_thresh, color='r', linestyle='--', linewidth=2, label=f'Global Best Threshold: {round(global_best_thresh,3)}')
ax.set_xlabel('Mouse ID')
ax.set_ylabel('Individual Best Threshold')
ax.set_title('')
ax.text(
    x=8,          # x-coordinate in data units (adjust as needed)
    y=global_best_thresh - 0.05, # slightly above 0.792
    s=f'Global Best Threshold: {round(global_best_thresh,3)}',
    ha='center',    # horizontal alignment
    va='bottom',    # vertical alignment
    fontsize=10,
    color='red'
)
plt.xticks(results_df['Mouse'], rotation=45, fontsize = 9, fontweight='bold')
plt.tight_layout(pad=0)
plt.show()

fig2_name = 'Individual Optimal Thresholds1'
fig2.savefig(os.path.join(output_path, fig2_name +".pdf")) 
fig2.savefig(os.path.join(output_path, fig2_name +".png"), dpi=600) 

# Panel C – histogram/violin of individual thresholds with global threshold
fig3, ax = plt.subplots(figsize=(3,3))
ax.hist(results_df['Best_Threshold'], bins=6, color='skyblue', edgecolor='k')
ax.axvline(global_best_thresh, color='r', linestyle='--', linewidth=2, label=f'Global Best Threshold: {round(global_best_thresh,3)}')
ax.set_xlabel('Individual Best Thresholds')
ax.set_ylabel('Count')
ax.set_ylim([0,6])
ax.set_title('') #Distribution of Individual Best Thresholds
legend = ax.legend(framealpha=0, edgecolor='none',fontsize = 10)
for text in legend.get_texts():
    text.set_color('red')
plt.tight_layout(pad=0)
plt.show()

fig3_name = 'Distribution of Individual Best Thresholds1'
fig3.savefig(os.path.join(output_path, fig3_name +".pdf"))                     # Save as PDF (vector graphics)
fig3.savefig(os.path.join(output_path, fig3_name +".png"), dpi=600) 

# Panel D – MAE comparison
mae_list = []
mae_sem = []  # store standard deviation for error bars
labels = []

# 5 candidate thresholds
for t in search_thresholds:
    maes = []
    for mouse in df['Mouse'].unique():
        sub = df[df['Mouse'] == mouse]
        manual = sub['Manual_Time'].iloc[0]
        auto_est = np.interp(t, sub['Threshold'], sub['Auto_Time'])
        pred = auto_est
        maes.append(abs(pred - manual))
    maes = np.array(maes)
    mae_list.append(np.mean(maes))
    mae_sem.append(sem(maes))
    labels.append(f'{t}')

# # Global best threshold
# maes_global = []
# for mouse in df['Mouse'].unique():
#     sub = df[df['Mouse'] == mouse]
#     manual = sub['Manual_Time'].iloc[0]
#     auto_est = np.interp(global_best_thresh, sub['Threshold'], sub['Auto_Time'])
#     maes_global.append(abs(auto_est - manual))
# maes_global = np.array(maes_global)
# mae_list.append(np.mean(maes_global))
# mae_sem.append(sem(maes_global))
# labels.append('GBT-predict')

# real maes from iMOSS-AS with GBT
real_maes_GBT = []

manuals = np.array([193.1813,
                    189.442,
                    36.19745,
                    79.8144,
                    181.7588,
                    101.4415,
                    24.30337,
                    4.766667,
                    150.8548,
                    11.76667,
                    94.0518,
                    0,
                    74.11023,
                    136.6209,
                    35.63934,
                    0])
iMoss = np.array([193.0054,
                187.9954,
                41.27221,
                88.55018,
                187.4886,
                103.5955,
                28.08837,
                4.637338,
                154.5085,
                14.64473,
                104.4113,
                0,
                74.41151,
                135.3559,
                43.81317,
                0])
real_maes_GBT = abs(iMoss - manuals)
real_maes_GBT = np.array(real_maes_GBT)
mae_list.append(np.mean(real_maes_GBT))
mae_sem.append(sem(real_maes_GBT))
labels.append('GBT')

# save the outputs:
df_mae = pd.DataFrame({
    'Label': labels,
    'MAE': mae_list,
    'SEM': mae_sem
})
df_mae.to_csv(os.path.join(output_path,'mae_results.csv'), index=False)
    
# # Individual optimal thresholds
# maes_indiv = np.abs(results_df['Predicted_Time'] - results_df['Manual'])
# mae_list.append(np.mean(maes_indiv))
# mae_sem.append(sem(maes_indiv))
# labels.append('Individual Best')

fig4, ax = plt.subplots(figsize=(3,3))
bars = ax.bar(labels, mae_list, color='lightcoral', edgecolor='k')

# Draw top-only error bars
for i, bar in enumerate(bars):
    ax.errorbar(
        bar.get_x() + bar.get_width()/2,
        mae_list[i],
        yerr=[[0], [mae_sem[i]]],  # lower=0, upper=sem
        fmt='none',
        ecolor='k',
        capsize=5
    )


ax.set_xticklabels(labels, rotation=0, ha='center',fontweight='bold')  # ha='right' aligns text nicely
ax.set_ylabel('MAE (s)')
ax.set_xlabel('Thresholds')
ax.set_ylim([0,45])
ax.set_title('') #MAE Comparison Across Methods

# Add numeric labels above top of error bars
for i, bar in enumerate(bars):
    height = mae_list[i] + mae_sem[i]  # place above error bar
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, f'{mae_list[i]:.2f}',
            ha='center', va='bottom', fontsize=10)
plt.tight_layout(pad=0)
plt.show()

fig4_name = 'MAE Comparison Across threshold in iMOSS'
fig4.savefig(os.path.join(output_path, fig4_name +".pdf"))                     # Save as PDF (vector graphics)
fig4.savefig(os.path.join(output_path, fig4_name +".png"), dpi=600) 

# save results_df as csv
csv_name = 'ML_result.csv'
csv_path = os.path.join(output_path, csv_name)
results_df.to_csv(csv_path, index=False)