import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

# ——— Config ———
DATA_DIR       = "processed_data"
MODEL_PATH     = os.path.join(DATA_DIR, "xgb_tier1_pruned.pkl")
SCALER_PATH    = os.path.join(DATA_DIR, "scaler.pkl")
FEATURES_PATH  = os.path.join(DATA_DIR, "top100_features.txt")
PIPELINE_PATH  = os.path.join(DATA_DIR, "processed_datasets.pkl")
TARGET_FPR     = 0.05  # e.g. 1% false-positive rate

# ——— Load artifacts ———
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(FEATURES_PATH) as f:
    top_feats = [line.strip() for line in f]
data = joblib.load(PIPELINE_PATH)
feature_names = data["feature_names"]
X_val_raw = pd.DataFrame(data["X_val"], columns=feature_names)
y_val     = data["y_val"]

# ——— Scale validation set for pruned features ———
orig_feats = list(scaler.feature_names_in_)
means      = scaler.mean_
scales     = scaler.scale_
Xp         = X_val_raw[top_feats].fillna(0).astype(float)
indices    = [orig_feats.index(f) for f in top_feats]
pruned_means  = means[indices]
pruned_scales = scales[indices]
X_scaled  = (Xp.values - pruned_means) / pruned_scales

# ——— Get probabilities and compute ROC ——
probs = model.predict_proba(X_scaled)[:,1]
fpr, tpr, thresholds = roc_curve(y_val, probs)

# ——— Skip infinite threshold entry ——
finite_mask = np.isfinite(thresholds)
fpr, tpr, thresholds = fpr[finite_mask], tpr[finite_mask], thresholds[finite_mask]

# ——— Identify thresholds with FPR ≤ target ——
valid_idxs = np.where(fpr <= TARGET_FPR)[0]
if valid_idxs.size > 0:
    best_idx = valid_idxs[-1]
    best_thr = thresholds[best_idx]
    print(f"Threshold for ≤{TARGET_FPR*100:.1f}% FPR: {best_thr:.4f}")
    print(f"    => at this cutoff: TPR (recall) = {tpr[best_idx]:.4f}, FPR = {fpr[best_idx]:.4f}")
else:
    # No finite threshold meets the FPR, choose the max threshold that gives any reduction
    thr_candidate = thresholds[np.argmin(fpr)]
    print(f"No threshold ≤ {TARGET_FPR*100:.1f}% FPR found among finite values;")
    print(f"using threshold {thr_candidate:.4f}, which gives FPR = {fpr.min():.4f}, TPR = {tpr[np.argmin(fpr)]:.4f}")