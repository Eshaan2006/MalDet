import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report

# ————— Config —————
DATA_DIR = os.path.join(os.path.dirname(__file__), "processed_data")
PIPELINE_PATH = os.path.join(DATA_DIR, "processed_datasets.pkl")
FULL_MODEL_PATH = os.path.join(DATA_DIR, "xgb_tier1.pkl")
PRUNED_MODEL_OUT = os.path.join(DATA_DIR, "xgb_tier1_pruned.pkl")
TOP_FEATS_OUT = os.path.join(DATA_DIR, "top100_features.txt")
N_TOP = 100  # adjust for more accuracy (e.g., 80 or 100)

# ————— Load everything —————
data = joblib.load(PIPELINE_PATH)
feature_names = data["feature_names"]

# wrap arrays in DataFrames for easy subsetting
X_train = pd.DataFrame(data["X_train"], columns=feature_names)
X_val   = pd.DataFrame(data["X_val"],   columns=feature_names)
X_test  = pd.DataFrame(data["X_test"],  columns=feature_names)
y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

# Load full model
model_full = joblib.load(FULL_MODEL_PATH)

# ————— Pick top N features by gain —————
# XGBoost importance keys are 'f0','f1',... corresponding to feature_names indices
gain_scores = model_full.get_booster().get_score(importance_type="gain")
sorted_feats = sorted(gain_scores.items(), key=lambda kv: kv[1], reverse=True)
# map f-index to actual feature name
top_feats = [f for f, _ in sorted_feats[:N_TOP]]
top_feature_names = [feature_names[int(f[1:])] for f in top_feats]

# save feature names
with open(TOP_FEATS_OUT, "w") as f:
    f.write("\n".join(top_feature_names))
print(f"Top {N_TOP} features written to {TOP_FEATS_OUT}")

# ————— Subset data —————
X_tr_p = X_train[top_feature_names]
X_val_p = X_val[top_feature_names]
X_te_p = X_test[top_feature_names]

# ————— Retrain on pruned set —————
# reuse params from full model
params = model_full.get_xgb_params()
pruned = XGBClassifier(
    max_depth=params.get('max_depth', 6),
    learning_rate=params.get('eta', 0.1),
    n_estimators=params.get('n_estimators', 100),
    subsample=params.get('subsample', 1.0),
    colsample_bytree=params.get('colsample_bytree', 1.0),
    eval_metric="auc",
    scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train),
    random_state=42,
    n_jobs=4,
    early_stopping_rounds=10
)
pruned.fit(
    X_tr_p, y_train,
    eval_set=[(X_val_p, y_val)],
    verbose=False
)

# ————— Evaluate —————
for split_name, X_sp, y_sp in [("Validation", X_val_p, y_val), ("Test", X_te_p, y_test)]:
    probs = pruned.predict_proba(X_sp)[:,1]
    preds = pruned.predict(X_sp)
    print(f"\n--- {split_name} (pruned) ---")
    print("ROC-AUC:", roc_auc_score(y_sp, probs))
    print(classification_report(y_sp, preds, digits=4))

# ————— Save pruned model —————
joblib.dump(pruned, PRUNED_MODEL_OUT)
print(f"Pruned model saved to {PRUNED_MODEL_OUT}")