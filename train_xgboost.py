import os
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "processed_data")
PIPELINE_PATH = os.path.join(DATA_DIR, "processed_datasets.pkl")
MODEL_OUT = os.path.join(DATA_DIR, "xgb_tier1.pkl")

# Load processed datasets
data = joblib.load(PIPELINE_PATH)
X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

# --------------------------------------------------------------------------
# 1) Hyperparameter tuning with Grid Search (optional, suppress warnings)
# --------------------------------------------------------------------------
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base_model = XGBClassifier(
    eval_metric='auc',
    scale_pos_weight=(len(y_train) - np.sum(y_train)) / np.sum(y_train),
    random_state=42,
    n_jobs=4
)

grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    verbose=1,
    n_jobs=-1
)
print("Starting hyperparameter search...")
grid.fit(X_train, y_train)
print("Best parameters found:", grid.best_params_)
model = grid.best_estimator_

# --------------------------------------------------------------------------
# 2) Train final model with early stopping on validation set
# --------------------------------------------------------------------------
print("Training final model with early stopping on validation set...")
model.set_params(early_stopping_rounds=10)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# --------------------------------------------------------------------------
# 3) Evaluate on validation and test sets
# --------------------------------------------------------------------------
for split_name, X_split, y_split in [("Validation", X_val, y_val), ("Test", X_test, y_test)]:
    probs = model.predict_proba(X_split)[:, 1]
    preds = model.predict(X_split)
    auc = roc_auc_score(y_split, probs)
    report = classification_report(y_split, preds, digits=4)
    print(f"\n=== {split_name} Results ===")
    print(f"ROC-AUC: {auc:.4f}")
    print(report)

    # Threshold selection for 95% recall
    precision, recall, thresholds = precision_recall_curve(y_split, probs)
    idx = np.where(recall >= 0.95)[0]
    if idx.size > 0:
        thr = thresholds[idx[0]]
        print(f"Threshold for 95% recall: {thr:.4f}")

# --------------------------------------------------------------------------
# 4) Save the trained model
# --------------------------------------------------------------------------
joblib.dump(model, MODEL_OUT)
print(f"Model saved to {MODEL_OUT}")