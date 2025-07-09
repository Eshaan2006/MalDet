import sys
import joblib
import pandas as pd
import numpy as np
from extract_features import PEFeatureExtractor

# Paths (adjust if your structure is different)
DATA_DIR      = "processed_data"
MODEL_PATH    = f"{DATA_DIR}/xgb_tier1_pruned.pkl"
SCALER_PATH   = f"{DATA_DIR}/scaler.pkl"
FEATURES_PATH = f"{DATA_DIR}/top100_features.txt"  # or top50

# Load model and scaler
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Extract full names and parameters from scaler
orig_feats = list(scaler.feature_names_in_)
means      = scaler.mean_
scales     = scaler.scale_

# Load pruned feature list
with open(FEATURES_PATH) as f:
    top_feats = [line.strip() for line in f]

# Check args
if len(sys.argv) != 2:
    print("Usage: python test_sample.py <path_to_pe_file.exe>")
    sys.exit(1)
file_path = sys.argv[1]

# Extract features
extractor = PEFeatureExtractor()
feat_dict = extractor.extract_features(file_path)
if feat_dict is None:
    print("❌ Failed to parse PE. Is this a valid Windows executable?")
    sys.exit(1)

# Build DataFrame with only pruned features
df = pd.DataFrame([feat_dict])
X = df[top_feats].fillna(0).astype(float)

# Manual scaling for pruned features
indices       = [orig_feats.index(f) for f in top_feats]
pruned_means  = means[indices]
pruned_scales = scales[indices]
x = X.values
x_scaled = (x - pruned_means) / pruned_scales

# Predict
prob_malware = model.predict_proba(x_scaled)[0,1]
print(f"🔍 {file_path}\n  → Malware probability: {prob_malware:.4f}")

# Apply decision threshold (e.g. 0.0345)
threshold = 0.75
verdict = "MALWARE" if prob_malware >= threshold else "BENIGN"
print(f"  → Verdict (@{threshold}): {verdict}")