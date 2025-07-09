import sys
import torch
import numpy as np
from tier2_pytorch import ByteCNN  # assumes tier2_pytorch.py is in your PYTHONPATH
from tier2_pytorch import ByteDataset  # for the data loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "byte_cnn_pt.pth"

def load_model():
    model = ByteCNN().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def infer(model, path):
    # wrap in a ByteDataset of length 1 to reuse the same preprocessing
    ds = ByteDataset([path], [0])      # label dummy
    x, _ = ds[0]
    with torch.no_grad():
        logit = model(x.unsqueeze(0).to(DEVICE))
        prob = torch.sigmoid(logit).item()
    return prob

if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_byte_cnn.py <path_to_exe>")
        sys.exit(1)

    path = sys.argv[1]
    model = load_model()
    p = infer(model, path)
    print(f"→ Malware probability: {p:.4f}")
    print("→ Verdict:", "MALWARE" if p >= 0.75 else "BENIGN")