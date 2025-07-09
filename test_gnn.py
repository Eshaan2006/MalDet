import torch, sys
import pefile
import numpy as np
from torch_geometric.data import Data
from tier2_pytorch import ImportGNN  # pull in your class

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/ubuntu/mal-dec/program/processed_data/gnn_tier2.pt"

def make_graph(path):
    try:
        pe = pefile.PE(path)
        nodes, edges = [], []
        for imp in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
            dll = imp.dll.decode(errors="ignore")
            for e in imp.imports:
                api = (e.name or f"ord{e.ordinal}").decode(errors="ignore")
                for n in (dll, api):
                    if n not in nodes: nodes.append(n)
                edges.append((nodes.index(dll), nodes.index(api)))
    except:
        nodes, edges = [], []
    if not nodes:
        x = torch.zeros((1,10))
        edge_index = torch.zeros((2,0),dtype=torch.long)
    else:
        dim = min(10, len(nodes))
        eye = torch.eye(len(nodes), dim)
        if dim < 10:
            eye = torch.cat([eye, torch.zeros(len(nodes), 10-dim)],1)
        x = eye
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

if __name__=="__main__":
    model = ImportGNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    g = make_graph(sys.argv[1]).to(DEVICE)
    with torch.no_grad():
        p = torch.sigmoid(model(g)).item()
    print(f"ImportGNN → Malware probability: {p:.4f}")