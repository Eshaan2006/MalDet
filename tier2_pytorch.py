import os
import glob
import argparse
import pefile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

# PyTorch Geometric imports for GNN
import torch_geometric
from torch_geometric.data import Data as GeoData
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ——— Config ———
DATA_DIR    = "/home/ubuntu/mal-dec/DikeDataset/files"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
TEST_RATIO  = 0.2
BATCH_SIZE  = 32
EPOCHS      = 5

# ——— Helpers ———
def is_pe(path):
    try:
        with open(path, 'rb') as f:
            return f.read(2) == b'MZ'
    except:
        return False

# ——— Byte-CNN ———
class ByteDataset(Dataset):
    def __init__(self, paths, labels, n_bytes=256*1024):
        self.paths = paths
        self.labels = labels
        self.n_bytes = n_bytes
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        with open(p, 'rb') as f:
            data = f.read(self.n_bytes)
        if len(data) < self.n_bytes:
            data += b"\x00" * (self.n_bytes - len(data))
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.float)

class ByteCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2)
        self.pool1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2)
        self.pool2 = nn.MaxPool1d(3)
        with torch.no_grad():
            dummy = torch.zeros(1,1,256*1024)
            x = self.pool2(torch.relu(self.conv2(self.pool1(torch.relu(self.conv1(dummy))))))
            self.flat_size = x.view(1, -1).shape[1]
        self.fc1 = nn.Linear(self.flat_size, 64)
        self.fc2 = nn.Linear(64, 1)
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

# ——— Import-Graph GNN ———
class ImportGraphDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p, lbl = self.paths[idx], self.labels[idx]
        try:
            pe = pefile.PE(p)
            nodes, edges = [], []
            for imp in getattr(pe, 'DIRECTORY_ENTRY_IMPORT', []):
                dll = imp.dll.decode(errors='ignore')
                for e in imp.imports:
                    api = e.name.decode(errors='ignore') if e.name else f"ord{e.ordinal}"
                    for n in (dll, api):
                        if n not in nodes: nodes.append(n)
                    edges.append((nodes.index(dll), nodes.index(api)))
        except pefile.PEFormatError:
            nodes, edges = [], []
        if not nodes:
            x = torch.zeros((1,10))
            edge_index = torch.zeros((2,0), dtype=torch.long)
        else:
            dim = min(10, len(nodes))
            eye = torch.eye(len(nodes), dim)
            if dim < 10:
                eye = torch.cat([eye, torch.zeros(len(nodes), 10-dim)], dim=1)
            x = eye
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return GeoData(x=x, edge_index=edge_index, y=torch.tensor([lbl], dtype=torch.float))

class ImportGNN(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.conv1 = GCNConv(10, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin   = nn.Linear(hidden, 1)
    def forward(self, data):
        x, edge_index, batch = data.x.to(DEVICE), data.edge_index.to(DEVICE), data.batch.to(DEVICE)
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x).squeeze(-1)

# ——— API-Sequence RNN ———
class APISequenceDataset(Dataset):
    def __init__(self, paths, labels, seq_len=500):
        self.paths, self.labels, self.seq_len = paths, labels, seq_len
        self.api2id = self._build_vocab()
    def _build_vocab(self):
        vocab = set()
        for p in self.paths:
            try:
                pe = pefile.PE(p, fast_load=True)
                for imp in getattr(pe, 'DIRECTORY_ENTRY_IMPORT', []):
                    for e in imp.imports:
                        name = e.name.decode(errors='ignore') if e.name else f"ord{e.ordinal}"
                        vocab.add(name)
            except:
                continue
        return {api: i+1 for i, api in enumerate(sorted(vocab))}
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p, lbl = self.paths[idx], self.labels[idx]
        seq = []
        try:
            pe = pefile.PE(p)
            for imp in getattr(pe, 'DIRECTORY_ENTRY_IMPORT', []):
                for e in imp.imports:
                    name = e.name.decode(errors='ignore') if e.name else f"ord{e.ordinal}"
                    seq.append(self.api2id.get(name, 0))
        except:
            pass
        seq = seq[:self.seq_len] + [0]*(self.seq_len - len(seq))
        return torch.tensor(seq, dtype=torch.long), torch.tensor(lbl, dtype=torch.float)

class APIRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size+1, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, bidirectional=True, batch_first=True)
        self.fc   = nn.Linear(hidden*2, 1)
    def forward(self, x):
        x = self.emb(x.to(DEVICE))
        _, (h, _) = self.lstm(x)
        out = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(out).squeeze(-1)

# ——— Common split + evaluate ———
def split_paths_labels():
    benign = glob.glob(os.path.join(DATA_DIR, 'benign', '**', '*.exe'), recursive=True)
    malware = glob.glob(os.path.join(DATA_DIR, 'malware', '**', '*.exe'), recursive=True)
    paths = [p for p in benign+malware if is_pe(p)]
    labels = [0]*len(benign) + [1]*len(malware)
    return paths, labels

def split_ds(ds):
    n = len(ds)
    ntr = int(n*TRAIN_RATIO)
    nval = int(n*VAL_RATIO)
    nte = n - ntr - nval
    return random_split(ds, [ntr, nval, nte])

def evaluate(model, loader, is_graph=False):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            if is_graph:
                batch = batch.to(DEVICE)
                logits = model(batch)
                y_batch = batch.y.cpu().numpy()
            else:
                X, y_batch = batch
                X, y_batch = X.to(DEVICE), y_batch.to(DEVICE)
                logits = model(X)
                y_batch = y_batch.cpu().numpy()
            prob = torch.sigmoid(logits)
            arr = prob.cpu().numpy()
            ys.extend(y_batch.tolist())
            if arr.ndim == 0:
                ps.append(float(arr))
            else:
                ps.extend(arr.tolist())
    return roc_auc_score(ys, ps), accuracy_score(ys, (np.array(ps)>=0.5).astype(int))

# ——— Training routines ———
def train_byte():
    paths, labels = split_paths_labels()
    ds = ByteDataset(paths, labels)
    ds_tr, ds_val, ds_te = split_ds(ds)
    # sampler for imbalance
    label_tr = [ds_tr[i][1].item() for i in range(len(ds_tr))]
    w = [1.0/sum(label_tr) if l==1 else 1.0/sum(1-np.array(label_tr)) for l in label_tr]
    sampler = WeightedRandomSampler(w, len(w), replacement=True)
    loader_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, sampler=sampler)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE)
    model = ByteCNN().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    for e in range(EPOCHS):
        model.train()
        for X,y in loader_tr:
            X,y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
        auc, acc = evaluate(model, loader_val)
        print(f"Epoch {e} ByteCNN Val AUC: {auc:.4f}, Acc: {acc:.4f}")
    auc, acc = evaluate(model, DataLoader(ds_te, batch_size=BATCH_SIZE))
    print(f"Test ByteCNN AUC: {auc:.4f}, Acc: {acc:.4f}")

def train_gnn():
    paths, labels = split_paths_labels()
    ds = ImportGraphDataset(paths, labels)
    ds_tr, ds_val, ds_te = split_ds(ds)
    loader_tr = GeoDataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    loader_val = GeoDataLoader(ds_val, batch_size=BATCH_SIZE)
    model = ImportGNN().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    for e in range(EPOCHS):
        model.train()
        for batch in loader_tr:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y.squeeze())
            loss.backward()
            opt.step()
        auc, acc = evaluate(model, loader_val, is_graph=True)
        print(f"Epoch {e} ImportGNN Val AUC: {auc:.4f}, Acc: {acc:.4f}")
    auc, acc = evaluate(model, GeoDataLoader(ds_te, batch_size=BATCH_SIZE), is_graph=True)
    print(f"Test ImportGNN AUC: {auc:.4f}, Acc: {acc:.4f}")

def train_rnn():
    paths, labels = split_paths_labels()
    ds = APISequenceDataset(paths, labels)
    ds_tr, ds_val, ds_te = split_ds(ds)
    loader_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE)
    vocab_size = len(ds.api2id)
    model = APIRNN(vocab_size).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    for e in range(EPOCHS):
        model.train()
        for X,y in loader_tr:
            X,y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
        auc, acc = evaluate(model, loader_val)
        print(f"Epoch {e} APIRNN Val AUC: {auc:.4f}, Acc: {acc:.4f}")
    auc, acc = evaluate(model, DataLoader(ds_te, batch_size=BATCH_SIZE))
    print(f"Test APIRNN AUC: {auc:.4f}, Acc: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tier-2 PyTorch models")
    parser.add_argument('model', choices=['byte','gnn','rnn','all'])
    args = parser.parse_args()
    if args.model in ('byte','all'):
        train_byte()
    if args.model in ('gnn','all'):
        train_gnn()
    if args.model in ('rnn','all'):
        train_rnn()