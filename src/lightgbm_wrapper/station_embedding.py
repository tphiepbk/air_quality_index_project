# Do not put station_id to LightGBM model (to avoid overfitting by ID)
# Learn a vector embedding using PyTorch
# Attach the embedding vector to dataset like "static features"
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

EMBED_DIM = 4
EMBED_EPOCHS = 50
EMBED_LR = 1e-2
EMBED_BS = 2048
EMBED_WD = 1e-4
RANDOM_STATE = 42

def learn_station_embeddings(df_feat,
                             target_col,
                             station_col="station_id",
                             embed_dim=EMBED_DIM,
                             epochs=EMBED_EPOCHS,
                             lr=EMBED_LR,
                             batch_size=EMBED_BS,
                             weight_decay=EMBED_WD,
                             seed=RANDOM_STATE,
                             LIGHTGBM_DIR="."):
    torch.manual_seed(seed)

    df_e = df_feat[[station_col, target_col]].dropna().copy()
    station_ids = df_e[station_col].astype(int).values
    y = df_e[target_col].astype(float).values

    # map station_id -> index [0..n_station-1]
    uniq_ids = np.unique(station_ids)
    id2idx = {sid:i for i,sid in enumerate(uniq_ids)}
    idx = np.array([id2idx[sid] for sid in station_ids], dtype=np.int64)

    # Randomly split
    n = len(idx)
    perm = np.random.RandomState(seed).permutation(n)
    n_train = int(0.8 * n)
    tr, va = perm[:n_train], perm[n_train:]

    x_tr = torch.tensor(idx[tr], dtype=torch.long)
    y_tr = torch.tensor(y[tr], dtype=torch.float32).unsqueeze(1)
    x_va = torch.tensor(idx[va], dtype=torch.long)
    y_va = torch.tensor(y[va], dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(x_va, y_va), batch_size=batch_size, shuffle=False)

    class StationEmbedReg(nn.Module):
        def __init__(self, n_station, d):
            super().__init__()
            self.emb = nn.Embedding(n_station, d)
            self.lin = nn.Linear(d, 1)
        def forward(self, x):
            e = self.emb(x)
            return self.lin(e)

    # Create model
    model = StationEmbedReg(n_station=len(uniq_ids), d=embed_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # Start training
    best_val = float("inf")
    best_state = None
    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(tr)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_loss += loss.item() * len(xb)
        va_loss /= max(len(va), 1)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}

        if ep % 10 == 0 or ep == 1:
            print(f"[Embed] epoch {ep:03d} | train MSE={tr_loss:.4f} | val MSE={va_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save the model
    torch.save(model.state_dict(), os.path.join(LIGHTGBM_DIR, f"{target_col}_station_embedding.pt"))

    # Get the embedding matrix
    # shape (n_station, embed_dim)
    emb_weight = model.emb.weight.detach().cpu().numpy()

    # Create dictionary look up: station_id => vector
    embed_lookup = {sid: emb_weight[id2idx[sid]] for sid in uniq_ids}

    return embed_lookup, uniq_ids

def attach_station_embedding(df, lookup, station_col="station_id", prefix="station_emb"):
    df = df.copy()
    # mean embedding dùng cho trạm lạ (nếu có)
    mean_emb = np.mean(np.stack(list(lookup.values())), axis=0)

    emb_cols = [f"{prefix}_{i}" for i in range(len(mean_emb))]
    # tạo ma trận embedding theo từng dòng
    embs = []
    for sid in df[station_col].astype(int).values:
        embs.append(lookup.get(sid, mean_emb))
    embs = np.vstack(embs)
    for j, col in enumerate(emb_cols):
        df[col] = embs[:, j]
    return df, emb_cols