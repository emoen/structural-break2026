"""MLP-over-champion-features search — hunt for a DECORRELATED ensemble member.

The CNN/SeqHist over raw windows failed (short/noisy). But every winning member (tuned HGB,
LGBM ranker E36, RandomForest E38) works on the 45 champion features. An MLP over those same
45 features is a different model class (smooth global function vs axis-aligned trees) that may
decorrelate like RF did. Cheap (45-dim in), so we search many architectures.

Each config is scored the way the ensemble is built (E36/E38): fixed-CDF rank blend with the
tuned champion HGB, b = w*Fhat_hgb + (1-w)*Fhat_mlp, swept w, paired vs HGB-alone (w=1), across
2 screen pool seeds. Also reports the MLP's standalone TS-AUC. Winners (member delta > 0 on both
seeds) get confirmed on 4 seeds next.

Usage:
  SMOKE=1 .venv/bin/python -u _mlp_search.py
  SEEDS=42,123 .venv/bin/python -u _mlp_search.py
"""
import math
import os
import time
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from _exp_fast import ts_auc_from
from _overnight import build_training
from _ensemble_tune import (
    SMOKE, WINDOW, champion_feat, make_hgb, precompute_fold, score_fold, select_picks,
)
from _nn_fusion import TUNED_HGB
from _nn_member import cdf_fn, load_pool_strat_seed

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
K_ENS = 1 if SMOKE else int(os.environ.get("K_ENS", "3"))  # MLP seeds averaged per config
EPOCHS = 3 if SMOKE else 40
BATCH = 1024
WS = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,123").split(",")]
if SMOKE:
    SEEDS = [42]

# --- architecture / hyperparameter search space -----------------------------
CONFIGS = [
    dict(name="64",            hidden=[64],           dropout=0.0, act="relu", bn=False, lr=1e-3, wd=1e-4),
    dict(name="128bn",         hidden=[128],          dropout=0.2, act="relu", bn=True,  lr=1e-3, wd=1e-4),
    dict(name="256bn_gelu",    hidden=[256],          dropout=0.3, act="gelu", bn=True,  lr=1e-3, wd=1e-4),
    dict(name="128-64",        hidden=[128, 64],      dropout=0.2, act="relu", bn=True,  lr=1e-3, wd=1e-4),
    dict(name="256-128",       hidden=[256, 128],     dropout=0.3, act="relu", bn=True,  lr=1e-3, wd=1e-4),
    dict(name="256-128_gelu",  hidden=[256, 128],     dropout=0.3, act="gelu", bn=True,  lr=1e-3, wd=1e-3),
    dict(name="256-128-64",    hidden=[256, 128, 64], dropout=0.3, act="relu", bn=True,  lr=1e-3, wd=1e-4),
    dict(name="512-256-128",   hidden=[512, 256, 128], dropout=0.4, act="gelu", bn=True, lr=5e-4, wd=1e-3),
    dict(name="128-128_ln",    hidden=[128, 128],     dropout=0.1, act="relu", bn=False, lr=1e-3, wd=1e-4),
    dict(name="64-64-64",      hidden=[64, 64, 64],   dropout=0.2, act="relu", bn=True,  lr=1e-3, wd=1e-4),
    dict(name="256-256_wide",  hidden=[256, 256],     dropout=0.4, act="relu", bn=True,  lr=5e-4, wd=1e-3),
    dict(name="128-64_lowdrop", hidden=[128, 64],     dropout=0.0, act="relu", bn=True,  lr=1e-3, wd=1e-4),
]
if SMOKE:
    CONFIGS = CONFIGS[:3]
_sel = os.environ.get("MLP_CONFIGS")
if _sel:
    keep = set(_sel.split(","))
    CONFIGS = [c for c in CONFIGS if c["name"] in keep]


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout, act, bn):
        super().__init__()
        a = nn.ReLU if act == "relu" else nn.GELU
        layers, d = [], in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(a())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPProba:
    def __init__(self, nets, scaler, batch=100_000):
        self.nets = nets
        self.scaler = scaler
        self.batch = batch

    def predict_proba(self, M):
        Xs = self.scaler.transform(M).astype(np.float32)
        n = len(Xs)
        acc = np.zeros(n)
        for net in self.nets:
            net.eval()
            with torch.no_grad():
                for i in range(0, n, self.batch):
                    xb = torch.from_numpy(Xs[i:i + self.batch]).to(DEVICE)
                    acc[i:i + len(xb)] += torch.sigmoid(net(xb).squeeze(1)).cpu().numpy()
        p = acc / len(self.nets)
        return np.column_stack([1.0 - p, p])


def train_mlp(Xs, y, cfg, epochs, k):
    nets = []
    yt = torch.from_numpy(y.astype(np.float32))
    Xt = torch.from_numpy(Xs)
    for j in range(k):
        torch.manual_seed(SEED + j)
        g = torch.Generator().manual_seed(SEED + j)
        net = MLP(Xs.shape[1], cfg["hidden"], cfg["dropout"], cfg["act"], cfg["bn"]).to(DEVICE)
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, shuffle=True, generator=g)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        lossf = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            net.train()
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                lossf(net(xb).squeeze(1), yb).backward()
                opt.step()
        nets.append(net)
    return nets


def seed_setup(seed):
    """Per-seed: tuned HGB baseline + champion caches + fold base TS-AUC + scaler/train rows."""
    X, yi, y_val, train_ids, folds = load_pool_strat_seed(seed)
    picks = select_picks(X, train_ids)
    Xtr, ytr = build_training(X, yi, picks, champion_feat, WINDOW)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr).astype(np.float32)
    hgb = make_hgb(**TUNED_HGB)
    with threadpool_limits(limits=1):
        hgb.fit(Xtr, ytr)
    F_hgb = cdf_fn(hgb.predict_proba(Xtr)[:, 1])
    caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]
    hgb_val = [score_fold(hgb, c) for c in caches]
    base = np.asarray([ts_auc_from(s.to_frame("prediction"), y_val) for s in hgb_val])
    return dict(Xtr=Xtr, ytr=ytr, Xtr_s=Xtr_s, scaler=scaler, F_hgb=F_hgb,
                caches=caches, hgb_val=hgb_val, y_val=y_val, base=base)


def eval_config(cfg, S):
    """Return (standalone_mean, {w: per-fold delta array}) for one config on one seed."""
    nets = train_mlp(S["Xtr_s"], S["ytr"], cfg, EPOCHS, K_ENS)
    mlp = MLPProba(nets, S["scaler"])
    F_mlp = cdf_fn(mlp.predict_proba(S["Xtr"])[:, 1])
    standalone, per_w = [], {w: [] for w in WS}
    for hs, cc in zip(S["hgb_val"], S["caches"]):
        ms = score_fold(mlp, cc)
        standalone.append(ts_auc_from(ms.to_frame("prediction"), S["y_val"]))
        fh, fm = S["F_hgb"](hs.values), F_mlp(ms.values)
        for w in WS:
            b = w * fh + (1.0 - w) * fm
            per_w[w].append(ts_auc_from(pd.Series(b, index=hs.index).to_frame("prediction"),
                                        S["y_val"]))
    d = {w: np.asarray(per_w[w]) - S["base"] for w in WS}
    return float(np.mean(standalone)), d


def main():
    t0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print(f"=== mlp search ({mode}) {datetime.now():%F %T} | seeds={SEEDS} K={K_ENS} "
          f"epochs={EPOCHS} | {len(CONFIGS)} configs ===", flush=True)

    setups = {}
    for seed in SEEDS:
        ts = time.time()
        setups[seed] = seed_setup(seed)
        print(f"seed {seed}: tuned HGB base mean={setups[seed]['base'].mean():.4f} "
              f"({(time.time() - ts) / 60:.1f}m)", flush=True)

    results = {}
    for cfg in CONFIGS:
        alone, dmeans, dts = [], {w: [] for w in WS}, {w: [] for w in WS}
        for seed in SEEDS:
            a, d = eval_config(cfg, setups[seed])
            alone.append(a)
            for w in WS:
                arr = d[w]
                md, sd = float(arr.mean()), float(arr.std(ddof=1))
                dmeans[w].append(md)
                dts[w].append(md / (sd / math.sqrt(len(arr))) if sd > 0 else float("inf"))
        best_w = max((w for w in WS if w != 1.0), key=lambda w: np.mean(dmeans[w]))
        ds, tsv = dmeans[best_w], dts[best_w]
        npos = sum(1 for x in ds if x > 0)
        results[cfg["name"]] = (float(np.mean(alone)), best_w, float(np.mean(ds)),
                                min(ds), npos, min(tsv))
        print(f"{cfg['name']:<16} standalone={np.mean(alone):.4f} | best w={best_w} "
              f"member_d={np.mean(ds):+.4f} pos={npos}/{len(SEEDS)} min_t={min(tsv):+.2f} "
              f"per_seed={[f'{x:+.4f}' for x in ds]}", flush=True)

    print("\n=== RANKED (by member delta) ===", flush=True)
    for name, (al, bw, md, mind, npos, mint) in sorted(results.items(), key=lambda kv: -kv[1][2]):
        gate = "CONFIRM(ship)" if npos == len(SEEDS) and mint >= 2.5 else (
            "PROMISING" if npos == len(SEEDS) and md > 0 else "noise")
        print(f"{name:<16} member_d {md:+.4f} (w{bw}) pos {npos}/{len(SEEDS)} "
              f"min_t {mint:+.2f} standalone {al:.4f} -> {gate}", flush=True)
    print(f"=== DONE in {(time.time() - t0) / 60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
