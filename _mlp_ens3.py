"""Neural net as a 3rd MEMBER of the shipped champion blend (the CORRECT test).

Champion (LB 58.78%) = tuned HGB + LGBM rank_xendcg, fixed train-CDF blend 0.6*F_hgb+0.4*F_rank.
E39 showed RandomForest looked great vs HGB-ALONE but collapsed on top of the ranker (both trees).
A NEURAL member is a NON-tree function -> potentially orthogonal to BOTH tree members. This mirrors
_rf_ens3.py EXACTLY (same fixed-CDF percentile blend, same HGB+ranker) but the 3rd member is a
neural net, and we SEARCH the neural design space:

  family    : mlp | resmlp (residual) | gru (recurrent over the 3 feature-scales)
  objective : bce (probability loss) | rank (pairwise ranking loss, matches the metric)
  + width/depth/activation/normalization/dropout/lr/weight-decay/ensemble-size

Each config is scored as the 3rd member vs the shipped 2-member baseline, paired-t across pool seeds.
Gate: positive on ALL seeds AND mean_d>=0.0015 AND mean_t>=2.5 (same hardened bar as the laptop).

Usage:
  SMOKE=1 .venv/bin/python -u _mlp_ens3.py
  SEEDS=42,7 NN_CONFIGS=mlp_128-128_bce,gru_scales_rank .venv/bin/python -u _mlp_ens3.py
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

import _ensemble_tune as ET
from _ensemble_tune import (
    WINDOW, champion_feat, load_pool_strat, make_hgb, precompute_fold, score_fold, select_picks,
)
from _exp_fast import ts_auc_from
from _lambdarank import build_grouped, sort_into_groups, fit_ranker
from _overnight import build_training, paired

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAR = int(os.environ.get("PAR", "0"))   # >0 = run this many configs in parallel on CPU (1 thread each)
SMOKE = os.environ.get("SMOKE") == "1"
POOL_SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,7" if SMOKE else "42,7,123").split(",")]
DENSE_K = 30 if SMOKE else 60
BATCH = 1024
N_SCALES, PER_SCALE = 3, 15   # champion_feat = base15 at scales (60,130,250) -> 45 = 3*15

TUNED = dict(learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
             min_samples_leaf=10, l2_regularization=1.0, max_features=0.5)
BASE_W = (0.6, 0.4, 0.0)                      # shipped 2-member champion
WEIGHT_SETS = [BASE_W, (0.50, 0.33, 0.17), (0.45, 0.30, 0.25), (0.40, 0.35, 0.25),
               (0.36, 0.34, 0.30)]


# --- neural members --------------------------------------------------------
def _act(name):
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[name]


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout, act, bn):
        super().__init__()
        a, layers, d = _act(act), [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h)] + ([nn.BatchNorm1d(h)] if bn else []) + [a()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResMLP(nn.Module):
    """Residual fully-connected net: skip connections around each width-preserving block."""

    def __init__(self, in_dim, width, blocks, dropout, act):
        super().__init__()
        a = _act(act)
        self.inp = nn.Sequential(nn.Linear(in_dim, width), a())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(width, width), nn.BatchNorm1d(width), a(),
                          nn.Dropout(dropout), nn.Linear(width, width))
            for _ in range(blocks)
        ])
        self.act = a()
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        h = self.inp(x)
        for blk in self.blocks:
            h = self.act(h + blk(h))
        return self.head(h)


class GRUScales(nn.Module):
    """Read the 45 features as a 3-step sequence (short/medium/long scale) with a GRU."""

    def __init__(self, per_scale, hidden, dropout):
        super().__init__()
        self.gru = nn.GRU(per_scale, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 1))

    def forward(self, x):
        seq = x.view(x.size(0), N_SCALES, PER_SCALE)
        out, _ = self.gru(seq)
        return self.head(out.max(dim=1).values)


def build_net(cfg, in_dim):
    if cfg["fam"] == "mlp":
        return MLP(in_dim, cfg["hidden"], cfg["dropout"], cfg["act"], cfg["bn"])
    if cfg["fam"] == "resmlp":
        return ResMLP(in_dim, cfg["width"], cfg["blocks"], cfg["dropout"], cfg["act"])
    if cfg["fam"] == "gru":
        return GRUScales(PER_SCALE, cfg["hidden"], cfg["dropout"])
    raise ValueError(cfg["fam"])


def rank_loss(logit, y):
    """Pairwise ranking (RankNet-style): every break vs every non-break in the batch."""
    s = logit.squeeze(1)
    pos, neg = s[y > 0.5], s[y < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return torch.nn.functional.binary_cross_entropy_with_logits(s, y)
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)      # (n_pos, n_neg)
    return torch.nn.functional.softplus(-diff).mean()


class NetProba:
    def __init__(self, nets, scaler, dev, batch=100_000):
        self.nets, self.scaler, self.dev, self.batch = nets, scaler, dev, batch

    def predict_proba(self, M):
        Xs = self.scaler.transform(M).astype(np.float32)
        acc = np.zeros(len(Xs))
        for net in self.nets:
            net.eval()
            with torch.no_grad():
                for i in range(0, len(Xs), self.batch):
                    xb = torch.from_numpy(Xs[i:i + self.batch]).to(self.dev)
                    acc[i:i + len(xb)] += torch.sigmoid(net(xb).squeeze(1)).cpu().numpy()
        p = acc / len(self.nets)
        return np.column_stack([1.0 - p, p])


def train_member(Xs, y, cfg, base_seed, dev):
    epochs = 3 if SMOKE else cfg["epochs"]
    k = 1 if SMOKE else cfg["K"]
    lossf = rank_loss if cfg["obj"] == "rank" else (
        lambda lg, yy: torch.nn.functional.binary_cross_entropy_with_logits(lg.squeeze(1), yy))
    Xt, yt = torch.from_numpy(Xs), torch.from_numpy(y.astype(np.float32))
    nets = []
    for j in range(k):
        torch.manual_seed(base_seed + j)
        g = torch.Generator().manual_seed(base_seed + j)
        net = build_net(cfg, Xs.shape[1]).to(dev)
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, shuffle=True, generator=g)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        for _ in range(epochs):
            net.train()
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                lossf(net(xb), yb).backward()
                opt.step()
        nets.append(net)
    return nets


# --- design-space search ---------------------------------------------------
CONFIGS = [
    dict(name="mlp_128-128_bce",  fam="mlp", hidden=[128, 128], dropout=0.1, act="relu", bn=False,
         obj="bce", lr=1e-3, wd=1e-4, epochs=40, K=5),
    dict(name="mlp_128-128_rank", fam="mlp", hidden=[128, 128], dropout=0.1, act="relu", bn=False,
         obj="rank", lr=1e-3, wd=1e-4, epochs=40, K=5),
    dict(name="mlp_64-64-64_bce", fam="mlp", hidden=[64, 64, 64], dropout=0.2, act="relu", bn=True,
         obj="bce", lr=1e-3, wd=1e-4, epochs=40, K=5),
    dict(name="resmlp_128x3_bce", fam="resmlp", width=128, blocks=3, dropout=0.1, act="gelu",
         obj="bce", lr=1e-3, wd=1e-4, epochs=40, K=5),
    dict(name="resmlp_128x3_rank", fam="resmlp", width=128, blocks=3, dropout=0.1, act="gelu",
         obj="rank", lr=1e-3, wd=1e-4, epochs=40, K=5),
    dict(name="gru_scales_bce",   fam="gru", hidden=48, dropout=0.1,
         obj="bce", lr=1e-3, wd=1e-4, epochs=40, K=5),
    dict(name="gru_scales_rank",  fam="gru", hidden=48, dropout=0.1,
         obj="rank", lr=1e-3, wd=1e-4, epochs=40, K=5),
]
_sel = os.environ.get("NN_CONFIGS")
if _sel:
    keep = set(_sel.split(","))
    CONFIGS = [c for c in CONFIGS if c["name"] in keep]
if SMOKE:
    CONFIGS = CONFIGS[:2]


def pct(x, ref):
    return np.searchsorted(ref, x, side="right") / len(ref)


def _eval_config(cfg, Xs, ytr_c, Xtr_c, scaler, caches, fh, fr, base_aucs, y_val, seed, dev_str):
    """Train one neural member + score it as the 3rd blend member. Module-level for loky."""
    torch.set_num_threads(1)
    dev = torch.device(dev_str)
    nets = train_member(Xs, ytr_c, cfg, seed, dev)
    nm = NetProba(nets, scaler, dev)
    ref_nn = np.sort(nm.predict_proba(Xtr_c)[:, 1])
    nn_series = [score_fold(nm, c) for c in caches]
    sa, per = [], {w: [] for w in WEIGHT_SETS}
    for a, b, ns in zip(fh, fr, nn_series):
        fn = pct(ns.to_numpy(), ref_nn)
        sa.append(ts_auc_from(ns.to_frame("prediction"), y_val))
        for w in WEIGHT_SETS:
            blend = w[0] * a + w[1] * b + w[2] * fn
            per[w].append(ts_auc_from(pd.Series(blend, index=ns.index).to_frame("prediction"), y_val))
    best = (BASE_W, -1e9, 0.0)
    for w in WEIGHT_SETS:
        if w == BASE_W:
            continue
        md, _s, t, _ = paired(np.asarray(per[w]), base_aucs)
        if md > best[1]:
            best = (w, md, t)
    return cfg["name"], best[0], best[1], best[2], float(np.mean(sa))


def main():
    print(f"=== NN 3rd-member search {datetime.now():%F %T} | seeds={POOL_SEEDS} "
          f"| {len(CONFIGS)} configs | device={DEVICE} ===", flush=True)
    d_store = {c["name"]: [] for c in CONFIGS}
    t_store = {c["name"]: [] for c in CONFIGS}
    standalone = {c["name"]: [] for c in CONFIGS}

    for seed in POOL_SEEDS:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]
        picks = select_picks(X, train_ids)
        Xtr_c, ytr_c = build_training(X, yi, picks, champion_feat, WINDOW)
        scaler = StandardScaler().fit(Xtr_c)
        Xs = scaler.transform(Xtr_c).astype(np.float32)

        # member 1: tuned HGB
        hgb = make_hgb(**TUNED, random_state=seed)
        with threadpool_limits(limits=1):
            hgb.fit(Xtr_c, ytr_c)
        ref_hgb = np.sort(hgb.predict_proba(Xtr_c)[:, 1])
        hgb_series = [score_fold(hgb, c) for c in caches]

        # member 2: LGBM rank_xendcg (shipped ranker)
        rng = np.random.default_rng(seed + 202)
        rd, ld, td = build_grouped(X, yi, train_ids, champion_feat, WINDOW, DENSE_K, rng)
        rr, lr_, gr = sort_into_groups(rd, ld, td)
        ranker = fit_ranker("rank_xendcg", rr, lr_, gr)
        ref_rank = np.sort(ranker.r.predict(rr))
        rank_series = [score_fold(ranker, c) for c in caches]

        # precompute per-fold percentiles for the two fixed members + 2-member base AUC
        fh = [pct(s.to_numpy(), ref_hgb) for s in hgb_series]
        fr = [pct(s.to_numpy(), ref_rank) for s in rank_series]
        base_aucs = np.asarray([
            ts_auc_from(pd.Series(0.6 * a + 0.4 * b, index=s.index).to_frame("prediction"), y_val)
            for a, b, s in zip(fh, fr, hgb_series)])
        print(f"\n[seed {seed}] 2-member base mean={base_aucs.mean():.4f} ({time.time()-t0:.0f}s)",
              flush=True)

        dev_str = "cpu" if PAR else str(DEVICE)
        if PAR:
            from joblib import Parallel, delayed
            res = Parallel(n_jobs=PAR, backend="loky")(
                delayed(_eval_config)(cfg, Xs, ytr_c, Xtr_c, scaler, caches, fh, fr,
                                      base_aucs, y_val, seed, "cpu") for cfg in CONFIGS)
        else:
            res = [_eval_config(cfg, Xs, ytr_c, Xtr_c, scaler, caches, fh, fr,
                                base_aucs, y_val, seed, dev_str) for cfg in CONFIGS]
        for name, bw, bd, bt, sa in res:
            d_store[name].append(bd)
            t_store[name].append(bt)
            standalone[name].append(sa)
            print(f"  {name:<18} standalone={sa:.4f} best_w={bw} d={bd:+.4f} t={bt:+.2f}", flush=True)
        del X, caches, hgb_series, rank_series

    print("\n=== SUMMARY (3rd member vs shipped 2-member champion) ===", flush=True)
    for name in sorted(d_store, key=lambda n: -np.mean(d_store[n])):
        ds, ts_ = d_store[name], t_store[name]
        npos = sum(1 for x in ds if x > 0)
        gate = "SHIP(hard)" if npos == len(ds) and np.mean(ds) >= 0.0015 and min(ts_) >= 2.5 else (
            "promising" if npos == len(ds) and np.mean(ds) > 0 else "noise")
        print(f"{name:<18} mean_d {np.mean(ds):+.4f} min_d {min(ds):+.4f} pos {npos}/{len(ds)} "
              f"min_t {min(ts_):+.2f} standalone {np.mean(standalone[name]):.4f} -> {gate}", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
