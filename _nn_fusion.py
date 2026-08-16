"""Feature FUSION: champion 45 features + seqhist NN probability as an extra HGB feature.

The architecture search (_nn_twosample.py, 10-fold honest) picked seqhist (CNN over the
ordered window + hist set) as the best NN; its linear BLEND with the champion reached only
t=1.94. Blending is linear; FUSION feeds the NN probability as a feature so the GBT can use
it CONDITIONALLY (only where the champion is ambiguous) -> the real test of the "NN catches
the champion's blind spots" hypothesis.

Leakage control: the NN probability for a champion TRAINING row must come from a seqhist that
never saw that row's series -> J-fold CROSS-FITTING over the fit series produces out-of-fold
(OOF) probabilities. For the eval folds, a seqhist trained on ALL fit series scores them.

Comparison (10 stratified folds, paired): HGB(champion45) vs HGB(champion45 + nn_prob).

Usage:
  SMOKE=1 .venv/bin/python -u _nn_fusion.py
  STEPS=40 .venv/bin/python -u _nn_fusion.py
"""
import os
import time
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from _exp_fast import ts_auc_from  # noqa: F401
from _overnight import build_training, paired
from _ensemble_tune import (
    FOLD_SIZE, SMOKE, WINDOW,
    champion_feat, eval_model, load_pool_strat, make_hgb,
    precompute_fold, report, select_picks,
)
from _nn_twosample import (
    BATCH, DEVICE, LR, SEED, STEPS, WD,
    PairProba, SeqHistTS, _split_pair, build_training_pairs, pair_vec, select_picks_n,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

J_CROSSFIT = 2 if SMOKE else 5
NN_EPOCHS = 2 if SMOKE else int(os.environ.get("NN_EPOCHS", "8"))  # seqhist es-picked ~8
ENSEMBLE = 1 if SMOKE else int(os.environ.get("ENSEMBLE", "1"))  # seeds averaged per net
PRETRAIN = os.environ.get("PRETRAIN") == "1"           # pretrain on synthetic then fine-tune
SYNTH_DIR = os.environ.get("SYNTH_DIR", "synthetic")   # synthetic dataset folder
PRE_EPOCHS = 1 if SMOKE else int(os.environ.get("PRE_EPOCHS", "4"))
PRE_STEPS = 6 if SMOKE else int(os.environ.get("PRE_STEPS", "40"))
TUNED = os.environ.get("TUNED") == "1"  # fuse into the SHIPPED tuned HGB, not the old champion
# shipped E32 tuned config (streaming_detector_tuned.ipynb); key lever = max_features=0.5
TUNED_HGB = dict(learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
                 min_samples_leaf=10, l2_regularization=1.0, max_features=0.5)


def hgb():
    return make_hgb(**TUNED_HGB) if TUNED else make_hgb()
EMBED = os.environ.get("EMBED") == "1"  # fuse 64-d penultimate vector instead of scalar prob


def train_fixed(model, Xtr, ytr, epochs, seed, init_state=None):
    """Train seqhist a fixed number of epochs; optionally warm-start from init_state."""
    if init_state is not None:
        model.load_state_dict(init_state)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model.to(DEVICE)
    dl = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))),
        batch_size=BATCH, shuffle=True, generator=g,
    )
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            win, hist = _split_pair(xb)
            opt.zero_grad()
            lossf(model(win, hist).squeeze(1), yb).backward()
            opt.step()
    return model


def train_ensemble(Xtr, ytr, k, epochs, seed, init_state=None):
    """Train k seqhist nets with different seeds (variance reduction)."""
    return [train_fixed(SeqHistTS(), Xtr, ytr, epochs, seed + j, init_state) for j in range(k)]


def pretrain_synth(synth_dir, steps, epochs, seed):
    """Pretrain one seqhist on the synthetic breaks; return its state_dict (warm-start init)."""
    Xs = pd.read_parquet(os.path.join(synth_dir, "X_train.parquet"))
    yis = pd.read_parquet(os.path.join(synth_dir, "y_train_index.parquet"))
    ids = yis.index.to_numpy()
    picks = select_picks_n(Xs, ids, steps)
    Xtr, ytr = build_training_pairs(Xs, yis, picks, WINDOW)
    print(f"pretrain(synthetic): {Xtr.shape[0]} windows from {len(ids)} series, "
          f"positives {ytr.mean():.3f}", flush=True)
    net = train_fixed(SeqHistTS(), Xtr, ytr, epochs, seed)
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


class EnsembleProba:
    """Average predict_proba over a list of trained nets (drop-in for score_fold)."""

    def __init__(self, nets):
        self.probas = [PairProba(n) for n in nets]

    def predict_proba(self, M):
        p = np.mean([pp.predict_proba(M)[:, 1] for pp in self.probas], axis=0)
        return np.column_stack([1.0 - p, p])

    def predict_embed(self, M):
        return np.mean([pp.predict_embed(M) for pp in self.probas], axis=0)


def main():
    t0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print(f"=== nn fusion ({mode}) start {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    print(f"device={DEVICE} STEPS={STEPS} NN_EPOCHS={NN_EPOCHS} J={J_CROSSFIT} "
          f"ENSEMBLE={ENSEMBLE} PRETRAIN={int(PRETRAIN)} EMBED={int(EMBED)} "
          f"TUNED={int(TUNED)} SEED={SEED}", flush=True)

    X, yi, y_val, train_ids, folds = load_pool_strat()
    picks_c = select_picks(X, train_ids)            # champion rows (canonical 10 steps)
    picks_n = select_picks_n(X, train_ids, STEPS)   # NN training windows (STEPS/series)

    pre_state = None
    if PRETRAIN:
        tf = time.time()
        pre_state = pretrain_synth(SYNTH_DIR, PRE_STEPS, PRE_EPOCHS, SEED)
        print(f"pretrain done in {(time.time() - tf) / 60:.1f} min", flush=True)

    # champion training features + labels (fusion HGB trains on these rows)
    Xtr_c, ytr = build_training(X, yi, picks_c, champion_feat, WINDOW)
    # pair vectors for the SAME champion rows (to attach OOF nn-prob), aligned row-for-row
    Xrows_p, yrows = build_training_pairs(X, yi, picks_c, WINDOW)
    assert len(Xrows_p) == len(Xtr_c) and np.array_equal(yrows, ytr), "row alignment broken"
    row_sids = np.array([sid for sid, ts in picks_c.items() for _ in ts])
    print(f"champion rows {Xtr_c.shape}, pair rows {Xrows_p.shape}", flush=True)

    # --- OOF nn-prob via J-fold cross-fitting over the fit series ---
    fit_series = np.array(list(picks_c.keys()))
    rng = np.random.default_rng(SEED)
    groups = np.array_split(rng.permutation(len(fit_series)), J_CROSSFIT)
    oof = np.full((len(Xrows_p), 64), np.nan) if EMBED else np.full(len(Xrows_p), np.nan)
    for gi, g in enumerate(groups):
        tf = time.time()
        hold = set(fit_series[g].tolist())
        picks_train = {s: picks_n[s] for s in fit_series if s not in hold and s in picks_n}
        Xtr_p, ytr_p = build_training_pairs(X, yi, picks_train, WINDOW)
        nets = train_ensemble(Xtr_p, ytr_p, ENSEMBLE, NN_EPOCHS, SEED + gi * 10, pre_state)
        ens = EnsembleProba(nets)
        mask = np.isin(row_sids, list(hold))
        oof[mask] = ens.predict_embed(Xrows_p[mask]) if EMBED \
            else ens.predict_proba(Xrows_p[mask])[:, 1]
        print(f"  crossfit {gi + 1}/{J_CROSSFIT}: train {Xtr_p.shape[0]} rows, "
              f"filled {int(mask.sum())} oof ({(time.time() - tf) / 60:.1f} min)", flush=True)
    assert not np.isnan(oof).any(), "some rows got no OOF prediction"
    Xf = np.hstack([Xtr_c, oof if EMBED else oof[:, None]])

    # --- eval caches: champion feats, and nn-prob from a net trained on ALL fit series ---
    tf = time.time()
    champ_caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]
    pair_caches = [precompute_fold(pair_vec, WINDOW, X, f) for f in folds]
    Xtr_p_all, ytr_p_all = build_training_pairs(X, yi, picks_n, WINDOW)
    final_nets = train_ensemble(Xtr_p_all, ytr_p_all, ENSEMBLE, NN_EPOCHS, SEED, pre_state)
    proba = EnsembleProba(final_nets)
    extra = 64 if EMBED else 1
    fused_caches = []
    for cc, pc in zip(champ_caches, pair_caches):
        assert len(cc["feat"]) == len(pc["feat"]), "champ/pair cache row mismatch"
        if len(cc["feat"]):
            add = proba.predict_embed(pc["feat"]) if EMBED \
                else proba.predict_proba(pc["feat"])[:, 1][:, None]
            feat = np.hstack([cc["feat"], add])
        else:
            feat = np.zeros((0, cc["feat"].shape[1] + extra))
        fused_caches.append({"index": cc["index"], "feat": feat, "active": cc["active"]})
    print(f"eval caches + final net in {(time.time() - tf) / 60:.1f} min", flush=True)

    # --- paired comparison ---
    base_aucs, _ = eval_model(hgb(), Xtr_c, ytr, champ_caches, y_val)
    fused_aucs, _ = eval_model(hgb(), Xf, ytr, fused_caches, y_val)
    report("champion (45)", base_aucs, None)
    report(f"champion + nn ({'embed64' if EMBED else 'prob'})", fused_aucs, base_aucs)
    md, sd, t, _ = paired(fused_aucs, base_aucs)
    print(f"\nchampion={base_aucs.mean():.4f} fused={fused_aucs.mean():.4f} "
          f"delta={md:+.4f} t={t:+.2f}", flush=True)
    print(f"=== DONE in {(time.time() - t0) / 60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
