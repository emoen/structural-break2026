"""CAUSAL per-step gate — the deployable form of the gated 3-member blend.

_gru_gate.py measured a real +0.0044 gain but its separators (onl_len, onl_std, gru/agreement
stats over the WHOLE series, AR over the full online window) use FUTURE points. The task reveals
x_online ONE POINT AT A TIME (infer yields one score per online point), so scoring step t may use
only x_online[:t+1]. That per-series gate is LOOKAHEAD -> not deployable.

The 3 members are already causal per step (GRU reads the online feature vectors left-to-right with
hidden state; percentiles use the fixed TRAIN CDF). Only the gate's FEATURES leaked. This harness
fixes that: a PER-STEP gate w_gru(t) in {0, W3_gru} driven by EXPANDING (causal) separators, so the
decision at step t uses only steps <= t. Target = the per-STEP concordant-pair marginal (does adding
the GRU help the cross-sectional AUC AT step t, for this series). Cross-fold honest gate over rows,
measured as REAL TS-AUC vs the shipped 2-member champion.

HONEST RISK: onl_len (the #1 separator in the diagnostic, Spearman +0.143) is inherently non-causal
(total length is unknown mid-stream; the causal proxy c_nseen = current online index cannot tell a
short series from a long one at the same step). So this gate loses the strongest signal and must
lean on member-disagreement + GRU-confidence (weaker). It may come out smaller or fail — but it is
the only version that can actually ship.

Gate: causal gate positive on ALL seeds AND mean_d>=0.0015 -> build the deployable notebook.

Usage:
  SMOKE=1 .venv/bin/python -u _gru_gate_causal.py
  nohup env PYTHONWARNINGS=ignore DIAG_SEEDS=42,7,31337,2025,555,99 .venv/bin/python -u \
      _gru_gate_causal.py > _gru_gate_causal.log 2>&1 & disown
"""
import os
import time
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_SMOKE = os.environ.get("SMOKE") == "1"
os.environ.setdefault("N_GRU", "1" if _SMOKE else "3")
os.environ.setdefault("HID", "32" if _SMOKE else "128")
os.environ.setdefault("EPOCHS", "3" if _SMOKE else "18")
os.environ.setdefault("DROPOUT", "0.2")
os.environ.setdefault("WD", "1e-5")
os.environ.setdefault("LR", "1e-3")
os.environ.setdefault("LAYERS", "1")

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import _ensemble_tune as ET
from _ensemble_tune import (
    champion_feat, load_pool_strat, make_hgb,
    precompute_fold, score_fold, select_picks, WINDOW,
)
from _lambdarank import build_grouped, sort_into_groups, fit_ranker
from _overnight import build_training, MIN_N, paired

import _seqtraj
from _seqtraj import build_seqs, standardize_fit, gru_scores
from _gru_strong import TUNED, train_one, active_probs, pct, N_GRU, HID, EPOCHS
from _gru_diag import marginal_delta_num
from _gru_gate import ts_auc_df

DIAG_SEEDS = [int(s) for s in os.environ.get("DIAG_SEEDS", "42,7").split(",") if s.strip()]
DENSE_K = 30 if _SMOKE else 60
W2 = (0.6, 0.4, 0.0)
W3 = (0.36, 0.34, 0.30)
EPS = 1e-9
THRESHOLDS = (0.50, 0.55, 0.60, 0.65)
GATE_FIT_CAP = 250_000       # cap rows for gate fit (speed)

CAUSAL_FEATS = [
    "c_nseen", "c_gb_mean", "c_gb_std", "c_gru_mean", "c_gru_std", "c_gru_max", "c_gru_cur",
    "c_gru_ramp", "c_gru_entropy", "c_gru_conf", "c_bias_gru_gb", "c_div_gru_gb",
    "c_corr_gru_gb", "c_bias_gru_rank",
]


def _expanding_seps(sub):
    """Vectorized causal (expanding) separators for one series' scored rows (time-ordered)."""
    fgb = sub["F_gb"].to_numpy(); frank = sub["F_rank"].to_numpy()
    fgru = sub["F_gru"].to_numpy(); g = sub["gru_raw"].to_numpy()
    n = len(sub)
    k = np.arange(1, n + 1, dtype=np.float64)

    def emean(x):
        return np.cumsum(x) / k

    def estd(x):
        m = emean(x)
        return np.sqrt(np.clip(np.cumsum(x * x) / k - m * m, 0.0, None))

    ent = -g * np.log(g + EPS) - (1 - g) * np.log(1 - g + EPS)
    dgg = fgru - fgb
    dgr = fgru - frank
    Sa = np.cumsum(fgru); Sb = np.cumsum(fgb)
    Saa = np.cumsum(fgru * fgru); Sbb = np.cumsum(fgb * fgb); Sab = np.cumsum(fgru * fgb)
    ma = Sa / k; mb = Sb / k
    cov = Sab / k - ma * mb
    va = np.clip(Saa / k - ma * ma, 0.0, None)
    vb = np.clip(Sbb / k - mb * mb, 0.0, None)
    corr = np.where((va > EPS) & (vb > EPS), cov / np.sqrt(va * vb + EPS), 0.0)
    return {
        "c_nseen": sub["time_online"].to_numpy().astype(np.float64),
        "c_gb_mean": emean(fgb), "c_gb_std": estd(fgb),
        "c_gru_mean": emean(g), "c_gru_std": estd(g),
        "c_gru_max": np.maximum.accumulate(g), "c_gru_cur": g.astype(np.float64),
        "c_gru_ramp": g - g[0],
        "c_gru_entropy": emean(ent), "c_gru_conf": emean(np.abs(g - 0.5)),
        "c_bias_gru_gb": emean(dgg), "c_div_gru_gb": emean(np.abs(dgg)),
        "c_corr_gru_gb": corr, "c_bias_gru_rank": emean(dgr),
    }


def build_seed_causal(seed):
    """Per-fold df with per-row marginal delta AND causal expanding separators."""
    t0 = time.time()
    ET.SEED = seed
    X, yi, y_val, train_ids, folds = load_pool_strat()
    caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]

    picks = select_picks(X, train_ids)
    Xtr_c, ytr_c = build_training(X, yi, picks, champion_feat, WINDOW)
    model_gb = make_hgb(**TUNED, random_state=seed)
    with threadpool_limits(limits=1):
        model_gb.fit(Xtr_c, ytr_c)
    ref_gb = np.sort(model_gb.predict_proba(Xtr_c)[:, 1])
    gb_series = [score_fold(model_gb, c) for c in caches]

    rng = np.random.default_rng(seed + 202)
    rd, ld, td = build_grouped(X, yi, train_ids, champion_feat, WINDOW, DENSE_K, rng)
    rr, lr, gr = sort_into_groups(rd, ld, td)
    ranker = fit_ranker("rank_xendcg", rr, lr, gr)
    ref_rank = np.sort(ranker.r.predict(rr))
    rank_series = [score_fold(ranker, c) for c in caches]

    tr_ids = train_ids[:_seqtraj.SEQ_N_TRAIN]
    train_seqs = build_seqs(X, yi, tr_ids)
    fold_seqs = [build_seqs(X, yi, f) for f in folds]
    del X
    mu, sd = standardize_fit(train_seqs)
    print(f"[seed {seed}] gb/rank/seqs ready, prep {time.time()-t0:.0f}s", flush=True)

    models = []
    for gi in range(N_GRU):
        tg = time.time()
        models.append(train_one(train_seqs, mu, sd, seed * 100 + gi))
        print(f"  gru {gi+1}/{N_GRU} trained ({time.time()-tg:.0f}s)", flush=True)
    ref_gru = np.sort(np.mean([active_probs(m, train_seqs, mu, sd) for m in models], axis=0))
    gru_series = []
    for fs, c in zip(fold_seqs, caches):
        ens = sum(gru_scores(m, fs, mu, sd).reindex(c["index"]).fillna(0.0) for m in models) / N_GRU
        gru_series.append(ens)

    warm = MIN_N - 1
    fold_dfs = []
    for hs, rs_, gs in zip(gb_series, rank_series, gru_series):
        fgb = pct(hs.to_numpy(), ref_gb)
        frank = pct(rs_.to_numpy(), ref_rank)
        fgru = pct(gs.to_numpy(), ref_gru)
        df = pd.DataFrame({"F_gb": fgb, "F_rank": frank, "F_gru": fgru, "gru_raw": gs.to_numpy(),
                           "b2": W2[0] * fgb + W2[1] * frank,
                           "b3": W3[0] * fgb + W3[1] * frank + W3[2] * fgru}, index=hs.index)
        df = df.join(y_val, how="left").reset_index()
        df["time_online"] = df.groupby("id").cumcount()

        # per-ROW marginal delta (only scored steps are non-zero)
        delta = np.zeros(len(df))
        for k, grp in df.groupby("time_online"):
            lab = grp["target"].to_numpy()
            npos = int(lab.sum()); nneg = int(len(lab) - npos)
            if npos == 0 or nneg == 0 or k < warm:
                continue
            delta[grp.index.to_numpy()] = marginal_delta_num(lab, grp["b2"].to_numpy(),
                                                             grp["b3"].to_numpy())
        df["delta"] = delta

        # causal expanding separators over scored rows, per series
        for c in CAUSAL_FEATS:
            df[c] = 0.0
        scored_mask = df["time_online"] >= warm
        scored = df[scored_mask]
        for sid, sub in scored.groupby("id"):
            sub = sub.sort_values("time_online")
            seps = _expanding_seps(sub)
            for c, vals in seps.items():
                df.loc[sub.index, c] = vals
        df["_scored"] = scored_mask.to_numpy()
        fold_dfs.append(df)

    del caches, gb_series, rank_series, gru_series, Xtr_c, ytr_c, rd, ld, td
    del train_seqs, fold_seqs, models, model_gb, ranker
    print(f"[seed {seed}] scored {len(folds)} folds, {time.time()-t0:.0f}s", flush=True)
    return fold_dfs


def eval_causal(fold_dfs, seed):
    n = len(fold_dfs)
    base, allon = [], []
    gated = {thr: [] for thr in THRESHOLDS}
    oof = []
    rng = np.random.default_rng(0)
    for fi in range(n):
        tr_rows = pd.concat([fold_dfs[j][fold_dfs[j]["_scored"]] for j in range(n) if j != fi],
                            ignore_index=True)
        if len(tr_rows) > GATE_FIT_CAP:
            tr_rows = tr_rows.iloc[rng.choice(len(tr_rows), GATE_FIT_CAP, replace=False)]
        ytr = (tr_rows["delta"].to_numpy() > 0).astype(int)
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                             l2_regularization=1.0, random_state=0)
        clf.fit(tr_rows[CAUSAL_FEATS].to_numpy(dtype=float), ytr)

        df = fold_dfs[fi].copy()
        sc = df[df["_scored"]]
        prob = np.zeros(len(df))
        p_sc = clf.predict_proba(sc[CAUSAL_FEATS].to_numpy(dtype=float))[:, 1]
        prob[sc.index.to_numpy()] = p_sc
        yv = (sc["delta"].to_numpy() > 0).astype(int)
        if yv.min() != yv.max():
            oof.append(roc_auc_score(yv, p_sc))
        base.append(ts_auc_df(df, "b2"))
        allon.append(ts_auc_df(df, "b3"))
        b2 = df["b2"].to_numpy(); b3 = df["b3"].to_numpy()
        for thr in THRESHOLDS:
            df["_bg"] = np.where(prob >= thr, b3, b2)
            gated[thr].append(ts_auc_df(df, "_bg"))
    base = np.asarray(base); allon = np.asarray(allon)
    md_a, _s, t_a, _ = paired(allon, base)
    print(f"\n[seed {seed}] 2-member base={base.mean():.4f}  all-on 3-member d={md_a:+.4f} "
          f"t={t_a:+.2f}  causal-gate OOF-AUC={np.mean(oof):.3f}", flush=True)
    res = {}
    for thr in THRESHOLDS:
        g = np.asarray(gated[thr])
        md, _s, t, _ = paired(g, base)
        res[thr] = md
        tag = "SHIP" if (md >= 0.0015 and t >= 2.0) else ("+" if md > 0 else "-")
        print(f"    thr>={thr:.2f}  gated={g.mean():.4f}  d={md:+.4f} t={t:+.2f} {tag}", flush=True)
    return md_a, res


def main():
    print(f"=== CAUSAL per-step gate {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    print(f"DIAG_SEEDS={DIAG_SEEDS} N_GRU={N_GRU} HID={HID} EPOCHS={EPOCHS} W2={W2} W3={W3} "
          f"thr={THRESHOLDS} feats={len(CAUSAL_FEATS)}", flush=True)
    allon_d = []
    store = {thr: [] for thr in THRESHOLDS}
    for seed in DIAG_SEEDS:
        fold_dfs = build_seed_causal(seed)
        md_a, res = eval_causal(fold_dfs, seed)
        allon_d.append(md_a)
        for thr in THRESHOLDS:
            store[thr].append(res[thr])

    print("\n=== SUMMARY (real causal-gate TS-AUC vs shipped 2-member, mean over seeds) ===",
          flush=True)
    aa = np.asarray(allon_d)
    print(f"all-on 3-member (ref): mean_d={aa.mean():+.4f} min_d={aa.min():+.4f} "
          f"pos_all={'YES' if (aa>0).all() else 'no'}", flush=True)
    print(f"{'thr':<8} {'mean_d':>8} {'min_d':>8}  pos_all  verdict", flush=True)
    for thr in THRESHOLDS:
        d = np.asarray(store[thr])
        pos = (d > 0).all()
        v = "CONFIRM-candidate" if (pos and d.mean() >= 0.0015) else ("promising" if pos else "noise")
        print(f"{thr:<8.2f} {d.mean():+8.4f} {d.min():+8.4f}   {'YES' if pos else ' no'}    {v}",
              flush=True)
    print("\nA causal threshold positive on ALL seeds with mean_d>=0.0015 is DEPLOYABLE -> build the "
          "3-member gated notebook (streaming GRU infer + per-step expanding gate).", flush=True)


if __name__ == "__main__":
    main()
