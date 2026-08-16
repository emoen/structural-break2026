"""Gated 3-member blend — turn the GRU ON only where a label-free gate predicts it helps.

The diagnostic (_gru_diag.py) proved the GRU's value is PER-SERIES heterogeneous: it helps ~58%
of series and hurts the rest, and the two halves cancel so a FIXED global weight nets ~0. A shallow
model on label-free separators predicted "GRU-helps" at OOF AUC 0.687 ACROSS both disagreeing pools.
That +0.0127 was in attribution units (marginal sum overcounts pair interactions) -> optimistic.

This harness measures the REAL, interaction-free gain of a deployable gate:
  - per fold: F_gb, F_rank, F_gru percentiles (fixed-CDF), 2-member score b2=0.6/0.4, 3-member
    score b3=0.36/0.34/0.30, per-series separators, and the EXACT per-series delta_i (concordant-
    pair marginal) used only as the gate's training target.
  - CROSS-FOLD gate (honest / deployable proxy): for each held-out fold, train the gate on the
    OTHER 9 folds' (separators -> delta_i>0), predict on the held-out fold -> per-series ON/OFF.
    ON series use the 3-member blend b3, OFF series fall back to the shipped 2-member b2.
  - measure REAL TS-AUC of the gated blend vs the shipped 2-member on that fold. paired-t across
    folds. This has NO interaction inflation (all gated series are swapped simultaneously).

Two gates reported: FULL (all separators, shallow HGB) and SIMPLE (top separators, logistic) —
the simple one is likelier to survive fresh seeds. Threshold swept.

Gate: gated blend positive on ALL seeds AND mean_d>=0.0015 -> fresh-seed confirm [31337,2025,555,99].

Usage (ONE 15-job harness; holds torch GRUs):
  SMOKE=1 .venv/bin/python -u _gru_gate.py
  nohup env PYTHONWARNINGS=ignore DIAG_SEEDS=42,7 .venv/bin/python -u _gru_gate.py > _gru_gate.log 2>&1 & disown
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
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import _ensemble_tune as ET
from _ensemble_tune import (
    champion_feat, load_pool_strat, make_hgb,
    precompute_fold, score_fold, select_picks, WINDOW,
)
from _exp_fast import split_series
from _lambdarank import build_grouped, sort_into_groups, fit_ranker
from _overnight import build_training, MIN_N, N_JOBS, paired

import _seqtraj
from _seqtraj import build_seqs, standardize_fit, gru_scores
from _gru_strong import TUNED, train_one, active_probs, pct, N_GRU, HID, EPOCHS
from _gru_diag import x_separators, marginal_delta_num, _corr

DIAG_SEEDS = [int(s) for s in os.environ.get("DIAG_SEEDS", "42,7").split(",") if s.strip()]
DENSE_K = 30 if _SMOKE else 60
W2 = (0.6, 0.4, 0.0)
W3 = (0.36, 0.34, 0.30)
EPS = 1e-9
THRESHOLDS = (0.50, 0.55, 0.60)

FEATS = [
    "hist_len", "onl_len", "onl_std", "hist_std", "std_ratio", "mean_shift", "cv", "onl_absmed",
    "onl_slope", "diff_std", "lag1_ac", "rollmean_var", "ar_hist_resid", "ar_onl_resid", "ar_blowup",
    "gru_mean", "gru_std", "gru_max", "gru_final", "gru_entropy", "gru_conf", "gru_ramp", "gru_mono",
    "corr_gru_gb", "corr_gru_rank", "div_gru_gb", "bias_gru_gb", "gb_mean", "gb_std",
]
SIMPLE = ["onl_len", "bias_gru_gb", "gb_mean", "rollmean_var", "div_gru_gb"]


def ts_auc_df(df, col):
    wsum = tot = 0.0
    for _, g in df.groupby("time_online"):
        lab = g["target"].to_numpy()
        npos = int(lab.sum()); nneg = int(len(lab) - npos)
        if npos == 0 or nneg == 0:
            continue
        wsum += npos * nneg * float(roc_auc_score(lab, g[col].to_numpy()))
        tot += npos * nneg
    return wsum / tot if tot > 0 else 0.5


def build_seed(seed):
    """Return per-fold (df, table) — df has per-step blends+target, table has per-series sep+delta_i."""
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

    val_jobs = []
    for f in folds:
        for sid in f:
            x_h, onl = split_series(X.loc[sid])
            val_jobs.append((sid, x_h, onl.to_numpy(dtype=np.float64)))
    with threadpool_limits(limits=1):
        xsep_list = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(x_separators)(sid, x_h, x_on) for (sid, x_h, x_on) in val_jobs)
    xsep = {r["sid"]: r for r in xsep_list}
    del X
    mu, sd = standardize_fit(train_seqs)
    print(f"[seed {seed}] gb/rank/seqs ready, prep {time.time()-t0:.0f}s", flush=True)

    models = []
    for g in range(N_GRU):
        tg = time.time()
        models.append(train_one(train_seqs, mu, sd, seed * 100 + g))
        print(f"  gru {g+1}/{N_GRU} trained ({time.time()-tg:.0f}s)", flush=True)
    ref_gru = np.sort(np.mean([active_probs(m, train_seqs, mu, sd) for m in models], axis=0))
    gru_series = []
    for fs, c in zip(fold_seqs, caches):
        ens = sum(gru_scores(m, fs, mu, sd).reindex(c["index"]).fillna(0.0) for m in models) / N_GRU
        gru_series.append(ens)

    warm = MIN_N - 1
    fold_dfs, fold_tables = [], []
    for hs, rs_, gs in zip(gb_series, rank_series, gru_series):
        fgb = pct(hs.to_numpy(), ref_gb)
        frank = pct(rs_.to_numpy(), ref_rank)
        fgru = pct(gs.to_numpy(), ref_gru)
        df = pd.DataFrame({"F_gb": fgb, "F_rank": frank, "F_gru": fgru, "gru_raw": gs.to_numpy(),
                           "b2": W2[0] * fgb + W2[1] * frank,
                           "b3": W3[0] * fgb + W3[1] * frank + W3[2] * fgru}, index=hs.index)
        df = df.join(y_val, how="left").reset_index()
        df["time_online"] = df.groupby("id").cumcount()

        denom = 0.0
        delta_num = {}
        for k, grp in df.groupby("time_online"):
            lab = grp["target"].to_numpy()
            npos = int(lab.sum()); nneg = int(len(lab) - npos)
            if npos == 0 or nneg == 0:
                continue
            denom += npos * nneg
            if k < warm:
                continue
            dnum = marginal_delta_num(lab, grp["b2"].to_numpy(), grp["b3"].to_numpy())
            for sid, dv in zip(grp["id"].to_numpy(), dnum):
                delta_num[sid] = delta_num.get(sid, 0.0) + dv

        recs = []
        scored = df[df["time_online"] >= warm]
        for sid, g in scored.groupby("id"):
            gp = g["gru_raw"].to_numpy()
            fgb_s = g["F_gb"].to_numpy(); frank_s = g["F_rank"].to_numpy(); fgru_s = g["F_gru"].to_numpy()
            rec = dict(xsep.get(sid, {"sid": sid}))
            rec.update(
                sid=sid, delta_i=delta_num.get(sid, 0.0) / denom if denom > 0 else 0.0,
                gru_mean=float(gp.mean()), gru_std=float(gp.std()), gru_max=float(gp.max()),
                gru_final=float(gp[-1]),
                gru_entropy=float(np.mean(-gp * np.log(gp + EPS) - (1 - gp) * np.log(1 - gp + EPS))),
                gru_conf=float(np.mean(np.abs(gp - 0.5))), gru_ramp=float(gp[-1] - gp[0]),
                gru_mono=float(np.mean(np.diff(gp) > 0)) if gp.size > 1 else 0.0,
                corr_gru_gb=_corr(fgru_s, fgb_s), corr_gru_rank=_corr(fgru_s, frank_s),
                div_gru_gb=float(np.mean(np.abs(fgru_s - fgb_s))),
                bias_gru_gb=float(fgru_s.mean() - fgb_s.mean()),
                gb_mean=float(fgb_s.mean()), gb_std=float(fgb_s.std()),
            )
            recs.append(rec)
        fold_dfs.append(df)
        fold_tables.append(pd.DataFrame(recs))

    del caches, gb_series, rank_series, gru_series, Xtr_c, ytr_c, rd, ld, td
    del train_seqs, fold_seqs, models, model_gb, ranker
    print(f"[seed {seed}] scored {len(folds)} folds, {time.time()-t0:.0f}s", flush=True)
    return fold_dfs, fold_tables


def _fit_gate(train_tab, kind):
    y = (train_tab["delta_i"].to_numpy() > 0).astype(int)
    if kind == "full":
        X = train_tab[FEATS].to_numpy(dtype=float)   # HGB handles NaN natively
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                             l2_regularization=1.0, random_state=0)
        clf.fit(X, y)
        return ("full", clf, None)
    X = train_tab[SIMPLE].to_numpy(dtype=float)
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isfinite(X), X, med)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(X), y)
    return ("simple", clf, (sc, med))


def _gate_prob(fit, tab):
    kind, clf, extra = fit
    if kind == "full":
        return clf.predict_proba(tab[FEATS].to_numpy(dtype=float))[:, 1]
    sc, med = extra
    X = tab[SIMPLE].to_numpy(dtype=float)
    X = np.where(np.isfinite(X), X, med)
    return clf.predict_proba(sc.transform(X))[:, 1]


def eval_gate(fold_dfs, fold_tables, kind):
    """Cross-fold honest gate -> real TS-AUC. Returns per-threshold (gated-vs-2member) arrays."""
    n = len(fold_dfs)
    base, allon = [], []
    gated = {thr: [] for thr in THRESHOLDS}
    aucs_oof = []
    for fi in range(n):
        train_tab = pd.concat([fold_tables[j] for j in range(n) if j != fi], ignore_index=True)
        fit = _fit_gate(train_tab, kind)
        tab = fold_tables[fi]
        prob = _gate_prob(fit, tab)
        y = (tab["delta_i"].to_numpy() > 0).astype(int)
        if y.min() != y.max():
            aucs_oof.append(roc_auc_score(y, prob))
        prob_by_sid = dict(zip(tab["sid"].to_numpy(), prob))
        df = fold_dfs[fi]
        p = df["id"].map(prob_by_sid).to_numpy()
        base.append(ts_auc_df(df, "b2"))
        allon.append(ts_auc_df(df, "b3"))
        for thr in THRESHOLDS:
            on = p >= thr
            df["_bg"] = np.where(on, df["b3"].to_numpy(), df["b2"].to_numpy())
            gated[thr].append(ts_auc_df(df, "_bg"))
    return np.asarray(base), np.asarray(allon), {t: np.asarray(v) for t, v in gated.items()}, \
        float(np.mean(aucs_oof)) if aucs_oof else float("nan")


def main():
    print(f"=== gated 3-member blend {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    print(f"DIAG_SEEDS={DIAG_SEEDS} N_GRU={N_GRU} HID={HID} EPOCHS={EPOCHS} W2={W2} W3={W3} "
          f"thr={THRESHOLDS}", flush=True)
    store = {k: {thr: [] for thr in THRESHOLDS} for k in ("full", "simple")}
    allon_d = []
    for seed in DIAG_SEEDS:
        fold_dfs, fold_tables = build_seed(seed)
        for kind in ("full", "simple"):
            base, allon, gated, oof = eval_gate(fold_dfs, fold_tables, kind)
            if kind == "full":
                md_a, _s, t_a, _ = paired(allon, base)
                allon_d.append(md_a)
                print(f"\n[seed {seed}] 2-member base={base.mean():.4f}  "
                      f"all-on 3-member d={md_a:+.4f} t={t_a:+.2f} (E46 ref)", flush=True)
            print(f"[seed {seed}] gate={kind:<6} OOF-AUC={oof:.3f}", flush=True)
            for thr in THRESHOLDS:
                md, _s, t, _ = paired(gated[thr], base)
                store[kind][thr].append(md)
                tag = "SHIP" if (md >= 0.0015 and t >= 2.0) else ("+" if md > 0 else "-")
                print(f"    thr>={thr:.2f}  gated={gated[thr].mean():.4f}  d={md:+.4f} t={t:+.2f} {tag}",
                      flush=True)

    print("\n=== SUMMARY (real gated TS-AUC vs shipped 2-member, mean over seeds) ===", flush=True)
    print(f"all-on 3-member (reference): mean_d={np.mean(allon_d):+.4f} "
          f"min_d={np.min(allon_d):+.4f} pos_all={'YES' if (np.asarray(allon_d)>0).all() else 'no'}",
          flush=True)
    print(f"{'gate/thr':<16} {'mean_d':>8} {'min_d':>8}  pos_all  verdict", flush=True)
    for kind in ("full", "simple"):
        for thr in THRESHOLDS:
            d = np.asarray(store[kind][thr])
            pos = (d > 0).all()
            v = "CONFIRM-candidate" if (pos and d.mean() >= 0.0015) else ("promising" if pos else "noise")
            print(f"{kind+' '+f'{thr:.2f}':<16} {d.mean():+8.4f} {d.min():+8.4f}   "
                  f"{'YES' if pos else ' no'}    {v}", flush=True)
    print("\nA gate positive on ALL seeds with mean_d>=0.0015 BEATS the fixed blend -> fresh-seed "
          "confirm [31337,2025,555,99], then wire the per-series gate into the deployable notebook.",
          flush=True)


if __name__ == "__main__":
    main()
