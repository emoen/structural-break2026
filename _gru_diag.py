"""GRU gating diagnostic — is the GRU's value PER-SERIES heterogeneous, and is it PREDICTABLE
from label-free signals? (E44/E46 follow-up.)

The GRU won on seed 42's pool and lost on seed 7's pool from the SAME dataset. Hypothesis
(user's): the GRU adds real signal on SOME series and noise on OTHERS, and the fixed GLOBAL
blend weight can't localize it — so whichever val draw has more "GRU-helps" series wins.

KEY MATH that makes this measurable exactly:
  TS-AUC = sum_over_time_online( n_pos*n_neg * roc_auc ) / sum_over_time_online( n_pos*n_neg ).
  The weighted numerator at each step is exactly the count of CONCORDANT pos/neg pairs
  (+0.5 per tie). So each series' MARGINAL value from adding the GRU is EXACT: swap ONLY that
  series from the shipped 2-member blend (0.6/0.4) to the 3-member blend (0.36/0.34/0.30),
  hold everyone else at 2-member, and recount just the concordant pairs that involve it
  (searchsorted on the opposite group's 2-member scores). Each val series lives in exactly one
  fold -> one clean delta_i per series, in fold-AUC units. delta_i > 0 => GRU helps that series.

Then we test whether delta_i is PREDICTABLE from ~30 strictly label-free per-series separators:
  - scale / RevIN-relevant : lengths, std, std-ratio hist->online, mean-shift, cv, slope,
      rolling-mean variance, lag-1 autocorr, diff-std   (these scope the RevIN lever)
  - GRU-internal           : mean/std/max/final prob, entropy, confidence, ramp, monotonicity
  - member agreement       : corr(GRU,gb), corr(GRU,rank), |F_gru-F_gb|, systematic bias, gb stats
  - AR cross-forecast      : fit AR(p) on history, roll-forecast the online window, residual RMSE
      + blow-up ratio online/hist   (these scope the AR cross-forecast feature lever)

Outputs:
  1. heterogeneity: fraction of series where GRU helps, sum of positive vs negative delta.
  2. ORACLE gate ceiling: net delta if we could zero the GRU on every delta_i<0 series.
  3. separator power: Spearman(sep, delta_i) and sign-AUC per separator, ranked.
  4. REALIZABLE gate: 5-fold CV shallow model on separators predicting delta_i>0; out-of-fold
     gate delta vs the oracle ceiling. If realizable/oracle is decent -> a constrained gate is
     worth building (then fresh-seed confirm). If not -> the GRU is genuinely dead.

This is a DIAGNOSTIC, not a ship candidate: it uses labels to compute delta_i (the target);
the separators are label-free so any gate it motivates is deployable.

Usage (ONE 15-job harness at a time; also holds torch GRUs):
  SMOKE=1 .venv/bin/python -u _gru_diag.py
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _gru_diag.py > _gru_diag.log 2>&1 & disown
  knobs: DIAG_SEEDS (default 42), plus GRU config inherited from E46 (N_GRU=3 HID=128 EPOCHS=18)
"""
import os
import time
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_SMOKE = os.environ.get("SMOKE") == "1"
# Pin the GRU member to the E46 config BEFORE importing _gru_strong (its constants read env
# at import time). Smoke shrinks everything for a plumbing check.
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
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import _ensemble_tune as ET
from _ensemble_tune import (
    champion_feat, load_pool_strat, make_hgb,
    precompute_fold, score_fold, select_picks, WINDOW, N_FOLDS,
)
from _exp_fast import split_series, hist_context, HIST_CAP
from _lambdarank import build_grouped, sort_into_groups, fit_ranker
from _overnight import build_training, MIN_N, N_JOBS

import _seqtraj
from _seqtraj import build_seqs, standardize_fit, gru_scores
# reuse the EXACT E46 GRU member (config pinned via env above)
from _gru_strong import TUNED, train_one, active_probs, pct, N_GRU, HID, EPOCHS

DIAG_SEEDS = [int(s) for s in os.environ.get("DIAG_SEEDS", "42").split(",") if s.strip()]
DENSE_K = 30 if _SMOKE else 60
W2 = (0.6, 0.4, 0.0)            # shipped 2-member champion
W3 = (0.36, 0.34, 0.30)        # best E46 3-member weights (seed 42 d+0.0058)
EPS = 1e-9
RES_TSV = "_gru_diag_results.tsv"


# ---------------------------------------------------------------- separators (x-based) -----
def _ar_resid(series, coef, p):
    """One-step residual std of AR(p) with fixed coef over `series` (uses only true past)."""
    n = len(series)
    if n <= p:
        return np.nan
    res = []
    for t in range(p, n):
        lags = series[t - p:t][::-1]        # series[t-1], series[t-2], ..., series[t-p]
        pred = coef[0] + np.dot(coef[1:], lags)
        res.append(series[t] - pred)
    return float(np.std(res)) if res else np.nan


def x_separators(sid, x_h, x_on, p=5):
    """Strictly label-free per-series features from raw history + online values."""
    d = {"sid": sid}
    h = np.asarray(x_h, dtype=np.float64)
    o = np.asarray(x_on, dtype=np.float64)
    hstd = float(h.std()) if h.size else 0.0
    ostd = float(o.std()) if o.size else 0.0
    hmean = float(h.mean()) if h.size else 0.0
    omean = float(o.mean()) if o.size else 0.0
    d["hist_len"] = float(h.size)
    d["onl_len"] = float(o.size)
    d["onl_std"] = ostd
    d["hist_std"] = hstd
    d["std_ratio"] = ostd / (hstd + EPS)
    d["mean_shift"] = abs(omean - hmean) / (hstd + EPS)
    d["cv"] = ostd / (abs(omean) + EPS)
    d["onl_absmed"] = float(np.median(np.abs(o))) if o.size else 0.0
    # trend / nonstationarity of the online window
    if o.size >= 3:
        t = np.arange(o.size)
        slope = float(np.polyfit(t, o, 1)[0])
        d["onl_slope"] = slope * o.size / (ostd + EPS)
        d["diff_std"] = float(np.std(np.diff(o))) / (ostd + EPS)
        oc = o - o.mean()
        d["lag1_ac"] = float(np.dot(oc[:-1], oc[1:]) / (np.dot(oc, oc) + EPS))
        # rolling-mean variance (regime drift): 20-pt blocks
        k = max(5, o.size // 10)
        means = [o[i:i + k].mean() for i in range(0, o.size - k + 1, k)]
        d["rollmean_var"] = float(np.var(means)) / (ostd ** 2 + EPS) if len(means) > 1 else 0.0
    else:
        d.update(onl_slope=0.0, diff_std=0.0, lag1_ac=0.0, rollmean_var=0.0)
    # AR(p) cross-forecast: fit on history, apply to online (real-time legal)
    d["ar_hist_resid"] = np.nan
    d["ar_onl_resid"] = np.nan
    d["ar_blowup"] = np.nan
    if h.size > p + 5:
        rows, targ = [], []
        for t in range(p, h.size):
            rows.append(np.concatenate([[1.0], h[t - p:t][::-1]]))
            targ.append(h[t])
        A = np.asarray(rows); b = np.asarray(targ)
        try:
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            hist_res = _ar_resid(h, coef, p)
            stream = np.concatenate([h[-p:], o]) if o.size else h[-p:]
            onl_res = _ar_resid(stream, coef, p)
            d["ar_hist_resid"] = hist_res
            d["ar_onl_resid"] = onl_res
            if hist_res and hist_res > EPS and onl_res == onl_res:
                d["ar_blowup"] = onl_res / hist_res
        except Exception:
            pass
    return d


# ---------------------------------------------------------------- marginal attribution -----
def _conc_below(sorted_arr, v):
    lo = np.searchsorted(sorted_arr, v, side="left")
    hi = np.searchsorted(sorted_arr, v, side="right")
    return lo + 0.5 * (hi - lo)


def _conc_above(sorted_arr, v, n):
    lo = np.searchsorted(sorted_arr, v, side="left")
    hi = np.searchsorted(sorted_arr, v, side="right")
    return (n - hi) + 0.5 * (hi - lo)


def marginal_delta_num(labels, b2, b3):
    """Per-row change in the TS-AUC concordant-pair numerator from swapping ONLY that row from
    the 2-member score b2 to the 3-member score b3 (everyone else stays at b2)."""
    labels = labels.astype(bool)
    d = np.zeros(len(labels))
    pos = labels
    neg = ~labels
    npos = int(pos.sum()); nneg = int(neg.sum())
    if npos == 0 or nneg == 0:
        return d
    b2n = np.sort(b2[neg])
    b2p = np.sort(b2[pos])
    # pos rows: concordant with negatives = #neg below its score
    d[pos] = _conc_below(b2n, b3[pos]) - _conc_below(b2n, b2[pos])
    # neg rows: concordant = #pos above its score
    d[neg] = _conc_above(b2p, b3[neg], npos) - _conc_above(b2p, b2[neg], npos)
    return d


def _corr(a, b):
    if a.size < 3 or a.std() < EPS or b.std() < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------- main -----
def run_seed(seed):
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

    # label-free x-separators for every val series (BEFORE freeing X)
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
    print(f"[seed {seed}] gb/rank/seqs ready, x-separators={len(xsep)}, prep {time.time()-t0:.0f}s",
          flush=True)

    models = []
    for g in range(N_GRU):
        tg = time.time()
        models.append(train_one(train_seqs, mu, sd, seed * 100 + g))
        print(f"  gru {g+1}/{N_GRU} trained ({time.time()-tg:.0f}s)", flush=True)
    ref_gru = np.sort(np.mean([active_probs(m, train_seqs, mu, sd) for m in models], axis=0))
    gru_series = []
    for fs, c in zip(fold_seqs, caches):
        ens = sum(gru_scores(m, fs, mu, sd).reindex(c["index"]).fillna(0.0)
                  for m in models) / N_GRU
        gru_series.append(ens)

    warm = MIN_N - 1
    rows = []
    fold_true_delta = []
    for fi, (hs, rs_, gs) in enumerate(zip(gb_series, rank_series, gru_series)):
        fgb = pct(hs.to_numpy(), ref_gb)
        frank = pct(rs_.to_numpy(), ref_rank)
        fgru = pct(gs.to_numpy(), ref_gru)
        b2 = W2[0] * fgb + W2[1] * frank
        b3 = W3[0] * fgb + W3[1] * frank + W3[2] * fgru
        df = pd.DataFrame({"F_gb": fgb, "F_rank": frank, "F_gru": fgru,
                           "gru_raw": gs.to_numpy(), "b2": b2, "b3": b3}, index=hs.index)
        df = df.join(y_val, how="left")            # adds 'target'
        df = df.reset_index()                       # columns: id, time, ...
        df["time_online"] = df.groupby("id").cumcount()

        # denom over ALL steps (matches ts_auc_from), numerator over SCORED steps only
        denom = 0.0
        num2 = num3 = 0.0
        delta_num = {}
        for k, grp in df.groupby("time_online"):
            lab = grp["target"].to_numpy()
            npos = int(lab.sum()); nneg = int((1 - lab).sum())
            if npos == 0 or nneg == 0:
                continue
            denom += npos * nneg
            if k < warm:
                continue
            b2v = grp["b2"].to_numpy(); b3v = grp["b3"].to_numpy()
            num2 += float(roc_auc_score(lab, b2v)) * npos * nneg
            num3 += float(roc_auc_score(lab, b3v)) * npos * nneg
            dnum = marginal_delta_num(lab, b2v, b3v)
            for sid, dv in zip(grp["id"].to_numpy(), dnum):
                delta_num[sid] = delta_num.get(sid, 0.0) + dv
        fold_true_delta.append((num3 - num2) / denom if denom > 0 else 0.0)

        # per-series model separators (scored steps only)
        scored = df[df["time_online"] >= warm]
        for sid, g in scored.groupby("id"):
            gp = g["gru_raw"].to_numpy()
            fgb_s = g["F_gb"].to_numpy(); frank_s = g["F_rank"].to_numpy(); fgru_s = g["F_gru"].to_numpy()
            rec = dict(xsep.get(sid, {"sid": sid}))
            rec.update(
                seed=seed, fold=fi, sid=sid,
                delta_i=delta_num.get(sid, 0.0) / denom if denom > 0 else 0.0,
                gru_mean=float(gp.mean()), gru_std=float(gp.std()), gru_max=float(gp.max()),
                gru_final=float(gp[-1]),
                gru_entropy=float(np.mean(-gp * np.log(gp + EPS) - (1 - gp) * np.log(1 - gp + EPS))),
                gru_conf=float(np.mean(np.abs(gp - 0.5))),
                gru_ramp=float(gp[-1] - gp[0]),
                gru_mono=float(np.mean(np.diff(gp) > 0)) if gp.size > 1 else 0.0,
                corr_gru_gb=_corr(fgru_s, fgb_s), corr_gru_rank=_corr(fgru_s, frank_s),
                div_gru_gb=float(np.mean(np.abs(fgru_s - fgb_s))),
                bias_gru_gb=float(fgru_s.mean() - fgb_s.mean()),
                gb_mean=float(fgb_s.mean()), gb_std=float(fgb_s.std()),
            )
            rows.append(rec)

    del caches, gb_series, rank_series, gru_series, Xtr_c, ytr_c, rd, ld, td
    del train_seqs, fold_seqs, models, model_gb, ranker
    print(f"[seed {seed}] attributed {len(rows)} series, "
          f"fold-mean true delta={np.mean(fold_true_delta):+.4f} "
          f"(E46 seed-42 ref +0.0058), {time.time()-t0:.0f}s", flush=True)
    return rows


def analyze(df):
    delta = df["delta_i"].to_numpy()
    N = len(delta)
    pos = delta > 0
    sum_pos = delta[pos].sum()
    sum_neg = delta[~pos].sum()
    print("\n=== HETEROGENEITY ===", flush=True)
    print(f"series={N}  GRU-helps={pos.mean()*100:.1f}%  "
          f"sum(+)={sum_pos/N_FOLDS:+.4f}  sum(-)={sum_neg/N_FOLDS:+.4f}  "
          f"net(all-on)={delta.sum()/N_FOLDS:+.4f}  (units = E46 per-fold-mean delta)", flush=True)
    qs = np.quantile(delta, [0.05, 0.25, 0.5, 0.75, 0.95]) / N_FOLDS
    print(f"delta_i quantiles /fold: p05={qs[0]:+.5f} p25={qs[1]:+.5f} med={qs[2]:+.5f} "
          f"p75={qs[3]:+.5f} p95={qs[4]:+.5f}", flush=True)
    print(f"ORACLE ceiling (GRU on only where delta_i>0) = {sum_pos/N_FOLDS:+.4f}", flush=True)

    sep_cols = [c for c in df.columns if c not in
                ("sid", "seed", "fold", "delta_i")]
    print("\n=== SEPARATOR POWER (ranked by |Spearman| with delta_i) ===", flush=True)
    y_sign = (delta > 0).astype(int)
    scored = []
    for c in sep_cols:
        v = df[c].to_numpy(dtype=float)
        m = np.isfinite(v)
        if m.sum() < 50 or np.nanstd(v[m]) < EPS:
            continue
        rho = spearmanr(v[m], delta[m]).correlation
        try:
            auc = roc_auc_score(y_sign[m], v[m])
            auc = max(auc, 1 - auc)
        except Exception:
            auc = np.nan
        scored.append((c, rho, auc, m.mean()))
    scored.sort(key=lambda r: -abs(r[1]) if r[1] == r[1] else 0)
    print(f"{'separator':<16} {'spearman':>9} {'sign_auc':>9} {'coverage':>9}", flush=True)
    for c, rho, auc, cov in scored:
        print(f"{c:<16} {rho:+9.3f} {auc:9.3f} {cov*100:8.0f}%", flush=True)

    # realizable gate: 5-fold CV shallow model on separators -> predict delta_i>0
    feat_cols = [c for c, *_ in scored]
    Xf = df[feat_cols].to_numpy(dtype=float)
    Xf = np.where(np.isfinite(Xf), Xf, np.nan)
    col_med = np.nanmedian(Xf, axis=0)
    inds = np.where(np.isnan(Xf))
    Xf[inds] = np.take(col_med, inds[1])
    y = (delta > 0).astype(int)
    oof = np.zeros(N)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in kf.split(Xf):
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                             l2_regularization=1.0, random_state=0)
        clf.fit(Xf[tr], y[tr])
        oof[te] = clf.predict_proba(Xf[te])[:, 1]
    gate_auc = roc_auc_score(y, oof)
    print("\n=== REALIZABLE GATE (5-fold CV, label-free separators -> predict GRU-helps) ===",
          flush=True)
    print(f"OOF sign-AUC={gate_auc:.3f}", flush=True)
    oracle = sum_pos / N_FOLDS
    allon = delta.sum() / N_FOLDS
    print(f"{'strategy':<26} {'net_delta/fold':>14}  {'vs oracle':>10}", flush=True)
    print(f"{'all-on (current blend)':<26} {allon:>+14.4f} {allon/oracle*100 if oracle else 0:9.0f}%",
          flush=True)
    for thr in (0.5, 0.55, 0.6):
        on = oof >= thr
        realized = delta[on].sum() / N_FOLDS
        print(f"{'gate p>='+f'{thr:.2f}':<26} {realized:>+14.4f} "
              f"{realized/oracle*100 if oracle else 0:9.0f}%  (on {on.mean()*100:.0f}% of series)",
              flush=True)
    print(f"{'oracle (delta_i>0)':<26} {oracle:>+14.4f} {'100%':>10}", flush=True)
    print("\nIf a gate captures a meaningful fraction of the oracle AND stays positive vs all-on,"
          " a constrained gate is worth building -> then fresh-seed confirm. Else GRU is dead.",
          flush=True)


def main():
    print(f"=== GRU gating diagnostic {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    print(f"DIAG_SEEDS={DIAG_SEEDS} N_GRU={N_GRU} HID={HID} EPOCHS={EPOCHS} W2={W2} W3={W3} "
          f"dense_k={DENSE_K}", flush=True)
    all_rows = []
    for seed in DIAG_SEEDS:
        all_rows.extend(run_seed(seed))
    df = pd.DataFrame(all_rows)
    df.to_csv(RES_TSV, sep="\t", index=False)
    print(f"\nwrote {len(df)} rows -> {RES_TSV}", flush=True)
    analyze(df)


if __name__ == "__main__":
    main()
