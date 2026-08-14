"""Night run: RE-EVALUATE every previously-investigated feature vs the NEW TUNED champion.

Motivation
----------
Feature value is conditional on the model's hyperparameters. All prior feature screening
(E8-E17 battery) was done under the DEFAULT HGB (lr 0.05, 31 leaves, 200 iters, mf 1.0).
Our champion is now the E32 TUNED HGB (lr 0.021, 63 leaves, 400 iters, min_leaf 10,
max_features 0.5). Feature subsampling + a deeper/regularised model can resurrect weak
features that the default model could not exploit. So we re-screen the WHOLE battery from
`_feat_lib.EXTRA`/`COMBOS` on top of the 45-feat multiscale champion, scored with the
tuned params, multi-seed, paired-t vs the tuned champion.

Design (reuses the proven _ensemble_tune harness)
-------------------------------------------------
- Baseline = champion_feat (base15 x scales (60,130,250) = 45 feats) with TUNED params.
- Candidate = champion + one EXTRA group (or COMBO) computed at the SAME 3 scales.
- Incremental columns: the 45-col champion cache (val) + champion train matrix are
  computed ONCE per seed; each candidate only computes its EXTRA columns and hstacks.
  A SMOKE assertion checks hstack(champ, extra) == a full combined feat_fn exactly.
- Multi-seed POOL_SEEDS=[42,7,123,2024] (each re-splits the stratified pool). Screen
  gate = positive on ALL seeds AND mean_d>=0.0015 AND mean_t>=2.5 -> flag for fresh-seed
  confirmation (a separate step; this run only SCREENS).
- Incremental TSV (_feat_reeval_raw.tsv per seed x candidate; survives a crash) + a ranked
  summary (_feat_reeval_summary.txt).

Usage
-----
  SMOKE=1 .venv/bin/python -u _feat_reeval.py            # ~2-3 min plumbing + correctness
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _feat_reeval.py \
       > _feat_reeval.log 2>&1 & disown                   # real night run (~2h)
"""
import os
import time
from datetime import datetime
from functools import partial

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import _ensemble_tune as ET
from _ensemble_tune import (
    WINDOW, SCALES, BASE_KEYS, _subwin, champion_feat,
    load_pool_strat, select_picks, precompute_fold, eval_model, make_hgb,
)
from _overnight import build_training, paired
from _feat_lib import EXTRA, COMBOS

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "_feat_reeval_raw.tsv")
SUMMARY = os.path.join(HERE, "_feat_reeval_summary.txt")

SMOKE = os.environ.get("SMOKE") == "1"

# E32 tuned champion params (LB 57.72%). early_stopping=True comes from HGB_BASE.
TUNED = dict(
    learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
    min_samples_leaf=10, l2_regularization=1.0, max_features=0.5,
)

POOL_SEEDS = [42, 7] if SMOKE else [42, 7, 123, 2024]

# candidate battery: every EXTRA group (except spec_entropy, already in the champion) + combos
SINGLES = sorted(k for k in EXTRA if k not in BASE_KEYS)
COMBO_NAMES = sorted(COMBOS)
if SMOKE:
    SINGLES = SINGLES[:3]
    COMBO_NAMES = COMBO_NAMES[:1]


def cand_keys(name):
    """Return the tuple of EXTRA keys for a candidate name (single group or combo)."""
    return COMBOS[name] if name in COMBOS else (name,)


def extra_feat(w, ctx, keys):
    """EXTRA-only multiscale block: each key at each scale (no base15)."""
    out = []
    for s in SCALES:
        sw = _subwin(w, s)
        for k in keys:
            out.extend(EXTRA[k](sw, ctx))
    return out


def combined_feat(w, ctx, keys):
    """Full champion + extra, for the SMOKE correctness assertion."""
    return champion_feat(w, ctx) + extra_feat(w, ctx, keys)


def hstack_caches(champ_caches, extra_caches):
    out = []
    for ch, ex in zip(champ_caches, extra_caches):
        if len(ch["feat"]) and len(ex["feat"]) == len(ch["feat"]):
            feat = np.hstack([ch["feat"], ex["feat"]])
        else:
            feat = ch["feat"]
        out.append({"index": ch["index"], "active": ch["active"], "feat": feat})
    return out


def screen_verdict(mean_d, mean_t, pos):
    if pos == len(POOL_SEEDS) and mean_d >= 0.0015 and mean_t >= 2.5:
        return "CONFIRM"
    if pos == len(POOL_SEEDS) and mean_d >= 0.0010 and mean_t >= 2.0:
        return "promising"
    if mean_d > 0 and mean_t >= 1.5:
        return "weak"
    return "noise"


def main():
    t0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    cands = SINGLES + COMBO_NAMES
    print(f"=== feature re-eval vs TUNED champion ({mode}) "
          f"{datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    print(f"seeds={POOL_SEEDS}  candidates={len(cands)} "
          f"({len(SINGLES)} singles + {len(COMBO_NAMES)} combos)  scales={SCALES}", flush=True)

    raw = open(RAW, "w", encoding="utf-8")
    raw.write("seed\tcandidate\tn_feat\tchamp_mean\tcand_mean\tdelta\tt\ttime\n")
    raw.flush()

    # per-candidate accumulation across seeds
    acc = {c: {"d": [], "t": [], "cand_mean": []} for c in cands}

    for si, seed in enumerate(POOL_SEEDS):
        ET.SEED = seed  # re-split the stratified pool for this seed
        ts = time.time()
        X, yi, y_val, train_ids, folds = load_pool_strat()
        picks = select_picks(X, train_ids)

        # champion baseline: features once, model once
        Xtr_ch, ytr = build_training(X, yi, picks, champion_feat, WINDOW)
        champ_caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]
        champ_aucs, _ = eval_model(make_hgb(**TUNED, random_state=seed),
                                   Xtr_ch, ytr, champ_caches, y_val)
        prep = time.time() - ts
        print(f"\n[seed {seed}] champion mean={champ_aucs.mean():.4f} "
              f"(prep {prep:.0f}s, train {Xtr_ch.shape})", flush=True)

        # one-time correctness check (SMOKE): hstack == full combined feat
        if SMOKE and si == 0:
            k = cand_keys(cands[0])
            a = hstack_caches(champ_caches,
                              [precompute_fold(partial(extra_feat, keys=k), WINDOW, X, f)
                               for f in folds])[0]["feat"]
            b = precompute_fold(partial(combined_feat, keys=k), WINDOW, X, folds[0])["feat"]
            assert np.allclose(a[:len(b)], b, equal_nan=True), "hstack != combined feat!"
            print("  [smoke] hstack(champ,extra) == combined_feat  OK", flush=True)

        for ci, name in enumerate(cands):
            cts = time.time()
            try:
                keys = cand_keys(name)
                ef = partial(extra_feat, keys=keys)
                Xtr_ex, _ = build_training(X, yi, picks, ef, WINDOW)
                extra_caches = [precompute_fold(ef, WINDOW, X, f) for f in folds]
                Xtr_full = np.hstack([Xtr_ch, Xtr_ex])
                caches_full = hstack_caches(champ_caches, extra_caches)
                aucs, _ = eval_model(make_hgb(**TUNED, random_state=seed),
                                     Xtr_full, ytr, caches_full, y_val)
                md, _sd, t, _ = paired(aucs, champ_aucs)
                acc[name]["d"].append(md)
                acc[name]["t"].append(t)
                acc[name]["cand_mean"].append(float(aucs.mean()))
                raw.write(f"{seed}\t{name}\t{Xtr_full.shape[1]}\t{champ_aucs.mean():.4f}\t"
                          f"{aucs.mean():.4f}\t{md:+.4f}\t{t:+.2f}\t"
                          f"{datetime.now():%H:%M:%S}\n")
                raw.flush()
                os.fsync(raw.fileno())
                print(f"  [{ci+1:>2}/{len(cands)}] {name:<20} +{Xtr_ex.shape[1]}col "
                      f"mean={aucs.mean():.4f} d={md:+.4f} t={t:+.2f} ({time.time()-cts:.0f}s)",
                      flush=True)
            except Exception as e:  # noqa: BLE001 - one bad candidate must not kill the night
                print(f"  [{ci+1:>2}/{len(cands)}] {name:<20} FAILED: {e!r}", flush=True)
                raw.write(f"{seed}\t{name}\tERR\t\t\t\t\t{datetime.now():%H:%M:%S}\n")
                raw.flush()
        print(f"[seed {seed}] done in {(time.time()-ts)/60:.1f} min", flush=True)

    # --- ranked summary ---
    rows = []
    for name in cands:
        a = acc[name]
        if not a["d"]:
            continue
        d = np.asarray(a["d"])
        t = np.asarray(a["t"])
        pos = int((d > 0).sum())
        rows.append((
            name, float(d.mean()), float(d.min()), float(t.mean()), pos,
            float(np.mean(a["cand_mean"])),
        ))
    rows.sort(key=lambda r: r[1], reverse=True)

    with open(SUMMARY, "w", encoding="utf-8") as fh:
        head = (f"feature re-eval vs TUNED champion  seeds={POOL_SEEDS}  scales={SCALES}\n"
                f"gate: pos_all AND mean_d>=0.0015 AND mean_t>=2.5 -> CONFIRM\n\n"
                f"{'candidate':<20} {'mean_d':>8} {'min_d':>8} {'mean_t':>7} "
                f"{'pos':>4} {'cand_mean':>9}  verdict\n")
        fh.write(head)
        print("\n=== SUMMARY (ranked by mean_d) ===", flush=True)
        print(head, end="", flush=True)
        for name, md, mnd, mt, pos, cm in rows:
            v = screen_verdict(md, mt, pos)
            line = (f"{name:<20} {md:>+8.4f} {mnd:>+8.4f} {mt:>+7.2f} "
                    f"{pos:>3}/{len(POOL_SEEDS)} {cm:>9.4f}  {v}\n")
            fh.write(line)
            print(line, end="", flush=True)

    raw.close()
    print(f"\n=== DONE in {(time.time()-t0)/60:.1f} min ===", flush=True)
    print(f"wrote {SUMMARY} and {RAW}", flush=True)


if __name__ == "__main__":
    main()
