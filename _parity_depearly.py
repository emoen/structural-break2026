"""Parity check: notebook compute_features_ms == validated b_no_accf feature vector.

Execs the depearly notebook's feature-defining cells (imports/helpers/config/residual/dep,
skipping the `crunch` cell and train/infer), then compares its 61-dim output against the
reference built from _night_banks (champ_base_row[51] + selected dep[10]) for random series.
"""
import numpy as np
import nbformat

from _exp_fast import hist_context as ref_hist_context, HIST_CAP
from _night_banks import fit_champ_refs, champ_base_row, fit_dep_refs, dep_feats

KEEP = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]
WINDOW = 250

nb = nbformat.read("streaming_detector_depearly.ipynb", as_version=4)
depidx = next(i for i, c in enumerate(nb.cells)
              if c.cell_type == "code" and "b_no_accf" in c.source and "def compute_features_ms" in c.source)
g = {}
for c in nb.cells[:depidx + 1]:
    if c.cell_type != "code":
        continue
    s = c.source
    if ("import crunch" in s or "crunch_tools" in s or "crunch." in s
            or s.lstrip().startswith("%") or "%pip" in s):
        continue
    exec(compile(s, "<nb>", "exec"), g)

nb_hist_context = g["hist_context"]
nb_compute = g["compute_features_ms"]

rng = np.random.default_rng(0)
maxerr = 0.0
for trial in range(6):
    nh = rng.integers(400, 3000)
    no = rng.integers(300, 1200)
    x_h = np.cumsum(rng.standard_normal(nh)) * 0.1 + rng.standard_normal(nh)
    x_on = rng.standard_normal(no)
    if trial % 2 == 0:  # inject a mid-series shift so dep features are exercised
        x_on[no // 2:] *= 2.3
    t = rng.integers(60, no)
    w = x_on[max(0, t - WINDOW + 1):t + 1]

    ctx_ref = ref_hist_context(x_h, HIST_CAP)
    cref = fit_champ_refs(x_h)
    dref = fit_dep_refs(x_h)
    ref_row = np.concatenate([
        champ_base_row(w, ctx_ref, cref),
        np.asarray([dep_feats(w, ctx_ref, dref)[i] for i in KEEP], dtype=np.float64),
    ])

    ctx_nb = nb_hist_context(x_h, HIST_CAP)
    nb_row = np.asarray(nb_compute(w, ctx_nb), dtype=np.float64)

    assert nb_row.shape == ref_row.shape == (61,), (nb_row.shape, ref_row.shape)
    err = float(np.max(np.abs(nb_row - ref_row)))
    maxerr = max(maxerr, err)
    print(f"trial {trial}: len={len(nb_row)} max|diff|={err:.2e}")

print(f"\nMAX ABS DIFF over all trials = {maxerr:.2e}  ->",
      "PARITY OK" if maxerr < 1e-9 else "MISMATCH!!")
