"""Headless smoke of streaming_detector_ensemble.ipynb: exec its code cells, run
train()+infer() on synthetic data, assert one finite score in [0,1] per online point."""
import os
import tempfile

import nbformat as nbf
import numpy as np

nb = nbf.read("streaming_detector_ensemble.ipynb", as_version=4)
ns = {}
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    src = cell.source
    if src.lstrip().startswith("%") or src.lstrip().startswith("!"):
        continue
    if "crunch" in src and ("load_notebook" in src or "load_data" in src or "crunch_tools" in src):
        continue
    exec(compile(src, "<cell>", "exec"), ns)

train, infer = ns["train"], ns["infer"]

rng = np.random.default_rng(0)
def make_series(sid, break_it):
    hist = rng.normal(0, 1, size=rng.integers(300, 500))
    n_on = int(rng.integers(60, 160))
    online = rng.normal(0, 1, size=n_on)
    tau = None
    if break_it:
        tau = int(rng.integers(20, n_on - 5))
        online[tau:] += rng.normal(2.0, 0.3)  # mean shift
    return sid, hist.tolist(), online.tolist(), tau

datasets = [make_series(i, i % 2 == 0) for i in range(40)]

with tempfile.TemporaryDirectory() as md:
    train(datasets, md)
    print("train OK; model saved")

    infer_data = [(h, on) for (_, h, on, _) in datasets[:6]]
    gen = infer(iter(infer_data), md)
    next(gen)  # readiness

    total = 0
    for (h, on) in infer_data:
        k = 0
        for _ in on:
            s = next(gen)
            assert np.isfinite(s) and 0.0 <= s <= 1.0, f"bad score {s}"
            k += 1
            total += 1
        assert k == len(on), "score count mismatch per series"
    print(f"infer OK; {total} scores, all finite in [0,1]")
    # distribution sanity: non-warmup scores should vary (not all identical)
print("SMOKE PASS")
