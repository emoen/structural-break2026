"""SCRATCH correctness checks for the calibration fold-in (delete after use)."""
import json
import numpy as np

nb = json.load(open("streaming_detector_stack4.ipynb"))
g = globals()
SKIP = ("%pip", "crunch setup-notebook", "import crunch", "crunch_tools",
        "prediction/prediction.parquet", ".submit(")
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if any(tok in src for tok in SKIP):
        continue
    exec(compile(src, "nb", "exec"), g)

rng = np.random.default_rng(0)
# synthetic history + trailing online window (long enough to build both nulls)
x_h = np.cumsum(rng.standard_normal(500)) * 0.3 + rng.standard_normal(500)
w = np.cumsum(rng.standard_normal(250)) * 0.3 + rng.standard_normal(250)

ctx = hist_context(x_h, HIST_CAP)
ctx["calib"] = fit_calib_ref(x_h, ctx)

base = np.asarray(_base79_producer(w, ctx), dtype=np.float64)
full = np.asarray(compute_features_ms(w, ctx), dtype=np.float64)

print("len(base79 producer)      =", len(base))
print("len(new compute_features) =", len(full))
print("BASE79-PRESERVED maxdiff  =", float(np.max(np.abs(full[:len(base)] - base))))

# affine invariance: x -> a*x + b (a>0). Whole 158-row (esp. calib block) should be ~invariant.
a, b = 3.7, -11.0
xh2 = a * x_h + b
w2 = a * w + b
ctx2 = hist_context(xh2, HIST_CAP)
ctx2["calib"] = fit_calib_ref(xh2, ctx2)
full2 = np.asarray(compute_features_ms(w2, ctx2), dtype=np.float64)
calib_block1 = full[len(base):]
calib_block2 = full2[len(base):]
print("AFFINE calib-block maxdiff=", float(np.max(np.abs(calib_block1 - calib_block2))))
print("AFFINE full-row  maxdiff  =", float(np.max(np.abs(full - full2))))

# None-calib safety path: passes base through, 158 cols still (base twice-ish)
ctx3 = hist_context(x_h, HIST_CAP)  # no calib key
full3 = np.asarray(compute_features_ms(w, ctx3), dtype=np.float64)
print("None-calib path len       =", len(full3), "(safety: z=base)")
print("MS_SCALES =", MS_SCALES, "WINDOW =", WINDOW, "HIST_CAP =", HIST_CAP)
