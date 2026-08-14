"""Generate streaming_detector_tuned.ipynb from the champion notebook.

Wires the E32 multi-seed-search winner (confirmed positive on 8/8 independent seeds)
into the champion's single-model train cell. Everything else (features, infer, config)
is identical to the champion, so infer() is unchanged (still loads saved["model"]).

Winner: lr=0.0211, max_iter=400, max_leaf_nodes=63, min_samples_leaf=10,
        l2_regularization=1.0, max_features=0.5   (vs champion 0.05/200/31/20/1.0/1.0)
"""
import json
import sys

SRC = "streaming_detector.ipynb"
DST = "streaming_detector_tuned.ipynb"

EDITS = [
    # 1) config: bump MAX_ITER 200 -> 400
    (
        "MAX_ITER         = 200",
        "MAX_ITER         = 200   # gradient-boosting trees (fewer -> faster per-step predict)",
        "MAX_ITER         = 400   # E32 tuned winner (was 200): slow-lr + many-trees + feature-bagging; robust 8/8 seeds",
    ),
    # 2) model constructor: champion params -> tuned winner params
    (
        "max_iter=MAX_ITER, learning_rate=0.05, max_leaf_nodes=31,",
        "        max_iter=MAX_ITER, learning_rate=0.05, max_leaf_nodes=31,\n"
        "        l2_regularization=1.0, early_stopping=True, random_state=SEED,\n",
        "        max_iter=MAX_ITER, learning_rate=0.021145795969329348, max_leaf_nodes=63,\n"
        "        min_samples_leaf=10, l2_regularization=1.0, max_features=0.5,\n"
        "        early_stopping=True, random_state=SEED,\n",
    ),
]


def main() -> int:
    with open(SRC, encoding="utf-8") as fh:
        nb = json.load(fh)

    applied = [0] * len(EDITS)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        for i, (anchor, old, new) in enumerate(EDITS):
            if anchor in src and old in src:
                src = src.replace(old, new, 1)
                applied[i] += 1
        cell["source"] = src.splitlines(keepends=True)

    missing = [i + 1 for i, n in enumerate(applied) if n == 0]
    if missing:
        print(f"ERROR: edits not applied: {missing} (counts={applied})", file=sys.stderr)
        return 1

    with open(DST, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"OK wrote {DST}  (edit counts={applied})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
