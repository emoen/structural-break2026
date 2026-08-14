"""Generate streaming_detector_seedens.ipynb from the champion notebook.

Seed-ensemble variant: train() fits K champions with different random_state and
saves them as a list; infer() averages their predict_proba per online step.
Everything else (features, config, window, warm-up) is identical to the champion.

Raw-JSON transform (a .ipynb is JSON) so we don't depend on nbformat and don't
risk hand-editing escaped cell source. Each replacement is asserted so a silent
mismatch fails loudly instead of producing a broken notebook.
"""
import json
import sys

SRC = "streaming_detector.ipynb"
DST = "streaming_detector_seedens.ipynb"

# (anchor_substring_to_locate_cell, old_text, new_text)
EDITS = [
    # 1) config: add ENSEMBLE_SEEDS after SEED (SEED is the cell's last line, no trailing \n)
    (
        "MAX_ITER         = 200",
        "SEED             = 42",
        "SEED             = 42\n"
        "ENSEMBLE_SEEDS   = (42, 7, 123, 2024, 31337)  # seed-ensemble: avg proba over K champions (variance reduction)",
    ),
    # 2) train(): fit K models instead of one, save as list under \"models\"
    (
        "model = HistGradientBoostingClassifier(",
        "    model = HistGradientBoostingClassifier(\n"
        "        max_iter=MAX_ITER, learning_rate=0.05, max_leaf_nodes=31,\n"
        "        l2_regularization=1.0, early_stopping=True, random_state=SEED,\n"
        "    )\n"
        "    model.fit(X_train, y_train)\n"
        "    print(f\"trained {model.n_iter_} trees\")\n",
        "    models = []\n"
        "    for _s in ENSEMBLE_SEEDS:\n"
        "        _m = HistGradientBoostingClassifier(\n"
        "            max_iter=MAX_ITER, learning_rate=0.05, max_leaf_nodes=31,\n"
        "            l2_regularization=1.0, early_stopping=True, random_state=_s,\n"
        "        )\n"
        "        _m.fit(X_train, y_train)\n"
        "        models.append(_m)\n"
        "    print(f\"trained {len(models)} seed-ensemble members "
        "({models[0].n_iter_} trees in member 0)\")\n",
    ),
    # 3) train(): save the list of models
    (
        '            "model": model,',
        '            "model": model,',
        '            "models": models,',
    ),
    # 4) infer(): load the list of models
    (
        'model = saved["model"]',
        '    model = saved["model"]\n',
        '    models = saved["models"]\n',
    ),
    # 5) infer(): average predict_proba across members
    (
        "score = float(model.predict_proba(feats)[0, 1])",
        "                    score = float(model.predict_proba(feats)[0, 1])",
        "                    score = float(np.mean([m.predict_proba(feats)[0, 1] for m in models]))",
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
