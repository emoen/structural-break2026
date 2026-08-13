"""Overnight sequential runner for the E26-E29 paradigm-search notebooks.

Runs each notebook's code cells to completion, ONE AT A TIME in a single process,
so only one Parallel(n_jobs) harness is ever active (no CPU oversubscription).
Each notebook gets a fresh namespace; a crash in one is caught and logged so the
remaining notebooks still run. Full 10-fold (SMOKE unset -> False).

Recommended order (cheapest/most-likely first): E27 -> E26 -> E28 -> E29.

Run detached:
    nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _run_overnight.py \
        > _overnight_search.log 2>&1 & disown
"""
import gc
import os
import time
import traceback

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, filename) in run order
NOTEBOOKS = [
    ("E27 more_data", "exp_more_data.ipynb"),
    ("E26 hazard_target", "exp_hazard_target.ipynb"),
    ("E28 sequential_detector", "exp_sequential_detector.ipynb"),
    ("E29 lightgbm_pipeline", "exp_lightgbm_pipeline.ipynb"),
]


def run_notebook(label, path):
    t0 = time.time()
    print("\n" + "=" * 78, flush=True)
    print(f">>> {label}  ({os.path.basename(path)})  start {time.strftime('%H:%M:%S')}", flush=True)
    print("=" * 78, flush=True)
    try:
        nb = nbformat.read(path, as_version=4)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        ns = {"__name__": "__main__"}
        for i, c in enumerate(code_cells, 1):
            print(f"--- {label} cell {i}/{len(code_cells)} ---", flush=True)
            exec(compile(c.source, f"<{os.path.basename(path)} cell {i}>", "exec"), ns)
        status = "OK"
    except SystemExit as e:
        status = f"SKIPPED (SystemExit: {e})"
        print(f"!!! {label} raised SystemExit: {e}", flush=True)
    except Exception:
        status = "FAILED"
        print(f"!!! {label} FAILED with exception:", flush=True)
        traceback.print_exc()
    finally:
        ns = None
        gc.collect()
    dt = (time.time() - t0) / 60
    print(f"<<< {label}  {status}  in {dt:.1f} min", flush=True)
    return label, status, dt


def main():
    t0 = time.time()
    print(f"##### OVERNIGHT SEARCH START {time.strftime('%Y-%m-%d %H:%M:%S')} #####", flush=True)
    print(f"queue: {[l for l, _ in NOTEBOOKS]}", flush=True)
    results = []
    for label, fn in NOTEBOOKS:
        results.append(run_notebook(label, os.path.join(HERE, fn)))
    print("\n" + "#" * 78, flush=True)
    print(f"##### OVERNIGHT SEARCH DONE in {(time.time()-t0)/60:.1f} min #####", flush=True)
    for label, status, dt in results:
        print(f"  {label:28s} {status:24s} {dt:6.1f} min", flush=True)
    print("#" * 78, flush=True)


if __name__ == "__main__":
    main()
