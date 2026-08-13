"""Headless runner for exp_bank_selection.ipynb (E25 / Option A).

Executes the notebook's code cells in order in one namespace so the full 10-fold
bank+selection+validation runs detached. Prints go to the log. Does NOT modify the
notebook. Run:  nohup .venv/bin/python -u _run_exp_a.py > _exp_a.log 2>&1 &
"""
import os
import time

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "exp_bank_selection.ipynb")


def main():
    t0 = time.time()
    nb = nbformat.read(NB, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    print(f"=== E25 bank+selection (Option A) headless run {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"{len(code_cells)} code cells\n", flush=True)
    ns = {"__name__": "__main__"}
    for i, c in enumerate(code_cells, 1):
        print(f"--- cell {i}/{len(code_cells)} ---", flush=True)
        exec(compile(c.source, f"<cell {i}>", "exec"), ns)
    print(f"\n=== DONE in {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
