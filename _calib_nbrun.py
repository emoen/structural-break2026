"""SCRATCH offline notebook runner (delete after use).

Executes the code cells of a champion notebook in one process, skipping ONLY:
  - the `%pip install` / `crunch setup-notebook` cell (needs network), and
  - the `crunch_tools.submit(...)` cell (must NOT submit).
Everything else runs exactly as the notebook would, so crunch_tools.test() does a
real local train+infer and the TS-AUC cell prints `Local TS-AUC`.
"""
import json
import sys

nb_path = sys.argv[1]
with open(nb_path) as f:
    nb = json.load(f)

# exec into THIS module's real __main__ globals so crunch_tools.test() can find
# train()/infer() by module introspection.
g = globals()
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "crunch setup-notebook" in src or "%pip" in src:
        print(f"[skip cell {i}: pip/setup]", flush=True)
        continue
    if "crunch_tools.submit" in src or ".submit(" in src:
        print(f"[skip cell {i}: submit]", flush=True)
        continue
    print(f"[run cell {i}]", flush=True)
    exec(compile(src, f"{nb_path}:cell{i}", "exec"), g)
