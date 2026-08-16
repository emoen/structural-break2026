"""Parse _forest_tune.log and aggregate per-config stats across whatever seeds completed."""
import re
import sys
from collections import defaultdict

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "_forest_tune.log"
seed_re = re.compile(r"\[seed (\d+)\]")
cfg_re = re.compile(r"cfg(\d+)\s+\[(.*?)\]\s+best\s+(\S+)\s+d=([+-][\d.]+)\s+t=([+-][\d.]+)")

d_by_cfg = defaultdict(dict)   # cfg -> {seed: d}
t_by_cfg = defaultdict(dict)
w_by_cfg = defaultdict(dict)
params = {}
cur_seed = None
seeds_seen = []

for line in open(path):
    m = seed_re.search(line)
    if m:
        cur_seed = int(m.group(1))
        if cur_seed not in seeds_seen:
            seeds_seen.append(cur_seed)
        continue
    c = cfg_re.search(line)
    if c and cur_seed is not None:
        i = int(c.group(1))
        params[i] = c.group(2)
        w_by_cfg[i][cur_seed] = c.group(3)
        d_by_cfg[i][cur_seed] = float(c.group(4))
        t_by_cfg[i][cur_seed] = float(c.group(5))

# only configs present on the FULL set of complete seeds
complete = [s for s in seeds_seen if all(s in d_by_cfg[i] for i in d_by_cfg)]
# a seed is "complete" if every cfg that appears anywhere also has it; approximate:
full_seeds = [s for s in seeds_seen
              if sum(1 for i in d_by_cfg if s in d_by_cfg[i]) == max(
                  sum(1 for i in d_by_cfg if ss in d_by_cfg[i]) for ss in seeds_seen)]

print(f"seeds seen: {seeds_seen}")
print(f"treating these as complete: {full_seeds}\n")
print(f"{'cfg':<5} {'mean_d':>8} {'min_d':>8} {'mean_t':>7}  pos_all  n_seeds  verdict  params")

rows = []
for i in sorted(d_by_cfg):
    ds = [d_by_cfg[i][s] for s in full_seeds if s in d_by_cfg[i]]
    ts = [t_by_cfg[i][s] for s in full_seeds if s in t_by_cfg[i]]
    if len(ds) < len(full_seeds):
        continue
    d = np.asarray(ds); t = np.asarray(ts)
    pos = (d > 0).all()
    confirm = pos and d.mean() >= 0.0015 and t.mean() >= 2.5
    verdict = "CONFIRM" if confirm else ("promising" if (pos and d.mean() > 0) else "noise")
    rows.append((d.mean(), i, d, t, pos, len(ds), verdict))

rows.sort(key=lambda r: -r[0])
for md, i, d, t, pos, n, verdict in rows:
    print(f"cfg{i:<2} {d.mean():>+8.4f} {d.min():>+8.4f} {t.mean():>+7.2f}   "
          f"{'YES' if pos else ' no':>3}     {n:>2}     {verdict:<9} {params[i]}")
