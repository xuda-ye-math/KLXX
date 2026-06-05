# Recover the full per-step training history -- every attempt, accepted AND
# rejected -- from train_status.log into ess_history_all_attempts.csv.
# Columns: tag, stage, t_k, attempt, accepted, val_ess, step, loss, ess.
# The .pth files keep only the accepted attempt per stage (until the
# ess_hist-in-attempts change); the log has every step line, so this parser
# is the ground-truth archive for convergence-rate analysis.
# Usage: /opt/torch/bin/python parse_status_log.py [train_status.log ...]
import re
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
logs = [Path(a) for a in sys.argv[1:]] or [HERE / 'train_status.log']

RE_RUN = re.compile(r'CLOCK RUN START (\S+) ')
RE_STAGE = re.compile(r'\[stage (\d+)\] t_\{k-1\}=([\d.]+) -> t_k=([\d.]+)')
RE_STEP = re.compile(r'\[train\] step\s+(\d+)/\d+\s+loss=([-\d.e+]+|nan|inf)'
                     r'\s+direct ESS=([\d.]+)')
RE_ATT = re.compile(r'\[stage (\d+)\] attempt (\d+): validation ESS=([\d.]+)')
RE_SHRINK = re.compile(r'\[stage (\d+)\] abort -> shrink to t_k=([\d.]+)')

rows, tag, stage, t_k, steps = [], None, 0, None, []
for log in logs:
    for line in open(log):
        if m := RE_RUN.search(line):
            tag, stage, t_k, steps = m.group(1), 0, None, []
        elif m := RE_STAGE.search(line):
            stage, t_k, steps = int(m.group(1)), float(m.group(3)), []
        elif m := RE_SHRINK.search(line):
            t_k, steps = float(m.group(2)), []
        elif m := RE_STEP.search(line):
            steps.append((int(m.group(1)), m.group(2), float(m.group(3))))
        elif m := RE_ATT.search(line):
            att, val = int(m.group(2)), float(m.group(3))
            for step, loss, ess in steps:
                rows.append([tag, stage, t_k, att, '', val, step, loss, ess])
            steps = []

# accepted = the last attempt of each (tag, stage); earlier attempts rejected
last = {}
for r in rows:
    last[(r[0], r[1])] = max(last.get((r[0], r[1]), 0), r[3])
# a stage's final attempt is accepted iff its val_ess >= 0.3 (the floor)
for r in rows:
    r[4] = int(r[3] == last[(r[0], r[1])] and r[5] >= 0.3)

out = HERE / 'ess_history_all_attempts.csv'
with open(out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['tag', 'stage', 't_k', 'attempt', 'accepted', 'val_ess',
                'step', 'loss', 'ess'])
    w.writerows(rows)
tags = sorted({r[0] for r in rows})
print(f"wrote {out.name}: {len(rows)} step records, {len(tags)} runs")
for t in tags:
    n_att = len({(r[1], r[3]) for r in rows if r[0] == t})
    n_rej = len({(r[1], r[3]) for r in rows if r[0] == t and not r[4]})
    print(f"  {t}: {n_att} attempts ({n_rej} rejected)")
