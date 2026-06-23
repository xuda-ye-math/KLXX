"""Smoke-test EVERY method variant to verify each executes end-to-end (tiny sizes).

Runs the 8 hetero_bg ablation variants on one molecule via ``train.py --smoke`` and
checks each reaches the ``##### DONE`` line without crashing. ``--smoke`` saves NO
``data_<TAG>.pth`` and logs to ``status_<TAG>.smoke.log``, so this never touches the
committed paper files. Live, flushed status log (tail smoke_all_methods.log).

Run (env active):  python zflows_md_tests/smoke_all_methods.py [glycerol_36d]
glycerol is the default because it is the only molecule that exercised all 8 rows.
"""
import os, sys, subprocess, time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                                  # .../X-regularization
MOL = sys.argv[1] if len(sys.argv) > 1 else "glycerol_36d"
FOLDER = os.path.join(REPO, "Molecular_BG", MOL)
LOG = os.path.join(HERE, "smoke_all_methods.log"); open(LOG, "w").close()


def log(m):
    line = f"[{datetime.now():%H:%M:%S}] {m}"
    print(line, flush=True); open(LOG, "a").write(line + "\n")


# the 8 hetero_bg ablation variants: (method, extra flags) -> TAG
VARIANTS = [
    ("asmc", []), ("asmc", ["--raw"]),
    ("kl", []), ("kl", ["--raw"]),
    ("klxx", []), ("klxx", ["--raw"]),
    ("klxx", ["--no-delta"]), ("klxx", ["--no-delta", "--raw"]),
]
# very small batch + steps (and modest valid/pool) so every variant runs fast
SIZES = ["--smoke", "--n_valid", "1000", "--n_pool", "500", "--n_batch", "100", "--steps", "5"]

log(f"START smoke_all_methods  molecule={MOL}  variants={len(VARIANTS)}  sizes={' '.join(SIZES)}")
results = []
for i, (method, flags) in enumerate(VARIANTS, 1):
    label = " ".join([method] + flags)
    log(f"[{i}/{len(VARIANTS)}] RUN  train.py --method {label}  (tail status_*.smoke.log for live stages)")
    t0 = time.perf_counter()
    cmd = [sys.executable, "train.py", "--method", method] + flags + SIZES
    p = subprocess.run(cmd, cwd=FOLDER, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    out = p.stdout + p.stderr
    done = "##### DONE" in out
    ok = (p.returncode == 0) and done
    tag_line = next((l for l in out.splitlines() if "VACUUM BG" in l), "").strip()
    done_line = next((l for l in out.splitlines() if "##### DONE" in l), "").strip()
    results.append((label, ok, dt))
    log(f"[{i}/{len(VARIANTS)}] {'PASS' if ok else 'FAIL'}  ({label})  exit={p.returncode}  done={done}  {dt:.0f}s")
    if tag_line:
        log(f"        {tag_line[:118]}")
    if done_line:
        log(f"        {done_line[:118]}")
    if not ok:                                                # surface the failure tail for debugging
        log("        --- last output ---\n" + out[-1800:])

n_pass = sum(1 for _, ok, _ in results if ok)
log(f"END smoke_all_methods: {n_pass}/{len(results)} variants ran OK")
for label, ok, dt in results:
    log(f"   {'PASS' if ok else 'FAIL'}  {label}  ({dt:.0f}s)")
sys.exit(0 if n_pass == len(results) else 1)
