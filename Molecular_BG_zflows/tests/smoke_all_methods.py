"""Exercise every historical method variant from complete JSON configs."""
from __future__ import annotations

import copy
from datetime import datetime
import json
import os
import subprocess
import sys
import tempfile
import time


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MOLECULE = sys.argv[1] if len(sys.argv) > 1 else "glycerol_36d"
FOLDER = os.path.join(ROOT, "Molecular_BG", MOLECULE)
TRAIN = os.path.join(FOLDER, "train.py")
BASE_CONFIG = os.path.join(FOLDER, "config.json")
LOG = os.path.join(HERE, "smoke_all_methods.log")
open(LOG, "w", encoding="utf-8").close()


def log(message: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {message}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


with open(BASE_CONFIG, encoding="utf-8") as handle:
    base = json.load(handle)

variants = [
    ("asmc", False, 0.0),
    ("asmc", True, 0.0),
    ("kl", False, 0.0),
    ("kl", True, 0.0),
    ("klxx", False, 0.1),
    ("klxx", True, 0.1),
    ("klxx", False, 0.0),
    ("klxx", True, 0.0),
]

log(f"START smoke_all_methods molecule={MOLECULE} variants={len(variants)}")
results = []
with tempfile.TemporaryDirectory(prefix="zflows_md_smoke_") as temporary:
    for index, (method, raw, delta) in enumerate(variants, 1):
        config = copy.deepcopy(base)
        config.update(
            method=method,
            raw=raw,
            delta=delta,
            save_data=False,
            log_suffix=".smoke",
            n_valid=1000,
            n_pool=500,
            n_batch=100,
            steps=5,
            md_frames=200,
            max_stages=3,
        )
        tag = method + ("_delta" if method == "klxx" and delta > 0 else "")
        tag += "_raw" if raw else ("" if method == "asmc" else "_sharpen")
        config_path = os.path.join(temporary, f"{index:02d}_{tag}.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
            handle.write("\n")
        log(f"[{index}/{len(variants)}] RUN {tag}")
        started = time.perf_counter()
        process = subprocess.run(
            [sys.executable, TRAIN, "--config", config_path],
            cwd=FOLDER,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": ROOT},
        )
        elapsed = time.perf_counter() - started
        output = process.stdout + process.stderr
        done = "##### DONE" in output
        ok = process.returncode == 0 and done
        results.append((tag, ok, elapsed))
        log(
            f"[{index}/{len(variants)}] {'PASS' if ok else 'FAIL'} {tag} "
            f"exit={process.returncode} done={done} {elapsed:.0f}s"
        )
        if not ok:
            log("--- last output ---\n" + output[-2400:])

passed = sum(ok for _, ok, _ in results)
log(f"END smoke_all_methods: {passed}/{len(results)} variants ran OK")
for tag, ok, elapsed in results:
    log(f"   {'PASS' if ok else 'FAIL'} {tag} ({elapsed:.0f}s)")
raise SystemExit(0 if passed == len(results) else 1)
