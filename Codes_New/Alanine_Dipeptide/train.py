#!/usr/bin/env python
"""Data-driven training for L-alanine dipeptide: one training from the source to the target.

Every step draws its target batch from the published equilibrium reference
(10^7 frames at 300 K, converted once to the bundle's internal coordinates)
through the `target_data` option of the jflows_md trainers, instead of
manufacturing it through the flow. The target is the regularized potential at
the ending regularization `RG_PARAM`.

    python train.py --method kl      # forward KL       -> artifacts/kl_data_driven
    python train.py --method klx     # KL + X_pi        -> artifacts/klx_data_driven
    python train.py --method kll1    # KL + L1 dispersion
    python train.py --method klxx    # KLXX (quench-and-temper mixture term)
    python train.py --method klxm    # KL + X over the equal mixture of ground-truth
                                     # subsamples and the detached pushforward (klxm.py)

Outputs: artifacts/<method>_data_driven/ (the converted reference is shared in
artifacts/target_data.npy; the flow and the batch ESS history),
artifacts/<method>_data_driven.log, results/<method>_data_driven.md with the
final validation ESS.
Ctrl+C stops the training at the next gradient step and evaluates the flow
as it is; a second Ctrl+C within three seconds stops the program.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np

from jflows.train import Monitor
from jflows_md import Mixed_NSF, Molecular_Bundle, Molecular_Potential
from jflows_md.boltzmann import Manual_Reject, _identity_weights, _push_and_weights
from jflows_md.train import train_forward_KLL1_G, train_forward_KLX_G, train_forward_KLXX_G
from jflows_md.utils.screen import compute_ESS_log

import parameters as P
from klxm import train_klxm_G


HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "bundle"
LABELS = {"kl": "forward KL", "klx": "KL+X_pi", "kll1": "KL+L1", "klxx": "KLXX", "klxm": "KL+X_mix"}


def reference_atom_names(handle):
    """Ordered atom names of the MDTraj topology embedded in the reference."""
    topology = json.loads(handle["topology"][0].decode("utf-8"))
    return tuple(
        atom["name"]
        for chain in topology["chains"]
        for residue in chain["residues"]
        for atom in residue["atoms"]
    )


def bundle_atom_names(pdb_path):
    """Ordered atom names of the bundle's reference PDB (the reference file's convention)."""
    return tuple(
        line[12:16].strip()
        for line in pdb_path.read_text().splitlines()
        if line.startswith(("ATOM", "HETATM"))
    )


def reference_internal(path, target, atom_names, log):
    """The reference converted to internal coordinates, one pass, kept under ``path``."""
    if path.exists():
        data = np.load(path, mmap_mode="r")
        log(f"reference in internal coordinates: {data.shape[0]} rows")
        return data
    to_internal = eqx.filter_jit(lambda x: target.coordinates.to_internal(x)[0])
    support = eqx.filter_jit(lambda x: target.support_mask(x))
    with h5py.File(HERE / P.REFERENCE, "r") as handle:
        names = reference_atom_names(handle)
        expected = tuple(atom_names)
        if names != expected:
            raise ValueError("reference topology does not match the bundle's atom order")
        frames = handle["coordinates"]
        total = frames.shape[0]
        temporary = path.with_suffix(".tmp.npy")
        out = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(total, target.dimension))
        kept, started = 0, time.perf_counter()
        for start in range(0, total, P.REFERENCE_CHUNK):
            stop = min(start + P.REFERENCE_CHUNK, total)
            x = jnp.asarray(np.asarray(frames[start:stop], dtype=np.float32))
            inside = np.asarray(support(x))
            q = np.asarray(to_internal(x), dtype=np.float32)
            finite = np.isfinite(q).all(axis=1)
            keep = inside & finite
            out[kept:kept + int(keep.sum())] = q[keep]
            kept += int(keep.sum())
            if (start // P.REFERENCE_CHUNK) % 10 == 0:
                log(f"converted {stop} of {total} frames, kept {kept}, {time.perf_counter() - started:.0f}s")
        out.flush()
        del out
    if kept < total:
        data = np.load(temporary, mmap_mode="r")[:kept]
        np.save(path, np.ascontiguousarray(data))
        del data
        os.remove(temporary)
    else:
        os.replace(temporary, path)
    log(f"reference converted: {kept} of {total} frames inside the support and finite")
    return np.load(path, mmap_mode="r")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=tuple(LABELS), default="klx")
    args = parser.parse_args()
    name = f"{args.method}_data_driven"
    artifacts = HERE / "artifacts"
    run_dir = artifacts / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifacts / f"{name}.log"
    open(log_path, "w").close()

    def log(message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(log_path, "a") as stream:
            stream.write(line + "\n")

    log(
        f"START L-alanine dipeptide 60D {name} ({LABELS[args.method]}) | RG_PARAM={P.RG_PARAM} "
        f"SCREEN_FRACTION={P.SCREEN_FRACTION} | VALID_SIZE={P.VALID_SIZE} "
        f"BATCH_SIZE={P.BATCH_SIZE} TRAIN_STEPS={P.TRAIN_STEPS} LR={P.LR} "
        f"LR_WARMUP={P.LR_WARMUP} U_CLIP={P.U_CLIP} G_CLIP={P.G_CLIP} CHUNKS={P.CHUNKS}"
    )
    bundle = Molecular_Bundle.load(BUNDLE)
    base = Molecular_Potential(bundle, temperature_kelvin=P.TEMPERATURE_KELVIN)
    target = base.regularized(P.RG_PARAM)
    source = base.source()
    domain = base.domain
    y_data = jnp.asarray(reference_internal(
        artifacts / "target_data.npy", base, bundle_atom_names(BUNDLE / "reference.pdb"), log,
    ))
    source_key, flow_key = jax.random.split(jax.random.key(P.SEED))
    x_valid = source.samples(source_key, N=P.VALID_SIZE)
    flow = Mixed_NSF(
        flow_key, domain, bins=P.BINS, transforms=P.TRANSFORMS,
        euclidean_bound=P.NSF_LIM, hidden_features=P.HIDDEN_FEATURES,
        slope=P.SLOPE, mask_strategy="balanced",
    ).zeros()

    reject = Manual_Reject()
    log(f"stop the training and evaluate: {reject.command}")
    monitor = Monitor(P.MONITOR_EVERY, f"[{P.MOLECULE} {name}] ", log)
    common = dict(
        monitor=monitor, seed=P.SEED, checkpoint=P.CHECKPOINT, u_clip=P.U_CLIP,
        g_clip=P.G_CLIP, lr_warmup=P.LR_WARMUP, screen_fraction=P.SCREEN_FRACTION,
        reject_requested=reject, target_data=y_data,
    )
    started = time.perf_counter()
    with reject:
        if args.method == "klxm":
            trained, history = train_klxm_G(
                x_valid, source, target, flow, domain, y_data, P.BATCH_SIZE, P.TRAIN_STEPS,
                P.LR, coeff_theta=1.0, coeff_alpha=0.5,
                **{k: v for k, v in common.items() if k != "target_data"},
            )
        elif args.method == "klxx":
            trained, history = train_forward_KLXX_G(
                x_valid, source, target, flow, domain, P.POOL_SIZE, P.BATCH_SIZE,
                P.TRAIN_STEPS, P.LR, 1, P.MELT, P.OPT_ALPHA, P.OPT_STEPS, P.MC_DT,
                0, P.MC_STEPS_2, coeff_lambda=1.0, coeff_qt=P.COEFF_QT,
                mc_image_radius=P.MC_IMAGE_RADIUS, chunks=P.CHUNKS, **common,
            )
        else:
            trainer = train_forward_KLL1_G if args.method == "kll1" else train_forward_KLX_G
            trained, history = trainer(
                x_valid, source, target, flow, domain, P.BATCH_SIZE, P.TRAIN_STEPS,
                P.LR, 1, 0.0, 0, 0, coeff_lambda=0.0 if args.method == "kl" else 1.0,
                **common,
            )
    trained = jax.block_until_ready(trained)
    history = np.asarray(history, dtype=np.float32)
    steps_done = int(np.sum(np.isfinite(history)))
    training_seconds = time.perf_counter() - started
    log(f"training done: {steps_done} of {P.TRAIN_STEPS} steps in {training_seconds:.1f}s")

    _, trained_log_weight = _push_and_weights(x_valid, source, target, trained, domain, P.CHUNKS)
    identity_log_weight = _identity_weights(x_valid, source, target, P.CHUNKS)
    trained_ess = float(compute_ESS_log(trained_log_weight, P.SCREEN_FRACTION))
    identity_ess = float(compute_ESS_log(identity_log_weight, P.SCREEN_FRACTION))
    log(f"validation ESS={trained_ess:.4f} (identity={identity_ess:.4f}) on {P.VALID_SIZE} source samples")

    eqx.tree_serialise_leaves(run_dir / "flow.eqx", jax.device_get(trained))
    np.save(run_dir / "batch_ess_hist.npy", history)
    factor = 1.0 / math.sqrt(trained_ess)
    (run_dir / "run.json").write_text(json.dumps({
        "method": f"{LABELS[args.method]}, data-driven", "reference": P.REFERENCE,
        "reference_rows": int(y_data.shape[0]), "rg_param": P.RG_PARAM,
        "batch_size": P.BATCH_SIZE, "steps_done": steps_done, "steps_total": P.TRAIN_STEPS,
        "training_seconds": training_seconds, "valid_trained_ess": trained_ess,
        "valid_identity_ess": identity_ess, "factor": factor,
    }, indent=2) + "\n")
    results = HERE / "results"
    results.mkdir(exist_ok=True)
    (results / f"{name}.md").write_text("\n".join([
        f"# L-alanine dipeptide 60D {LABELS[args.method]}, data-driven on the published reference",
        "",
        f"- Target data: `{y_data.shape[0]}` reference frames at 300 K in internal coordinates",
        f"- Regularization: `{P.RG_PARAM}`",
        f"- Batch size: `{P.BATCH_SIZE}`; steps: `{steps_done}` of `{P.TRAIN_STEPS}`",
        f"- Training time: `{training_seconds / 60:.2f} min`",
        f"- Validation ESS (source population of {P.VALID_SIZE}, one stage 0 -> 1): `{trained_ess:.6f}` (identity `{identity_ess:.6f}`)",
        f"- Factor F_hat = ESS^(-1/2): `{factor:.6g}`",
        "",
    ]) + "\n")
    log(json.dumps({"steps": steps_done, "valid_trained_ess": trained_ess, "factor": factor}))


if __name__ == "__main__":
    main()
