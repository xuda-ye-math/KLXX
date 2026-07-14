#!/usr/bin/env python
"""Train ethane with KLXX and compare its flow torsions with OpenMM."""

from __future__ import annotations

import gc
import itertools
import json
import math
from pathlib import Path
import time

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from openmm import LangevinMiddleIntegrator, Platform, XmlSerializer, unit
from openmm import app
from scipy.ndimage import gaussian_filter1d

from alkane_bg import main as train_klxx
import parameters as P


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "results"

TORSION_ATOMS = (
    (2, 1, 0, 3),
    (2, 1, 0, 4),
    (2, 0, 1, 5),
    (2, 0, 1, 6),
    (2, 0, 1, 7),
)

TORSION_TITLES = (
    "H1-C2-C1-H2",
    "H1-C2-C1-H3",
    "H1-C1-C2-H4",
    "H1-C1-C2-H5",
    "H1-C1-C2-H6",
)


def _energy_kj_mol(simulation: app.Simulation) -> np.float32:
    state = simulation.context.getState(getEnergy=True)
    value = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    return np.float32(value)


def _dihedral4(positions: np.ndarray, atoms: tuple[int, int, int, int]) -> np.ndarray:
    a, b, c, d = (positions[:, atom] for atom in atoms)
    b1 = b - a
    b2 = c - b
    b3 = d - c
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2_hat = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    m1 = np.cross(n1, b2_hat)
    sine = np.sum(m1 * n2, axis=-1, dtype=np.float32)
    cosine = np.sum(n1 * n2, axis=-1, dtype=np.float32)
    return np.asarray(np.arctan2(sine, cosine), dtype=np.float32)


def _torsions(positions: np.ndarray) -> np.ndarray:
    return np.stack(
        tuple(_dihedral4(positions, atoms) for atoms in TORSION_ATOMS),
        axis=-1,
    ).astype(np.float32, copy=False)


def _openmm_reference() -> None:
    bundle = HERE / "bundle"
    dcd_path = ARTIFACTS / "openmm_reference.dcd"
    data_path = ARTIFACTS / "openmm_reference.npz"
    started = time.time()
    system = XmlSerializer.deserialize((bundle / "system.xml").read_text())
    if system.getNumConstraints() != 0:
        raise RuntimeError("the ethane OpenMM benchmark must remain unconstrained")
    pdb = app.PDBFile(str(bundle / "reference.pdb"))
    integrator = LangevinMiddleIntegrator(
        np.float32(300.0) * unit.kelvin,
        np.float32(P.OPENMM_FRICTION_PER_PS) / unit.picosecond,
        np.float32(P.OPENMM_DT_FS) * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(P.OPENMM_SEED)
    platform = Platform.getPlatformByName("CUDA")
    simulation = app.Simulation(
        pdb.topology,
        system,
        integrator,
        platform,
        {"Precision": "single"},
    )
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy(maxIterations=1000)
    simulation.context.setVelocitiesToTemperature(
        np.float32(300.0) * unit.kelvin, P.OPENMM_SEED + 1
    )
    minimized_energy = _energy_kj_mol(simulation)
    simulation.step(P.OPENMM_BURNIN_STEPS)
    equilibrated_energy = _energy_kj_mol(simulation)
    simulation.reporters.append(
        app.DCDReporter(
            str(dcd_path), P.OPENMM_STRIDE_STEPS, enforcePeriodicBox=False
        )
    )
    simulation.step(P.OPENMM_FRAMES * P.OPENMM_STRIDE_STEPS)
    final_energy = _energy_kj_mol(simulation)
    del simulation
    del integrator
    gc.collect()

    trajectory = md.load_dcd(str(dcd_path), top=str(bundle / "reference.pdb"))
    positions = np.asarray(trajectory.xyz, dtype=np.float32)
    if positions.shape != (P.OPENMM_FRAMES, 8, 3):
        raise RuntimeError(
            f"OpenMM wrote positions with shape {positions.shape}, "
            f"expected {(P.OPENMM_FRAMES, 8, 3)}"
        )
    base_torsions = _torsions(positions)
    permutations = np.asarray(
        [
            left + right
            for left in itertools.permutations((2, 3, 4))
            for right in itertools.permutations((5, 6, 7))
        ],
        dtype=np.int16,
    )
    torsion_blocks = []
    for permutation in permutations:
        order = np.concatenate((np.asarray((0, 1), dtype=np.int16), permutation))
        torsion_blocks.append(_torsions(positions[:, order]))
    torsions = np.asarray(np.concatenate(torsion_blocks, axis=0), dtype=np.float32)
    if positions.dtype != np.float32 or torsions.dtype != np.float32:
        raise TypeError("OpenMM reference arrays must remain float32")
    np.savez_compressed(
        data_path,
        schema_version=np.int32(1),
        base_torsions_rad=base_torsions,
        torsions_rad=torsions,
        hydrogen_permutations=permutations,
        platform=np.asarray(platform.getName()),
        precision=np.asarray("single"),
        temperature_kelvin=np.float32(300.0),
        dt_fs=np.float32(P.OPENMM_DT_FS),
        friction_per_ps=np.float32(P.OPENMM_FRICTION_PER_PS),
        burnin_steps=np.int32(P.OPENMM_BURNIN_STEPS),
        frames=np.int32(P.OPENMM_FRAMES),
        stride_steps=np.int32(P.OPENMM_STRIDE_STEPS),
        seed=np.int32(P.OPENMM_SEED),
        minimized_energy_kj_mol=minimized_energy,
        equilibrated_energy_kj_mol=equilibrated_energy,
        final_energy_kj_mol=final_energy,
        wall_seconds=np.float32(time.time() - started),
    )
    print(
        f"OpenMM reference: {P.OPENMM_FRAMES} base frames, "
        f"{torsions.shape[0]} symmetry-expanded float32 torsion vectors",
        flush=True,
    )


def _wrap(values: np.ndarray) -> np.ndarray:
    pi = np.float32(math.pi)
    period = np.float32(2.0 * math.pi)
    return np.asarray(np.remainder(values + pi, period) - pi, dtype=np.float32)


def _periodic_density(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(
        np.float32(-math.pi),
        np.float32(math.pi),
        P.DIHEDRAL_BINS + 1,
        dtype=np.float32,
    )
    count, _ = np.histogram(_wrap(values), bins=edges)
    density = np.asarray(count, dtype=np.float32)
    density = np.asarray(
        gaussian_filter1d(
            density, sigma=P.DIHEDRAL_SMOOTH_SIGMA, mode="wrap"
        ),
        dtype=np.float32,
    )
    width = np.float32(edges[1] - edges[0])
    density /= np.asarray(density.sum() * width, dtype=np.float32)
    centers = np.asarray(
        np.float32(0.5) * (edges[:-1] + edges[1:]), dtype=np.float32
    )
    return centers, density


def _js_bits(first: np.ndarray, second: np.ndarray) -> float:
    edges = np.linspace(
        np.float32(-math.pi),
        np.float32(math.pi),
        P.DIHEDRAL_BINS + 1,
        dtype=np.float32,
    )
    p = np.asarray(np.histogram(_wrap(first), bins=edges)[0], dtype=np.float32)
    q = np.asarray(np.histogram(_wrap(second), bins=edges)[0], dtype=np.float32)
    p /= np.asarray(p.sum(), dtype=np.float32)
    q /= np.asarray(q.sum(), dtype=np.float32)
    middle = np.asarray(np.float32(0.5) * (p + q), dtype=np.float32)
    value = np.float32(0.0)
    p_mask = p > 0
    q_mask = q > 0
    value += np.float32(0.5) * np.sum(
        p[p_mask] * np.log2(p[p_mask] / middle[p_mask]), dtype=np.float32
    )
    value += np.float32(0.5) * np.sum(
        q[q_mask] * np.log2(q[q_mask] / middle[q_mask]), dtype=np.float32
    )
    return float(value)


def _plot_dihedrals() -> None:
    with h5py.File(ARTIFACTS / "flow_samples.h5", "r") as handle:
        flow_samples = handle["particles"][:]
    if flow_samples.dtype != np.float32:
        raise TypeError(f"flow samples must be float32, got {flow_samples.dtype}")
    flow_torsions = np.asarray(flow_samples[:, -5:], dtype=np.float32)
    with np.load(ARTIFACTS / "openmm_reference.npz", allow_pickle=False) as saved:
        reference_torsions = saved["torsions_rad"]
        precision = str(saved["precision"].item())
    if reference_torsions.dtype != np.float32 or precision != "single":
        raise TypeError("saved OpenMM reference is not single-precision float32")
    if flow_torsions.shape[1] != 5 or reference_torsions.shape[1] != 5:
        raise ValueError("ethane torsion arrays must contain five coordinates")
    training = json.loads((ARTIFACTS / "summary.json").read_text(encoding="utf-8"))
    cap_active_fraction = training["target"]["flow_cap_active_fraction"]
    js_values = [
        _js_bits(flow_torsions[:, index], reference_torsions[:, index])
        for index in range(5)
    ]

    plt.rcParams.update(
        {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 12}
    )
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 8.0), sharey=True)
    flat = axes.reshape(-1)
    for index, axis in enumerate(flat[:5]):
        centers, reference_density = _periodic_density(reference_torsions[:, index])
        _, flow_density = _periodic_density(flow_torsions[:, index])
        axis.fill_between(
            centers,
            reference_density,
            color="0.80",
            alpha=0.95,
            label="OpenMM 300 K reference",
        )
        axis.plot(centers, reference_density, color="0.35", lw=1.1)
        axis.plot(
            centers,
            flow_density,
            color="#d62728",
            lw=2.2,
            label="KLXX Boltzmann flow",
        )
        axis.set_title(TORSION_TITLES[index], fontsize=16, pad=8)
        axis.set_xlim(-math.pi, math.pi)
        axis.set_xticks((-math.pi, 0.0, math.pi), (r"$-\pi$", "0", r"$\pi$"))
        axis.set_xlabel("dihedral (rad)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            0.03,
            0.94,
            f"JS = {js_values[index]:.4f} bits",
            transform=axis.transAxes,
            ha="left",
            va="top",
        )
    flat[5].set_visible(False)
    axes[0, 0].set_ylabel("density")
    axes[1, 0].set_ylabel("density")
    figure.suptitle(
        "ethane ($d=18$) — c50 KLXX flow vs OpenMM",
        fontsize=21,
        y=0.995,
    )
    handles, labels = flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.subplots_adjust(bottom=0.13, top=0.88, hspace=0.42, wspace=0.15)
    figure.savefig(RESULTS / "dihedrals.png", dpi=220, bbox_inches="tight")
    plt.close(figure)
    joined = ", ".join(f"{value:.6f}" for value in js_values)
    print(
        f"dihedrals.png: JS={joined} bits; "
        f"flow cap-active={cap_active_fraction:.6g}",
        flush=True,
    )


if __name__ == "__main__":
    train_klxx(P)
    _openmm_reference()
    _plot_dihedrals()
