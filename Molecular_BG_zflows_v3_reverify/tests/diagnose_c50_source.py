"""Diagnose non-finite c50 energies/gradients on matched glycerol particles."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "Molecular_BG" / "glycerol_36d"
sys.path.insert(0, str(ROOT))

from zflows_md.boltzmann import build  # noqa: E402


def _load_driver():
    spec = importlib.util.spec_from_file_location("glycerol_train", RUN_DIR / "train.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _summary(name: str, values: torch.Tensor) -> None:
    finite = torch.isfinite(values)
    selected = values[finite]
    print(
        f"{name}: finite={finite.sum().item()}/{values.numel()} "
        f"nan={torch.isnan(values).sum().item()} "
        f"+inf={torch.isposinf(values).sum().item()} "
        f"-inf={torch.isneginf(values).sum().item()}"
    )
    if selected.numel():
        q = torch.quantile(
            selected.float(), torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0])
        )
        print(f"  quantiles[min,50,90,99,max]={q.tolist()}")


def main() -> None:
    with open(RUN_DIR / "config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    driver = _load_driver()
    data = ROOT / "zflows_md" / "data"
    prmtop = data / config["prmtop"]
    crd = prmtop.with_suffix(".rst7")
    print("generating the configured whitening trajectory ...", flush=True)
    frames = driver.short_md(str(prmtop), str(crd), config)

    common = dict(
        prmtop=str(prmtop),
        crd=str(crd),
        md_frames=np.asarray(frames),
        T=float(config["temperature_K"]),
        device="cpu",
        dtype=torch.float32,
        environment="vacuum",
    )
    c_problem = build(
        **common,
        regularization="c",
        c=50.0,
        c_scale=50.0,
        c_tail_fraction=0.0,
    )
    er_problem = build(
        **common,
        regularization="er",
        r_floor=0.1,
        e_cap=200.0,
        e_cap_scale=50.0,
    )

    torch.manual_seed(1701)
    particles = c_problem["u0"].samples(120000)
    raw_parts, c_parts, er_parts = [], [], []
    with torch.no_grad():
        for xi in particles.split(6000):
            cartesian = c_problem["u"].xi_to_cartesian(xi)
            raw_parts.append(c_problem["ff"](cartesian))
            c_parts.append(c_problem["u"](xi))
            er_parts.append(er_problem["u"](xi))
    _summary("raw Amber Cartesian energy", torch.cat(raw_parts))
    _summary("c50 pulled-back potential", torch.cat(c_parts))
    _summary("historical e/r pulled-back potential", torch.cat(er_parts))

    probe = particles[:12000].clone().requires_grad_(True)
    c_value = c_problem["u"](probe)
    c_grad = torch.autograd.grad(c_value.sum(), probe)[0]
    _summary("c50 gradient norm", torch.linalg.vector_norm(c_grad, dim=-1))

    probe = particles[:12000].clone().requires_grad_(True)
    er_value = er_problem["u"](probe)
    er_grad = torch.autograd.grad(er_value.sum(), probe)[0]
    _summary("historical e/r gradient norm", torch.linalg.vector_norm(er_grad, dim=-1))


if __name__ == "__main__":
    main()
