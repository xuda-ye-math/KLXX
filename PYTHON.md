# Python environment setup

> You must be using a Linux system with an NVIDIA graphical card (or WSL, native
> Windows and Apple not supported, AMD cards not tested).

A step-by-step, interactive guide to building the conda environment for the
zflows-md Boltzmann-generator tasks. Run each step yourself in a terminal and let
it finish before moving on to the next.

> **Important — always `conda activate zflows` first; never run the interpreter
> path directly.** The `torch.compile` / Triton fast paths only work from inside
> the *activated* environment: activation puts the env's bundled, Blackwell-capable
> `ptxas` (CUDA 13.x, at `.../envs/zflows/bin/ptxas`) first on `PATH`. Launching the
> interpreter by its full path instead (e.g.
> `~/miniconda3/envs/zflows/bin/python script.py`) leaves the *system* `ptxas` on
> `PATH`, so TorchInductor fails with `Cannot find ptxas-blackwell` on Blackwell
> GPUs (sm_120). This is not a broken environment and does **not** call for
> `TORCHDYNAMO_DISABLE=1` — just activate the env, then run `python`.

## Step 1 — Create the `zflows` environment

```bash
conda create -n zflows python=3.12 -y
```

Creates a fresh environment named `zflows` on Python 3.12.

## Step 2 — Add the conda-forge channel

```bash
conda config --add channels conda-forge
```

Registers conda-forge globally (written to `~/.condarc`) so later installs pick it
up automatically — no `-c conda-forge` each time.

## Step 3 — Install OpenMM (CUDA build) and ParmEd

```bash
conda activate zflows
conda install openmm cuda-version=13
conda install parmed
```

The MD-reference helper (`short_md` in `zflows_md.bg.hetero_bg`) requests OpenMM's
`CUDA` platform (with a CPU
fallback), so pin the CUDA 13 build — matching the cu130 / Blackwell toolchain —
then install ParmEd.

### Smoke test — is the GPU visible to OpenMM?

Start an interactive Python session and paste the source directly:

```python
from openmm import Platform
ps = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
print("platforms:", ps)
print("CUDA available:", "CUDA" in ps)
```

Expect `CUDA` in the list and `CUDA available: True`. For a fuller check that the
CUDA platform actually computes forces, run `python -m openmm.testInstallation` —
every platform, including `CUDA`, should report "Successfully computed forces".

## Step 4 — Install PyTorch (GPU)

```bash
conda install pytorch-gpu
```

Pulls the GPU PyTorch build from conda-forge; the env's existing `cuda-version=13`
constraint keeps it on the CUDA 13 toolchain. The latest PyTorch ships its own
local `nvcc`/`ptxas` (not the system CUDA), which is what `torch.compile` uses to
build Triton kernels.

### Smoke test — PyTorch GPU + Triton / `torch.compile`

Start an interactive Python session and paste the source directly:

```python
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))

import triton
print("triton:", triton.__version__)

@torch.compile
def f(x):
    return (x * x).sin().sum()

x = torch.randn(4096, device="cuda")
print("torch.compile ok:", float(f(x)))
```

Expect `CUDA available: True`, your NVIDIA GPU as the device (the exact name and
compute capability depend on your card), a Triton version, and a finite
`torch.compile ok:` value — the last line confirms Triton plus the bundled
`nvcc`/`ptxas` actually compile and run a GPU kernel.

## Step 5 — Install the scientific stack

```bash
conda install matplotlib scikit-learn scikit-image pillow ase
```

`numpy`, `scipy`, `pandas`, and `networkx` are already pulled in by PyTorch/OpenMM.
`matplotlib` drives every figure script; `ase` supplies covalent radii and CPK
colors for the conformer renders; `scikit-learn`, `scikit-image`, and `Pillow`
build the Python-logo potential (`2D_Benchmark/2D_Python/core.py`).

Smoke test — paste into an interactive Python session:

```python
import matplotlib, sklearn, skimage, PIL, ase
print("matplotlib:", matplotlib.__version__)
print("scikit-learn:", sklearn.__version__)
print("scikit-image:", skimage.__version__)
print("Pillow:", PIL.__version__)
print("ase:", ase.__version__)
```

> **Optional** — only needed for specific auxiliary scripts; install if you use them:
> - `pymol-open-source` — molecule renders in `zflows_md/bg/pymol_render.py`

## Step 6 — Install the `zflows` and `zflows_md` packages

With `zflows` active:

```bash
conda install xudaye::zflows xudaye::zflows_md
```

Installs both packages directly from the `xudaye` anaconda.org channel. They are
**independent**: `zflows_md` vendors its own copy of the zflows flow / loss /
potential / utils machinery, so neither imports the other and either can be
installed on its own — installing both simply makes `import zflows` and
`import zflows_md` available together.

Smoke test — paste into an interactive Python session (run from any directory):

```python
import zflows, zflows_md            # both channel packages import independently
from zflows.utils import check_compile_available   # both ship compile support; check via zflows
check_compile_available()
```

This both confirms `zflows_md` is importable and runs its `torch.compile`
diagnostic. Example output:

```text
[OK ]   OS = Linux
[OK ]   nvcc = .../envs/zflows/bin/nvcc
[OK ]   sanity test passed (device=cuda, mode=reduce-overhead)

Note: please run check_compile_available() interactively or in a standalone python
script. Do not call it from your main training code — the sanity test really
invokes torch.compile, which costs compile time on every call and consumes a
Dynamo cache slot.
True
```

The environment is now complete: OpenMM (GPU), PyTorch + Triton, the scientific
stack, and the `zflows` + `zflows_md` packages (from the `xudaye` channel).
