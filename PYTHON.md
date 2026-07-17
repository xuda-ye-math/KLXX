# Python environment

The active local environment is the pip-only virtual environment
`~/.envs/jflows`. It is used for every current JAX and molecular workflow in
`Codes/` and `Molecular_BG/`. The former Conda `jflows` environment and the old
`~/.envs/jax` environment are retired.

The two live source packages remain outside the environment:

- `/data/projects/jflows`
- `/data/projects/jflows_md`

Local runs select them explicitly with `PYTHONPATH`. Do not install either
package into `~/.envs/jflows`, create a persistent `.pth` file, or use an
editable install for local experiments. This guarantees that every run uses
the current checked-out source.

## Clean construction

The environment uses the system Python and latest compatible pip releases; it
is not a bit-for-bit lockfile. On this workstation the system interpreter is
Python 3.14.

```bash
rm -rf "$HOME/.envs/jflows"
mkdir -p "$HOME/.envs"
/usr/bin/python3.14 -m venv "$HOME/.envs/jflows"
source "$HOME/.envs/jflows/bin/activate"

pip install --upgrade pip
pip install --upgrade \
  "jax[cuda13]" equinox "openmm[cuda13]" parmed mdtraj \
  scipy matplotlib h5py scikit-learn pyyaml ambertools-unofficial
```

The brackets are pip extras and should be quoted in shells such as zsh:

- `jax[cuda13]` installs JAX plus its CUDA-13 PJRT/plugin and NVIDIA runtime
  dependencies.
- `openmm[cuda13]` installs the OpenMM Python API plus the matching
  `OpenMM-CUDA-13` platform package. There is no generic `cuda` extra.
- `ambertools-unofficial` supplies the optional AmberTools command-line
  programs used to construct newly versioned small-molecule bundles. It is an
  unofficial repackaging and is not required for training from frozen bundles.

Use the corresponding CUDA 12 extras on a CUDA 12 machine. Plain `jax` and
plain `openmm` are sufficient only when GPU support is not required.

## Required validation

Check dependency closure and the preferred accelerator:

```bash
pip check
XLA_PYTHON_CLIENT_PREALLOCATE=false python
```

Then enter:

```python
>>> import jax
>>> import jax.numpy as jnp
>>> print("backend:", jax.default_backend())
>>> print("devices:", jax.devices())
>>> x = jnp.arange(4096, dtype=jnp.float32)
>>> y = jax.jit(lambda value: jnp.sin(value).sum())(x)
>>> jax.block_until_ready(y)
>>> print("compiled device:", y.device)
```

The expected backend is `gpu` and the compiled value should live on `cuda:0`.
Validate the molecular simulator separately:

```bash
python -m openmm.testInstallation
```

Reference, CPU, CUDA, and OpenCL should all compute forces within tolerance.
Finally, confirm that local packages are not installed:

```bash
cd /tmp
env -u PYTHONPATH python
```

Then enter:

```python
>>> import importlib.metadata as metadata
>>> import importlib.util
>>> print("jflows module:", importlib.util.find_spec("jflows"))
>>> print("jflows_md module:", importlib.util.find_spec("jflows_md"))
>>> names = {distribution.metadata["Name"].lower() for distribution in metadata.distributions()}
>>> "jflows" in names
False
>>> "jflows-md" in names
False
```

Both modules and distributions should be absent without `PYTHONPATH`.

## Running local experiments

Activate the environment once in each new terminal. Commands then use ordinary
`python` and `pip` names:

```bash
source "$HOME/.envs/jflows/bin/activate"
```

For `jflows` experiments:

```bash
cd /data/projects/X-regularization
PYTHONPATH=/data/projects/jflows \
  python Codes/Lattice_Clock/train.py
```

For molecular experiments:

```bash
cd /data/projects/X-regularization
PYTHONPATH=/data/projects/jflows:/data/projects/jflows_md \
  python Molecular_BG/methane_9d/train.py
```

Every molecular driver is full-size. The current 9D--45D results are complete;
do not rerun or overwrite them without explicit authorization. No smoke-sized
ESS run is a substitute for these configurations.

## Isolated package smoke tests

Verification should not write caches or generated artifacts into public source
trees. Copy the required trees to a temporary directory:

```bash
tmp=$(mktemp -d /tmp/jflows-smoke.XXXXXX)
mkdir -p "$tmp/jflows" "$tmp/jflows_md"
rsync -a --exclude='.git/' --exclude='__pycache__/' \
  /data/projects/jflows/jflows /data/projects/jflows/smoke \
  /data/projects/jflows/pyproject.toml "$tmp/jflows/"
rsync -a --exclude='.git/' --exclude='__pycache__/' \
  /data/projects/jflows_md/jflows_md /data/projects/jflows_md/smoke \
  /data/projects/jflows_md/bundles /data/projects/jflows_md/pyproject.toml \
  "$tmp/jflows_md/"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH="$tmp/jflows:$tmp/jflows_md" \
  python "$tmp/jflows_md/smoke/run_all.py"

rm -rf "$tmp"
```

The suite does not launch production molecular training.

## Current verification snapshot

The environment rebuilt on 2026-07-11 resolved the following releases. These
are evidence, not installation pins:

- Python 3.14.6
- JAX/JAXlib/CUDA plugin 0.10.2 and Equinox 0.13.8
- OpenMM and OpenMM-CUDA-13 8.5.2
- ParmEd 4.3.1 and MDTraj 1.11.1.post2
- NumPy 2.4.6 and SciPy 1.18.0
- Matplotlib 3.11.0, h5py 3.16.0, and scikit-learn 1.9.0
- PyYAML 6.0.3 for local skill/frontmatter validation
- `ambertools-unofficial` 26.0.0, providing working `antechamber`,
  `parmchk2`, `tleap`, and `sqm` commands

The complete isolated `jflows_md` smoke suite passed on `cuda:0`, including all
three molecular potentials, Mixed_NSF, MALA, SMC/AIS, artifact reconstruction,
one tiny Boltzmann-generator stage, and the real float32 glycerol compile path.

## Optional bundle reconstruction

Checked-in bundles are complete runtime inputs. Training and evaluation do not
need AmberTools, and the pure-JAX molecular potential does not invoke OpenMM at
runtime.

The installed `ambertools-unofficial` 26.0.0 toolchain can construct newly
versioned glycerol, diethanolamine, and other GAFF2/AM1-BCC targets. Because it
is an unofficial repackaging, use it only with explicit provenance and hashes.
The existing small-molecule bundles are frozen to AmberTools 24.8, so their
exact historical rebuild gate intentionally rejects version 26 output. ADP is
independent of AmberTools; its regenerated Hamiltonian and molecular potential
match the frozen FAB target.

The corresponding old molecular trees below `.archive/` retain historical
JAX/PyTorch/zflows instructions. They are not active environment documentation.
