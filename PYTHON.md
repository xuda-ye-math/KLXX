# Python environment

KLXX requires the published `jflows` and `jflows_md` packages. They are
installed together; a sibling source checkout, editable installation, or
manually configured `PYTHONPATH` is not required.

## Installation

Use Python 3.11 or newer in a virtual environment. For example, from the KLXX
repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install 'jax[cuda13]' 'openmm[cuda13]'
pip install jflows==0.6.0 'jflows_md[bundles]==0.6.1'
```

Quoting the extras prevents shells such as zsh from expanding the brackets.
The first command installs the CUDA 13 builds of JAX and OpenMM. The second
installs both required project packages, Equinox through their declared
dependencies, and the complete bundle-construction stack through the
published `bundles` extra.

> **Remark.** This environment is pip-only or uv-only, and the supported
> platform is Linux with CUDA. JAX provides no GPU backend on Windows or
> macOS, and installing this stack inside a Conda environment can cause
> unexpected compilation problems. The installed NVIDIA driver must be newer
> than the minimum required by the JAX version being installed.

> **Remark.** PyPI treats hyphens and underscores as equivalent in project
> names, so `pip install jflows==0.6.0 'jflows-md[bundles]==0.6.1'` performs the same
> installation as the second command above. Python imports still use
> `jflows_md`.

## Validation

Check the installed dependency set and backend status:

```bash
pip check
python - <<'PY'
import jflows_md

jflows_md.backend()
PY
```

The report shows the installed JAX, Equinox, and OpenMM versions and their
available accelerator backends. For the installation above, it should report
CUDA backends for JAX and OpenMM.

## Running experiments

Run drivers from the KLXX repository root. They import the packages installed
in the active environment directly:

```bash
python Codes/Lattice_Clock/train.py
python Codes/Molecular_BG/alkane_family/methane_9d_raw/train.py
```

The reported molecular runs use their full configurations and are complete.
Do not rerun or overwrite their saved results without explicit authorization.

## Molecular bundles

All molecular bundles are stored locally under `Codes/Molecular_BG`, with
exactly one local copy for each target. Achiral, chiral, and raw-alkane drivers
load the `bundle/` directory in their own working folder. The sole exception
to this same-folder layout is a sharpening alkane: it reuses the bundle in the
corresponding `*_raw` folder rather than keeping a duplicate. For example,
`Codes/Molecular_BG/alkane_family/methane_9d_raw/` has the layout:

```text
methane_9d_raw/
├── bundle/
│   ├── coordinates.json
│   ├── manifest.json
│   ├── reference.pdb
│   ├── system.json
│   ├── system.xml
│   └── validation.json
├── parameters.py
└── train.py
```

Each same-folder driver resolves its local data with
`BUNDLE = HERE / "bundle"`. A sharpening alkane instead uses a relative
sibling path, such as
`BUNDLE = HERE.parent / "butane_36d_raw" / "bundle"`. Every retained bundle is
a complete runtime input: training and evaluation do not require AmberTools,
and the pure-JAX molecular potential does not invoke OpenMM at runtime.

## Software environment

The package versions below were read directly from the installed environment:

- Python: 3.14.7
- `jax[cuda13]`: 0.11.1
- `equinox`: 0.13.8
- `openmm[cuda13]`: 8.6.0
- `numpy`: 2.4.6
- `scipy`: 1.18.1
- `parmed`: 4.3.1
- `jflows`: 0.6.0
- `jflows_md[bundles]`: 0.6.1
