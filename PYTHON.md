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
pip install jflows==0.5.4 'jflows_md[bundles]==0.5.4'
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
> names, so `pip install jflows==0.5.4 'jflows-md[bundles]==0.5.4'` performs the same
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

## System specifications

The following snapshot records the machine used for the reported computations
on 2026-07-24.

### Hardware and operating system

<div align="center">

| component | specification |
|---|---|
| CPU | AMD Ryzen 9 9950X3D, 16 cores and 32 threads, up to 5.76 GHz |
| GPU | MSI GeForce RTX 5090 32G Gaming Trio OC; NVIDIA GeForce RTX 5090 with 32,607 MiB VRAM; NVIDIA driver 610.43.03 |
| RAM | 64 GB installed as two 32 GB DDR5-6000 CL36 modules; 6000 MT/s effective data rate, corresponding to a 3000 MHz DDR clock; Linux reports 60.2 GiB usable |
| motherboard | MSI MPG X870E Carbon WiFi (MS-7E49) |
| operating system | Arch Linux, rolling release, x86-64 |
| kernel | Linux 7.1.4-arch1-1 |

</div>

The GPU runs an overclocked profile: a 600 W TDP, a `+250` MHz GPU core clock
offset, and a `+2000` MHz VRAM clock offset.

The RAM description is based on the two detected DDR5 SPD devices and their
`KF560C36-32` module strings. The unprivileged kernel interfaces expose the
module rating but not an independent live memory-controller clock, so
3000 MHz is the clock corresponding to the rated DDR5-6000 profile.

### Software environment

The package versions below were read directly from the installed environment:

- Python: 3.14.6
- `jax[cuda13]`: 0.11.0
- `equinox`: 0.13.8
- `openmm[cuda13]`: 8.5.2
- `numpy`: 2.4.6
- `scipy`: 1.18.0
- `jflows`: 0.5.4
- `jflows_md[bundles]`: 0.5.4
