# DDSC-GPG H200 smoke runs

This repository packages the isolated DDSC-GPG trainer and a licensed
16-image NIPS2017 subset for reproducible GPU smoke testing. Two launchers use
the same data and optimization settings while changing only DDSC damping.

## Fixed smoke settings

| Setting | Value |
| --- | ---: |
| Batch size | 16 |
| Samples | 16 |
| Epochs | 5 |
| Warm-up epochs | 2 |
| Damping variants | 0.25, 0.5 |
| GPU visible to Python | `cuda:0` |

Five one-batch epochs are intentional: damping first changes the controller's
second update and affects a subsequent training epoch. This is still a smoke
run, not a benchmark run.

## Environment

Use Python 3.12 and install an H200-compatible CUDA build of PyTorch first.
Then install the remaining pinned ranges:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the PyTorch wheel matching the host CUDA runtime if needed.
python -m pip install -r requirements.txt
```

The first run downloads torchvision's ResNet-50 IMAGENET1K_V1 weights into
`.cache/torch` unless `TORCH_HOME` already contains them.

## Run

Run the variants separately:

```bash
GPU_ID=0 PYTHON=.venv/bin/python ./scripts/run_nips2017_smoke_damping_025.sh
GPU_ID=0 PYTHON=.venv/bin/python ./scripts/run_nips2017_smoke_damping_050.sh
```

Or run both sequentially:

```bash
GPU_ID=0 PYTHON=.venv/bin/python ./scripts/run_nips2017_smoke_both.sh
```

Outputs are written under `runs/nips2017_smoke/` and are ignored by Git.
Override the location with `OUT_ROOT=/path/to/output`.

## Data and attribution

The included subset preserves official NIPS2017 `TrueLabel` values rather
than inferring labels from directory names. See
[`data/nips2017_smoke/README.md`](data/nips2017_smoke/README.md) and
[`data/nips2017_smoke/images.csv`](data/nips2017_smoke/images.csv) for source,
license, and per-image attribution details.

## Verification

```bash
python -m unittest discover -s tests -v
```
