# DDSC-GPG H200 smoke runs

This repository packages the three-mode DDSC-GPG trainer and a licensed
16-image NIPS2017 subset for reproducible GPU smoke testing. Two launchers use
the same data and optimization settings while changing only DDSC damping.

## Generator modes

The trainer defaults to `isolated`, which uses the frozen ResNet-50 layer1
encoder and shared-lite decoder. Select `isolated_split` to keep the same
frozen encoder, adapter, and shared residual body while giving the perturbation
and mask branches independent upsampling trunks. Select `legacy` to use the
original learned dual-encoder GPG generator while retaining DDSC dynamic
lambda-1, pixel-space PGD guidance, the original trainable feature-guidance
loss, and the configured attack-objective Dropout-EOT behavior. In `legacy`
mode, `Grad_block1` through `Grad_block3` are trainable and receive the
feature-guidance gradient.

The supplied launchers expose the selection through `GENERATOR_MODE`:

```bash
GENERATOR_MODE=isolated_split bash scripts/run_nips2017_smoke_damping_025.sh
GENERATOR_MODE=isolated_split bash scripts/run_imagenet_train_damping_025.sh
GENERATOR_MODE=legacy bash scripts/run_nips2017_smoke_damping_025.sh
GENERATOR_MODE=legacy bash scripts/run_imagenet_train_damping_025.sh
```

For an ImageNet timing smoke run that performs exactly 16 training batches in
one epoch, use:

```bash
GENERATOR_MODE=legacy OUT_ROOT=/app/output/sharpit1 bash scripts/run_imagenet_smoke_16_batches_damping_025.sh
```

The trainer prints synchronized epoch seconds, batches/second, and
images/second. The launcher also prints total process wall time, including
model loading, dataset discovery, and checkpoint writing.

ImageNet launchers keep support smoothing disabled by default. Override it for
one run without editing the checkout:

```bash
DDSC_EMA_DECAY=0.5 GENERATOR_MODE=legacy OUT_ROOT=/app/output/sharpit1 bash scripts/run_imagenet_train_damping_025.sh
```

Frequency-channel dropout remains disabled by default. Enable it with a
default drop probability of `0.4` and four stochastic EOT members as follows:

```bash
LAYER1_DROPOUT_MODE=frequency_channel GENERATOR_MODE=legacy OUT_ROOT=/app/output/sharpit1 bash scripts/run_imagenet_train_damping_025.sh
```

The launcher also accepts `LAYER1_DROPOUT_P`,
`LAYER1_DROPOUT_CHANNEL_RATIO`, `LAYER1_DROPOUT_HF_RATIO`,
`LAYER1_DROPOUT_EOT_SAMPLES`, and `LAYER1_DROPOUT_EOT_REDUCTION` overrides.

Training checkpoints are mode-specific. Continuation rejects a checkpoint if
its generator mode differs from the requested mode. Legacy training
checkpoints created while the gradient encoder was frozen cannot be resumed,
because they used a different objective and optimizer parameter set; their
inference checkpoints remain readable. Existing v6/v7 training and v2
inference checkpoints remain readable as `isolated` checkpoints.

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

The target FaaS image `kau/pytorch-master` already supplies CUDA 13.0,
PyTorch 2.11.0, torchvision 0.26.0, NumPy 2.2.6, and Pillow 12.2.0.
No virtual environment or PyTorch reinstall is required. If building a new
environment, install the pinned ranges below:

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

Run the smoke variants separately with commands accepted by the FaaS runner:

```bash
bash scripts/run_nips2017_smoke_damping_025.sh
bash scripts/run_nips2017_smoke_damping_050.sh
```

Or run both sequentially:

```bash
bash scripts/run_nips2017_smoke_both.sh
```

Outputs are written under `/app/output/nips2017_smoke/`.
Override the location with `OUT_ROOT=/path/to/output`.

## Full ImageNet train set

The ImageNet training root must use torchvision `ImageFolder` layout with
exactly 1,000 class directories, normally WordNet IDs such as `n01440764`.
The FaaS launcher checks an explicit path, `IMAGENET_TRAIN_DIR`,
`/app/data/ImageNet-2012/train`, other standard mount locations, and sibling
input mounts in that order. The training launcher does not use the sibling
`/app/data/ImageNet-2012/val` directory.

Run one damping variant:

```bash
bash scripts/run_imagenet_train_damping_025.sh
bash scripts/run_imagenet_train_damping_050.sh
```

If automatic discovery does not find the mount and the FaaS command field
allows script arguments, pass it explicitly:

```bash
bash scripts/run_imagenet_train_damping_025.sh /path/to/ILSVRC2012_img_train
```

Both variants can be run sequentially, but this requires approximately twice
the FaaS execution time:

```bash
bash scripts/run_imagenet_train_both.sh
```

The full protocol uses batch size 16, all samples, 15 epochs, two warm-up
epochs, eight DataLoader workers, and an epoch checkpoint interval of one.
Outputs are written under `/app/output/imagenet_train/`. Override the location
with `OUT_ROOT=/path/to/output`.

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
