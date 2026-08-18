# DDSC-GPG H200 smoke runs

This repository packages the four-mode DDSC-GPG trainer and a licensed
16-image NIPS2017 subset for reproducible GPU smoke testing. Two launchers use
the same data and optimization settings while changing only DDSC damping.

## Generator and feature options

### Generator selection

The launchers read `GENERATOR_MODE`; the default is `isolated`.

| `GENERATOR_MODE` | Encoder and latent body | Decoder | Feature guidance |
| --- | --- | --- | --- |
| `isolated` | Frozen ResNet-50 through layer1, trainable 256-to-128 adapter, three shared depthwise residual blocks | One shared upsampling trunk with separate 3-channel perturbation and 1-channel mask heads | Off by default; adapter guidance can be enabled |
| `isolated_split` | Same frozen encoder, adapter, and shared residual body as `isolated` | Independent perturbation and mask upsampling trunks | Off by default; adapter guidance can be enabled |
| `frozen_legacy` | Frozen ResNet-50 through layer1, no adapter, six 256-channel full residual blocks | Original independent legacy decoders | Disabled; adapter guidance is rejected |
| `legacy` | Original trainable clean and PGD encoders with six 256-channel full residual blocks | Original independent legacy decoders | Original trainable feature guidance is always used; adapter guidance is rejected |

All four modes retain the existing pixel-space PGD-guidance loss. Decoder
width, residual-block count, and upsampling-backend overrides apply only to
`isolated` and `isolated_split`.

### Adapter feature guidance

`ADAPTER_FEATURE_GUIDANCE` controls only the two adapter-based generators.
Its launcher default is `0`.

| Option | Accepted values | Default | Effect |
| --- | --- | ---: | --- |
| `ADAPTER_FEATURE_GUIDANCE` | `0`, `1` | `0` | Passes `--adapter_feature_guidance` when set to `1` |
| `--adapter_feature_guidance` | CLI flag | Off | Enables the adapter feature-guidance branch |

When enabled, the clean image and `image + grad_delta` pass through the same
frozen ResNet prefix and separate trainable 256-to-128 adapters. The raw L2
distance between the two adapter outputs is computed immediately before the
shared residual body. Only the clean adapter output is decoded. The PGD
adapter adds 33,024 training parameters and is not used to produce inference
outputs.

The pixel-space PGD loss remains enabled. Its coefficient and the adapter
feature-loss coefficient both remain `lam_3`; no separate loss multiplier is
introduced. `ADAPTER_FEATURE_GUIDANCE=1` is rejected for `legacy` and
`frozen_legacy`.

### Generator examples

```bash
GENERATOR_MODE=legacy bash scripts/run_nips2017_smoke_damping_025.sh
GENERATOR_MODE=legacy bash scripts/run_imagenet_train_damping_025.sh
GENERATOR_MODE=frozen_legacy bash scripts/run_imagenet_train_damping_025.sh
GENERATOR_MODE=isolated_split bash scripts/run_imagenet_train_damping_025.sh
GENERATOR_MODE=isolated_split ADAPTER_FEATURE_GUIDANCE=1 bash scripts/run_imagenet_train_damping_025.sh
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

## Layer1 feature-dropout options

Layer1 feature dropout is independent of the generator and adapter feature
guidance. It affects only the frozen attack-objective ResNet-50 used for PGD
guidance and adversarial loss. It is never applied to clean-label inference or
the generator's frozen encoder.

The launcher defaults are:

| Environment option | Accepted values | Default | Effect |
| --- | --- | ---: | --- |
| `LAYER1_DROPOUT_MODE` | `off`, `frequency_channel` | `off` | Enables frequency-ranked layer1 channel dropout |
| `LAYER1_DROPOUT_P` | Float in `(0, 1)` when enabled | `0.4` | Drop probability for eligible channels |
| `LAYER1_DROPOUT_CHANNEL_RATIO` | Float in `(0, 1]` | `0.3` | Fraction of the highest high-frequency-energy channels eligible for dropout |
| `LAYER1_DROPOUT_HF_RATIO` | Float in `(0, 1]` | `0.35` | Centered low-frequency width/height ratio removed before channel-energy ranking |
| `LAYER1_DROPOUT_EOT_SAMPLES` | Positive integer | `4` | Number of stochastic dropout members in addition to one clean member |
| `LAYER1_DROPOUT_EOT_REDUCTION` | `logits`, `loss` | `logits` | Averages member logits before loss, or averages member losses |

`frequency_channel` requires the ResNet-50 attack model. Eligible channels are
ranked per sample using the L2 energy of their high-frequency projection.
Inverted dropout scaling preserves their expectation. With the launcher
default of four stochastic members, the ResNet suffix processes five members:
one clean member plus four dropout members.

Enable the default feature-dropout configuration as follows:

```bash
LAYER1_DROPOUT_MODE=frequency_channel GENERATOR_MODE=legacy OUT_ROOT=/app/output/sharpit1 bash scripts/run_imagenet_train_damping_025.sh
```

The feature-dropout and adapter feature-guidance options can be combined:

```bash
GENERATOR_MODE=isolated_split \
ADAPTER_FEATURE_GUIDANCE=1 \
LAYER1_DROPOUT_MODE=frequency_channel \
LAYER1_DROPOUT_P=0.4 \
LAYER1_DROPOUT_EOT_SAMPLES=4 \
bash scripts/run_imagenet_train_damping_025.sh
```

Training checkpoints are mode-specific. Continuation rejects a checkpoint if
its generator mode differs from the requested mode. Legacy training
checkpoints created while the gradient encoder was frozen cannot be resumed,
because they used a different objective and optimizer parameter set; their
inference checkpoints remain readable. Existing v6/v7/v8 training and v2
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
