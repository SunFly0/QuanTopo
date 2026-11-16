# QuanTopo

Implementation of topology-directed discrete planning for text-to-motion generation.
The code contains the three training stages, text-conditioned generation, topology
preprocessing, and the model components used by the ablations.

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Python 3.10 or newer and PyTorch 2.2 or newer are required. The default configuration
uses CUDA and BF16. If the system requires a specific CUDA build of PyTorch, install
that build first; the remaining dependencies can then be installed from
`requirements.txt`.

## Configuration and commands

Architecture, optimization, sampling, and training settings are defined in
`configs/model/quantopo.yaml`. Dataset locations and dimensions are defined in
`configs/data/`. Runtime choices such as the training stage, checkpoint paths, output
directory, and device are command-line arguments. The supported training-step,
batch-size, and device overrides can be passed as `--override key=value`; all other
model settings remain explicit in the YAML file.

Installing the package registers the four `quantopo-*` commands used below. The files
in `scripts/` are thin convenience wrappers around the same command implementations;
they contain no separate hyperparameter settings. For example,
`python scripts/train_stage_a.py <arguments>` is equivalent to
`quantopo-train --stage a <arguments>`.

## Data layout

Dataset paths are relative to the project root and configured in `configs/data/`:

```text
data/humanml3d/
  train.txt
  val.txt
  test.txt
  Mean.npy
  Std.npy
  canonical_offsets.npy
  new_joint_vecs/<motion-id>.npy
  texts/<motion-id>.txt
```

Physical joints are recovered from denormalized RIC arrays. Optional `new_joints` files
may be supplied for source validation but are not required for training.

Place the local OpenCLIP ViT-B/32 checkpoint at `assets/clip/ViT-B-32.pt`, or pass a
different project-relative path to `--clip-checkpoint`.

## Training

First compute the topology targets:

```bash
quantopo-precompute-topology \
  --project-root . \
  --model-config configs/model/quantopo.yaml \
  --data-config configs/data/humanml3d.yaml \
  --split train \
  --output artifacts/topology/train
```

Train Stage A and fit the continuous-realization statistics:

```bash
quantopo-train \
  --project-root . --stage a --split train \
  --topology-cache artifacts/topology/train \
  --output-dir runs/stage_a

quantopo-fit-statistics \
  --project-root . --split train \
  --stage-a-checkpoint runs/stage_a/stage_a_final.pt \
  --output artifacts/realization_stats.npz \
  --output-dir runs/statistics
```

Train the text-conditioned planner and realization flow:

```bash
quantopo-train \
  --project-root . --stage b1 --split train \
  --stage-a-checkpoint runs/stage_a/stage_a_final.pt \
  --clip-checkpoint assets/clip/ViT-B-32.pt \
  --topology-cache artifacts/topology/train \
  --output-dir runs/stage_b1

quantopo-train \
  --project-root . --stage b2 --split train \
  --stage-a-checkpoint runs/stage_a/stage_a_final.pt \
  --stage-b1-checkpoint runs/stage_b1/stage_b1_final.pt \
  --clip-checkpoint assets/clip/ViT-B-32.pt \
  --realization-statistics artifacts/realization_stats.npz \
  --output-dir runs/stage_b2
```

For distributed training, invoke the training module directly:

```bash
torchrun --standalone --nproc-per-node=4 -m quantopo.cli.train \
  --project-root . --stage a --split train \
  --topology-cache artifacts/topology/train \
  --output-dir runs/stage_a
```

Configured batch sizes are global and must be divisible by the number of processes.

## Generation

```bash
quantopo-generate \
  --project-root . --split test \
  --stage-a-checkpoint runs/stage_a/stage_a_final.pt \
  --stage-b1-checkpoint runs/stage_b1/stage_b1_final.pt \
  --stage-b2-checkpoint runs/stage_b2/stage_b2_final.pt \
  --clip-checkpoint assets/clip/ViT-B-32.pt \
  --realization-statistics artifacts/realization_stats.npz \
  --text "a person walks forward and raises both arms" \
  --length 196 \
  --output-dir outputs/example
```

The generated NPZ contains RIC features, physical joints, plan tokens, persistence
predictions, and continuous realization coordinates.

## Source layout

- `contracts`, `data`: configuration, tensors, and dataset loading;
- `topology`, `text`, `planner`: trajectory targets and discrete planning;
- `representation`, `decoder`, `geometry`: motion representation and reconstruction;
- `flow`, `models`, `training`: continuous generation and stage training;
- `ablations`: alternative summaries, predictors, quantizers, readouts, decoders, routing,
  and deterministic realization components.

The `src/quantopo` package layout keeps these modules independent while allowing the
project to be installed with standard Python packaging tools.

## Matched power-spectrum summary

`PowerSpectrumSummary` evaluates frequencies m/32 (m=1,...,16) on the same
normalized trajectory states used by the other summary controls. It applies
log(1 + |F|^2/N), then training-fitted centered PCA and variance matching.
For N=32 the spectrum equals the non-DC orthonormal real FFT; shorter sequences
use direct Fourier sums on the same fixed grid, without spectral interpolation.
Fitted PCA buffers are restored from checkpoints without fitting on test data.

Run the independent numerical and checkpoint tests with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
