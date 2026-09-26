# PVFM reproduction package

Anonymous code and inference weights for probabilistic photovoltaic forecasting.
Run every command below from this repository's root; none requires a particular
username, machine, home directory, or original research checkout.

## What is included

- One PVFM architecture, with 3.9M / 15.3M / 45.0M configurations.
- Two trained 3.9M checkpoints: all-region and Australia region-holdout.
- Core training with masked Q11 loss, dynamic temporal-window sampling,
  Adam, WSD, AMP, DDP and strict resume checks.
- Frozen evaluation-window manifests, reference scores, and one shared common-Q9 scorer.
- Eleven Australian target stations with their own train/validation/test PV
  and weather data for full-shot baseline training. No PVFM pretraining corpus.
- All 16 full-shot baseline variants and six external-TSFM adapters.

The package supports baseline retraining on the included target stations.
Prepared evaluation NPZ arrays are not included: checkpoint and external-TSFM
evaluation require those separate replay assets. No Hong Kong/UK demonstration
data is included. It **cannot regenerate the pretrained checkpoint
from scratch without the omitted pretraining corpus**. Synthetic training is
a functionality test, not an accuracy experiment.

## Setup

Use Python 3.11 and install a suitable PyTorch wheel (the original environment
used PyTorch 2.6.0). Then:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt
python verify.py
python -m pytest -q
```

Large assets are excluded from ordinary Git by `.gitignore`; a code-only clone
needs the matching asset bundle listed in `ASSETS.json`. `SHA256SUMS.json`
verifies the listed distributed files, including code, raw data and weights;
passing it does not imply optional evaluation arrays are present. Do not upload
local output folders or private backup directories with the release.

## Evaluate PVFM

The commands below require the prepared arrays referenced by the frozen
Australian manifests. These arrays have been removed from the distribution.
Without them the commands stop before loading a model or creating outputs;
they do not reconstruct approximate inputs from raw CSVs. See
[data availability](docs/AUSTRALIA_DATA.md).

```bash
# All 11 Australian unseen targets, all five tasks.
python evaluate.py --model-profile all_region \
  --device cuda --output outputs/australia_all_region

# Entire-region hold-out checkpoint, same Australian evaluation windows.
python evaluate_australia.py --model-profile australia_holdout \
  --device cuda --output outputs/australia_holdout
```

Use `--help` for station/task filters and bounded window checks. Output
directories must be new and inside the repository; existing results are not
overwritten. Point forecasts use q50. AQL and CRPS are rescored on the shared
0.1–0.9 grid; native Q11 training outputs are not changed.

| Task | Resolution | C (steps) | H (steps) | History → forecast |
|---|---|---:|---:|---|
| `1h_C24_H6` | hourly | 24 | 6 | 24h → 6h |
| `1h_C72_H24` | hourly | 72 | 24 | 72h → 24h |
| `1h_C336_H168` | hourly | 336 | 168 | 336h → 168h |
| `15min_C64_H16` | 15-minute | 64 | 16 | 16h → 4h |
| `15min_C96_H24` | 15-minute | 96 | 24 | 24h → 6h |

## Train and resume

```bash
# Two synthetic optimization steps; no test data or pretrained weights used.
python train.py --synthetic --device cpu --stop-after 2 \
  --output outputs/train_smoke
python train.py --synthetic --device cpu --stop-after 4 \
  --resume outputs/train_smoke/step_000002.pt \
  --output outputs/train_smoke_resume

# For a separately provided, authorized pretraining corpus:
python train.py --data-config configs/my_data.json --prepare-only
torchrun --standalone --nproc_per_node=4 train.py \
  --data-config configs/my_data.json --output outputs/pretrain
```

Create `configs/my_data.json` from `configs/external_data.example.json` with
your own data locations. Pool preparation is CPU-only. The supplied inference
checkpoints cannot resume optimizer state. See [training](docs/TRAINING.md).

## Full-shot and external baselines

```bash
python -m pip install -r requirements-baselines.txt

# Dry run: lists configurations without launching training.
python baselines/run.py --model iTransformer --mode time_concat \
  --task 1h_short_6h --station Bannerton --seed 2021

# Bounded CPU input/forward/backward audit.
python baselines/run.py --model iTransformer --mode time_concat \
  --task 1h_short_6h --station Bannerton --seed 2021 \
  --audit --output outputs/baseline_audit

# Original full-shot configuration: train, validate, test, common-Q9 rescore.
python baselines/run.py --model iTransformer --mode time_concat \
  --task 1h_short_6h --station Bannerton --seed 2021 \
  --execute --output outputs/baseline_train
```

The catalog has 2,500 station/task/seed jobs; missing archived seed coverage
is recorded in `baselines/coverage.csv`, not filled with invented numbers.
Configurations are organized under `baselines/configs/` by model variant,
task, station and seed; archived reference scores are consolidated in
`baselines/reference_metrics.csv`. New run outputs follow that same readable
hierarchy. All original training options and reference metric values are retained.
Historical retraining checks are in `reference/baseline_reproduction/`:
three tested model/task groups matched exactly; native 15-minute iTransformer
4h retraining was close but not bitwise-identical. These checks do not establish
full retraining parity for every model and seed.

Baseline implementations live in **one** `baselines/engine/` tree. Only genuine
experiment-version differences are stored in `baselines/compat/` (four files).
The runner creates a disposable merged engine in the new output directory;
it does not modify the shared source or silently substitute a newer protocol.
All six external TSFM adapters are retained; their weights and model-specific
environments must be obtained separately. See [baseline instructions](docs/BASELINES.md).

## Source map

| File | Responsibility |
|---|---|
| `pvfm/model.py` | Complete published forward path and Q11 reconstruction |
| `pvfm/blocks.py` | Factorized historical encoder and PV←NWP decoder |
| `pvfm/layers.py` | Patch/solar/position embeddings, RoPE and attention |
| `pvfm/loss.py` | Training loss |
| `pvfm/metrics.py`, `pvfm/probabilistic.py` | Common-Q9 evaluation |
| `pvfm/data/sampler.py` | Average-supply source/region/window sampling |
| `pvfm/data/window_shapes.py` | Context/horizon distributions and collation |
| `pvfm/data/temporal_pool_v1.py` | Reusable station/segment pool |
| `pvfm/data/temporal_quality_index.py` | Fast legal-window queries |
| `train.py`, `evaluate.py`, `evaluate_australia.py` | Entry points |

The [model card](docs/MODEL_CARD.md) explicitly documents learned absolute
positions, eight-dimensional site/window conditioning, both decoder FFN
updates and the final refinement block. These are active parts of the model,
not experimental branches. [Data](docs/DATA_CARD.md), [metrics](docs/METRICS.md)
and [licensing](LICENSE_NOTICE.md) describe scope and limitations. Third-party
attribution and licenses are preserved; project/data redistribution rights
still require the authors' confirmation before publication.
