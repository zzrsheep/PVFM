# Australia full-shot baseline reproduction

## Scope

The release contains all **11 full-shot model families / 16 input variants**
in the current paper, plus adapters for all **six external TSFMs**. DLinear is
not an experiment in this release; its source is a transitive import of the
native quantile wrapper. No model architecture was rewritten while packaging.

| Family | Included variants |
|---|---|
| PatchTST, iTransformer, Crossformer, TimeMixer, LightTS | `feature_concat`, `time_concat` for each |
| Cross-Unet-v2 (`Cross_Unet`) | `cross_unet_nwp_direct` |
| FusionSF-no-spatial (`FusionSFNoSpatial`) | original PV+NWP branch (`mode=none` in CLI) |
| DAG, GCGNet, TiDE, TimeXer | their original `*Wrapper`, `dag_external_direct` |
| External TSFMs | Chronos-2, CITRAS-FM, Moirai-2, TiRex-2 + NWP, TimesFM-3, TabPFN-TS3 |

The full-shot catalog contains 2,500 archived station/task/seed configurations.
This is not a claim of complete three-seed coverage in every cell. See
`baselines/coverage.csv` for station counts per seed and task. Missing archived
seeds are not invented. Full-shot means training a separate model on each
unseen target station's own training split, selecting with its validation
split, then evaluating its test split. These stations remain unseen to the
released **pretrained PVFM** checkpoint.

## Install and inspect first

Use a fresh Python 3.11 environment. Install a suitable PyTorch CPU/CUDA wheel,
then `pip install -r requirements-baselines.txt`. All commands below are run
from the unpacked package root, not the original PVFM repository.

```bash
# Print the exact selected configuration; nothing trains without --execute.
python baselines/run.py --model iTransformer --mode time_concat \
  --task 1h_short_6h --station Bannerton --seed 2021

# Bounded CPU data/forward/backward check, not a performance experiment.
python baselines/run.py --model iTransformer --mode time_concat \
  --task 15min_short_4h --station Bannerton --seed 2021 \
  --audit --output outputs/audit_itf

# From-scratch full-shot training, validation, test and common-Q9 rescoring.
python baselines/run.py --model iTransformer --mode time_concat \
  --task 1h_short_6h --station Bannerton --seed 2021 \
  --execute --output outputs/itf_bannerton_6h

# All available archived seeds for one variant/task across Australia.
python baselines/run.py --model iTransformer --mode time_concat \
  --task 1h_short_6h --all-seeds --execute --output outputs/itf_australia_6h
python baselines/summarize.py --runs outputs/itf_australia_6h \
  --output outputs/itf_australia_6h_summary
```

Omitting `--model`, `--mode`, `--task`, and `--station` selects all catalogued
jobs for the chosen seed (default 2021); `--all-seeds` selects all 2,500.
This is a large training budget, so the launcher is **dry-run by default**.
Jobs execute sequentially. Use `CUDA_VISIBLE_DEVICES` to control an available
GPU; this package never kills/preempts other work or assumes four free GPUs.
For disconnected terminals, use `nohup` or your scheduler with a fresh log.
Output directories must be new and inside the package.

Task names are `1h_short_6h`, `1h_dayahead_24h`, `1h_week_168h`,
`15min_short_4h`, and `15min_short_6h`. Canonical contexts are 24/72/336 hourly
points and 64/96 native 15-minute points. **Some baseline interfaces use their
archived shorter context** (for example Cross-Unet); no silent context
extension is applied. Exact C/H, features, model dimensions, Q9 head, seed,
learning rate, 50-epoch maximum, patience and batch size are in each job's
configuration under
`baselines/configs/<model-variant>/<task>/<station>/seed<seed>.json`.
For example, `iTransformer_TC/1h_short_6h/Bannerton_Solar_Park/seed2021.json`.
They are not forced to PVFM's 80k-step pretraining recipe.

Each executed job records its command and `train.log`. The original engine
saves best-checkpoint/test artifacts under that job directory. Its raw
quantiles are rescored into `common_q9_metrics.csv`; `summarize.py` averages
seeds within each station before taking the station-equal mean. It refuses
incomplete regional coverage unless `--allow-partial` is explicit. Archived
`baselines/reference_metrics.csv` merges all 2,500 archived station/seed
reference rows without changing their values. Filter by model, mode, task,
station and seed. Its probability columns may use the old scorer; use the
release scorer for new comparisons. New outputs follow the readable
`<output>/<model-variant>/<task>/<station>/seed<seed>/` hierarchy; the
summarizer discovers their results recursively.

## Shared engine and frozen compatibility profiles

All models share `baselines/engine/`. `baselines/compat/` contains only four
files that differ between the archived experiment protocols. `catalog.json`
binds each job to its protocol and readable configuration path; the runner materializes the matching source
in the new output directory. This retains model/data behavior without five
copies of the implementation. No private machine paths or original shell
transcripts are distributed; portable seed configuration files define commands.
Historical hash IDs remain only as traceable metadata, not directory names.

Shared CSVs use readable station/task paths, not hashed filenames:

- `baselines/manifests/1h/<station>.csv` and
  `baselines/manifests/15min/<station>.csv`: one station-selection row each.
- `baselines/eval_windows/15min_C64_H16/<station>.csv` and
  `baselines/eval_windows/15min_C96_H24/<station>.csv`: frozen native-resolution
  test windows. The C/H numbers here are 15-minute time steps (16h/4h and
  24h/6h, respectively).

The [artifact index](../baselines/artifact_index.csv) lists every CSV and its
row count. There are 22 station lists and 22 evaluation-window lists for the
11 Australian stations. All models/seeds share these files via their job
configurations. These are input protocol files, not predictions or repeated
copies of raw PV/weather measurements. Historical machine-root metadata is
excluded from the station lists; the station identity and data columns are
unchanged. Hourly jobs retain their existing task/QC evaluation rules; this
layout change does not generate new windows or change the evaluation protocol.

Separate raw-data roots preserve time-axis padding and weather variants.
The shared vendored tree is restricted to the dependency closure of the
released models. Upstream mathematical model operations are unchanged.

## External TSFMs

Frozen-array evaluation additionally requires the prepared Australian NPZ
assets, which are not included in the current distribution. Raw CSV full-shot
training is unaffected. See `docs/AUSTRALIA_DATA.md`.

`baselines/external/evaluate.py` uses mechanically extracted original adapters;
upstream model implementations and weights come from their respective public
packages/repositories. Install each model in a **separate environment**, using
`baselines/external/environments/` and `weights.json`. Gated model downloads
require the user to accept upstream terms; no token or private credential is
included. External checkpoints are not redistributed in this archive.

```bash
# After downloading the model to a local path in its isolated environment:
python baselines/external/evaluate.py --model chronos2 \
  --checkpoint /path/to/chronos2 --task 15min_C64_H16 \
  --station Bannerton --limit-windows 1 --output outputs/chronos_smoke

# Remove --station and --limit-windows for all 11 Australia stations.
# Other model names: citras, moirai2, timesfm3, tirex2, tabpfn3.
# TabPFN's --checkpoint is the actual .ckpt file, not its parent directory.
```

`--input-role fullshot` (default) uses the exported verified per-station
full-shot weather/target preparation. `--input-role pvfm` uses the released
PVFM hourly cache contract instead; native 15min roles share the verified
replay. Hourly preprocessing of some **historical external-TSFM** evaluations
differs from the current full-shot replay. Therefore the new unified runner
does **not promise exact reproduction of those historical table numbers**.
Its input contract and predictions are saved explicitly. Model forward smoke
tests are not equivalent to complete five-task metric reproduction.

No future PV is given to a predictor. Weather for the forecast horizon is an
input; PV labels for that horizon are used only for scoring. The FEV wrapper
stores labels in a full dataset but exposes masked/cut-off model inputs using
its official EvaluationWindow API.
