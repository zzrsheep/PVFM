# Baseline reproduction

See [BASELINES.md](../docs/BASELINES.md) for training/evaluation commands and
model coverage. The [CSV index](artifact_index.csv) explains all shared input
lists by station and task.

```text
baselines/
  manifests/
    1h/Bannerton_Solar_Park.csv
    15min/Bannerton_Solar_Park.csv
    ...                            # 11 stations per resolution
  eval_windows/
    15min_C64_H16/Bannerton_Solar_Park.csv
    15min_C96_H24/Bannerton_Solar_Park.csv
    ...                            # 11 stations per task
  artifact_index.csv              # purpose, task, station, row count, path
  configs/
    iTransformer_TC/
      1h_short_6h/Bannerton_Solar_Park/seed2021.json
    ...                            # model variant / task / station / seed
  catalog.json                     # searchable index and config paths
  reference_metrics.csv            # all 2,500 archived station/seed results
  engine/                         # shared baseline implementation
  compat/                         # only required historical protocol differences
  external/                       # six external foundation-model adapters
```

`manifests` contains **22 single-row station lists**. The runner uses them to
select a station, its resolution and the input filenames. These are not
evaluation scores or actual power measurements.

`eval_windows` contains **22 frozen native-15-minute test-window lists**
(52,964 rows in total). C64/H16 means 16 hours of history and four forecast
hours; C96/H24 means 24 hours and six hours. Each row fixes the window's
station and timestamps. The window CSV contents are unchanged by the
human-readable rename. Their strict inclusion/order is part of reproduction.

Hourly jobs use the existing hourly task and QC definitions. No hourly or
native task was regenerated or otherwise changed by this file reorganization.

Models and seeds share these lists; do not delete them independently of their
job configurations. Raw data lives in `datasets/australia/`, reference scores
in the job/reference directories, and newly generated predictions in the
user-selected output directory.

The 2,500 configurations retain their original bytes and training options.
`FC` means feature concatenation and `TC` means time concatenation. Archived
job IDs remain metadata for traceability and `--id` selection; directories
and new run outputs use readable names. No archived experiment was removed.

`reference_metrics.csv` is one consolidated table, not training input. Filter
by `model`, `mode`, `task`, `station_dir` and `seed` (or the unique `id`). Every
original metric cell is retained without recomputation or rounding. Its
probability columns can use the historical scorer: this is an archived
reference, **not** a newly harmonized common-Q9 result. New runs still use
`score.py` and `summarize.py` as described in the baseline instructions.
