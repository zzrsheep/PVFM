# Core training and supplied-data boundary

This release contains the **real production sampler** and materializer, not a
uniform-window demo replacement. The original prepared pretraining corpus is
excluded from the release. Without that corpus a reader can load the supplied
weights and run synthetic training, but cannot retrain the reported pretrained
model from the bundled unseen target stations. Frozen-array inference requires
the separately supplied replay assets described in AUSTRALIA_DATA.md.

## Sampling (formal all-region PVFM-3.9M-80k)

The compact module map is in `pvfm/data/README.md`. Only the average-supply
full-data training policy is exposed; origin thinning and short-horizon
continued-pretraining branches are not part of this release.

1. Choose a data source: 1h=0.75, 15min=0.25; resolve cache within resolution.
2. Choose actual C/H before cutoff. In each resolution, typical-H branch has
   probability 0.6, random-H branch 0.4.
   - 1h: typical H={6,24,168} with equal weights; random H in [1,168].
     C/H between 0.5 and 4, maximum C=672; integer rounding and minimum-history
     boundary details are preserved by the original sampling function.
   - 15min: typical H={1,4,6,24} hours with equal weights; random H between
     0.25 and 24 hours in native quarter-hour steps. C/H in [0.5,4], max C=96h.
3. Query legal station/cutoff pairs for those actual C/H from the reusable QC
   index. Reuse calibrated average-supply regional probabilities with alpha
   0.25, regional cap 0.15 and oversampling bound 20. Calibration uses 5,000
   shapes/source and seed 20260913, not empirical exposure of an older run.
4. Select feasible region/station/cutoff with the original repeat protection,
   check target/weather masks, train split, cadence and bounds, then collate.
   Infeasible shape/source/caps fail; no hidden C/H substitution.

The loader retains reusable pool, QC and window-materialization mechanics;
the old task-index/shape-plan materialization branches are removed. Source and
C/H decisions are synchronized in DDP; station/cutoff sampling is rank-local.
Resume saves each rank's sampler state, pending descriptors and RNG.

## Training on an external compatible corpus

Prepare a copy of `configs/external_data.example.json` with absolute paths:
`binary_cache_registry`, `manifest_path`, `pool_dir`, `quality_dir`,
`region_profile`. Set role exactly `pretrain_train`. No file is downloaded
or private path automatically used. The training manifest must not overlap
the eleven Australian unseen stations in `baselines/australia_stations.json`.
This restriction applies to PVFM pretraining, not target-site full-shot baselines.

The registry/cache format is the original binary format: cache `metadata.json`,
`stations.csv`, task indices, and per-station `views/<resolution>/<station_key>/`
arrays (`target`, `target_mask`, historical/future covariates and masks,
`time_features`, `timestamps`, `site_features`, and scaler `metadata.json`).
The exact field contract is checked by the packaged binary loader. This is
an interface to already prepared data, **not a raw-CSV ingestion toolkit**.

First build a new pool/QC/profile (CPU only; all new directories inside this
package). Existing incompatible artifacts are rejected, not overwritten.

```bash
python train.py --data-config configs/my_data.json --prepare-only
```

Then, if four GPUs are available:

```bash
torchrun --standalone --nproc-per-node=4 train.py \
  --data-config configs/my_data.json --device cuda \
  --model-config configs/pvfm_3_9m.json --output outputs/pvfm_3_9m_80k
```

80k is the default, not a 60k run extended after LR decay. Adam uses beta1=0.9,
beta2=0.999, weight decay 0. Peak LR=1e-4, floor=1e-6; exact original WSD
5k warmup / 59k stable / 16k cosine decay. Per-rank batch=256. CUDA uses FP16
GradScaler. Loss is pure Q11 pinball; no added MSE, RevIN, clipping penalty,
early stopping or model-selection change. The last-step checkpoint is used
for fixed-window evaluation. Training validation/original orchestration is
not bundled; there is no unseen-test-driven model selection.

`--workers 0` is a portable default (the original launcher used more workers).
A dedicated DataLoader generator isolates iterator construction from dropout
RNG. Resume is tested **within this standalone trainer**. It does not claim
bitwise equivalence with the older GPU job or support its run-local `.resume`
sidecars. Data/pool/profile fingerprints, model config, batch size, world size
and schedule must match. Use a new output directory on each resume.

```bash
torchrun --standalone --nproc-per-node=4 train.py \
  --data-config configs/my_data.json --device cuda \
  --resume outputs/pvfm_3_9m_80k/step_010000.pt --output outputs/pvfm_3_9m_80k_resume
```

Change only `--model-config configs/pvfm_15_3m.json` or `configs/pvfm_45_0m.json` to
instantiate the other architectures. Their full-corpus 80k runs were not
repeated while assembling this package. GPU memory feasibility is not
asserted by the CPU smoke test.

## What can be reproduced without pretraining data?

- Exact model architecture / original trained PVFM-3.9M tensors.
- All exported inputs and predictions for the five test tasks.
- Common-Q9 metrics and aggregation, including regression references.
- Forward/backward, synthetic interruption/resume, original sampler tests.

Not reproducible from the demo alone: full 99-/79-station tables, the complete
pretraining run, global data filtering/ingestion or the historical GPU RNG
trajectory. This boundary is intentional, not an undisclosed dependency.
