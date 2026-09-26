# Satellite-enhanced PVFM

This add-on contains one model path and the selected step-2,500 checkpoint.
It uses the shared `pvfm/` layers in the parent repository, without changing
the original PVFM model or its forecasting checkpoints. Run commands from
the repository root, not from this directory. Use the parent requirements.

## Model and weight

The task is **hourly resolution, 16 hours of history, four forecast hours**.
It is not the 15-minute C64/H16 benchmark.

```
PV history (historical weather masked) -> frozen PVFM history encoder
                                             |
Historical satellite images -> satellite encoder -> gated PV residual
                                             |
                     frozen history-to-future projection
                     -> frozen future-NWP decoder + refinement -> q50
```

- Backbone: 3,879,252 active parameters, width 128, eight heads, 6/6
  encoder/decoder layers. It was already adapted using q50 MSE before
  satellite training and is frozen throughout satellite adaptation.
- Satellite encoder and fusion: **621,121 trainable parameters**. Four image
  channels, 64x64 pixels, 8x8 spatial patches, width 128, one spatial and one
  temporal attention layer, four attention heads, dropout zero.
- Spatial attention uses station-relative east/north coordinates with 2-D
  rotary encoding. Temporal windows use the same length-12, stride-6 layout
  as the PV backbone. Frozen position, solar and site/window embeddings
  anchor satellite memory to the PV representation.
- Each historical PV patch queries its aligned satellite memory. A learned
  gate scales a residual initialized to zero. Missing images are masked;
  an entirely missing satellite window contributes exactly zero residual.
- **No history weather does not mean no meteorological inputs.** Six future
  NWP variables still condition the decoder. There is no historical-weather
  input argument in this release, so it cannot be accidentally re-enabled.
- The original ordered Q11 head is retained. This downstream experiment
  optimizes and reports **q50 point predictions only**. The other quantiles
  were not recalibrated; no downstream CRPS/AQL claim is made.

`checkpoints/satellite_forecaster.pt` includes both backbone and adapter.
It is an inference export: no private paths, station lists, original optimizer
state or unrelated experiment settings. Active tensor names and values are
unchanged. Six inactive calendar/scalar-head tensors were removed.
This exported weight cannot resume the original research optimizer at step
2,500. The clean trainer can resume **its own** training checkpoints.

## Quick checks

```bash
python -m satellite.verify
python -m pytest -q satellite/tests
python -m satellite.evaluate --synthetic --device cpu \
  --output outputs/satellite_smoke_eval
python -m satellite.train --synthetic --device cpu --stop-after 2 \
  --output outputs/satellite_smoke_train
```

Synthetic data checks execution only. Its metrics are not experimental results.
All output directories must be new; existing results are never overwritten.

## Evaluate real data

The private Anhui PV/NWP/satellite arrays and original cache paths are **not
included**. No real images or station traces were copied into this add-on.
See [DATA.md](DATA.md) for the portable, pre-aligned tensor contract.
The interface deliberately does not guess station cohorts, normalization,
missing frames or forecast origins from arbitrary raw files.

```bash
python -m satellite.evaluate --manifest data/satellite/test/manifest.json \
  --device cuda --batch-size 32 --output outputs/satellite_test
```

ON and OFF use identical windows and frozen weights. OFF zeros only the image
availability mask; it does not disable future NWP. Reported MAE, RMSE and R2
use valid capacity-factor targets, both pooled and station-equal. R2 is null
for near-constant targets (total centered sum of squares <= 1e-6).
ON/OFF measures the contribution of the trained adapter, not image-content
causality. The archived `reference_metrics.json` comes from the original full
test run and is clearly separate from fresh evaluation output.

## Adapter training

This reuses the **already q50-adapted frozen backbone** from the exported
checkpoint and freshly initializes the satellite adapter. It is not PVFM
pretraining, and does not repeat the preceding q50 backbone fine-tuning.

```bash
torchrun --standalone --nproc_per_node=2 -m satellite.train \
  --train-manifest data/satellite/train/manifest.json \
  --val-manifest data/satellite/val/manifest.json \
  --device cuda --num-workers 4 --output outputs/satellite_adaptation
```

The original recipe is fixed in `config.json`: two GPUs, 32 windows per GPU,
AdamW at 1e-4 with weight decay 0.01, cosine decay to 1e-6 over 5,000 successful
updates, gradient clipping at 1.0, FP16 and complete seen validation every 500
steps. The objective is globally mask-weighted q50 MSE. All backbone layers
stay in evaluation mode. The frozen decoder remains differentiable with
respect to its input so adapter gradients propagate through it. AMP overflow
skips are synchronized across ranks. Best checkpoint selection uses seen
validation only. A smoke stop does not change the full learning-rate schedule.

```bash
torchrun --standalone --nproc_per_node=2 -m satellite.train \
  --resume outputs/satellite_adaptation/latest.pt \
  --train-manifest data/satellite/train/manifest.json \
  --val-manifest data/satellite/val/manifest.json \
  --device cuda --num-workers 4 --output outputs/satellite_resumed
```

Resume rejects changed data fingerprints, world size or recipe. The trimmed
backbone constructor no longer initializes unused modules, so fresh adapter
initialization is **not claimed to replay the research run's RNG sequence**.
Without the full aligned training data, this package cannot reproduce the
original training trajectory or benchmark numbers from scratch.

## Source map and verification

| File | Responsibility |
|---|---|
| `model.py` | Freeze contract and satellite residual insertion |
| `backbone.py` | Expose the history/decoder boundary using shared PVFM layers |
| `encoder.py`, `attention.py` | Satellite spatial/temporal attention |
| `fusion.py`, `anchors.py` | Gated residual and frozen PVFM anchors |
| `alignment.py`, `coordinates.py` | No-future-frame check and geographic coordinates |
| `data.py` | Checksummed, pre-aligned tensor shards |
| `train.py`, `evaluate.py`, `metrics.py` | Adapter training and paired point evaluation |
| `checkpoint.py`, `verify.py` | Strict weight loading and file integrity |

`verification/parity.json` records exact CPU prediction parity on 51 real
windows across 17 stations, both ON/OFF, plus synthetic missing-frame cases
and exact adapter gradients. This is a bounded equivalence audit, not a
rerun of all 8,554 test windows or a GPU numerical-parity guarantee.
