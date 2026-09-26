# Prepared-window contract

No private station data is bundled. The clean data loader reads already
aligned windows; it does not reproduce the original private raw-data pipeline.
An authorized data provider must export its approved windows with this contract.
Never create train/validation windows from unseen test stations.

Example `manifest.json`, beside its `.npz` shards:

```json
{
  "format": "satellite_windows_v1",
  "split": "test",
  "cohort": "unseen",
  "resolution_minutes": 60,
  "context_steps": 16,
  "horizon_steps": 4,
  "shards": [
    {"file": "windows_000.npz", "samples": 32, "sha256": "<actual SHA256>"}
  ]
}
```

Allowed splits are `train`, `val`, `test`; train/val require `cohort=seen`.
The checksum is calculated on the exact NPZ file bytes. The manifest preserves
shard and row order. Two shards per worker are cached; keep shards small (for
example, 8--32 windows). Compressed NPZ is convenient for portability, not a
claim of production raw-cache throughput.

Each NPZ uses NumPy numeric arrays (no object arrays/pickle), with leading
sample axis N:

| Key | Shape | Meaning |
|---|---|---|
| `past_target` | N,16,1 | Capacity-factor PV history |
| `past_observed_mask` | N,16,1 | Float32 0/1 |
| `future_covariates` | N,4,6 | Original normalized future NWP, original channel order |
| `future_covariates_mask` | N,4,6 | Float32 0/1 |
| `past_time_features` | N,16,5 | Original normalized minute/hour/weekday/day/day-of-year |
| `future_time_features` | N,4,5 | Same five-field schema |
| `static_features` | N,4 | Original collated C/H/label-length/duration values |
| `site_features` | N,5 | Original site tensor; lat at 0, lon at 1, UTC offset at 4 |
| `satellite_history` | N,16,4,64,64 | Original image values, before checkpoint channel normalization |
| `satellite_frame_mask` | N,16 | Boolean available-frame flags |
| `satellite_patch_coords` | N,64,2 | East/north AEQD patch coordinates divided by 1,600 km |
| `satellite_frame_time_ns` | N,16 | Int64 site-local-clock timestamps |
| `forecast_origin_time_ns` | N | Int64 timestamp of **first predicted hour** |
| `future_target` | N,4,1 | Capacity-factor observations |
| `future_target_mask` | N,4,1 | Float32 0/1 |
| `station_id` | N | Unicode station key, no machine filesystem path |

Floating tensors use float32. No historical weather is required. No model
argument accepts future PV targets; they are used only for loss/scoring.
The satellite values must not be normalized twice: mean/std are stored in the
checkpoint and applied by the image encoder. NWP values must retain the
original training-partition scaling; fitting a scaler on test data is invalid.

Time slots for available satellite frames must be exactly origin-16h through
origin-1h, in order. The origin itself is already future information and is
rejected. A missing frame may carry the int64 NaT sentinel, but must be masked.
No filling/interpolation or relaxed matching is applied by this loader.
Use the corrected geographic grids with
`geographic_grid_to_patch_coordinates(..., patch_size=8, coordinate_scale_km=1600)`;
do not replace real station grids with a synthetic regular grid.

For strict replay, export the original approved train/val/test membership,
including source order, masks, target scaling and cutoff stride. Tensor
substitution, different cohorts or recomputed origins cannot be called the
same evaluation protocol. No generic CSV-to-benchmark conversion is claimed.
