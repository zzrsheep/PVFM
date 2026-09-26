# Training loader

The entry point is `pvfm/training_data.py`. There is one production sampling
policy: `AverageSupplyTemporalSampler`. This directory contains code, not
the private pretraining dataset.

| Responsibility | Modules |
|---|---|
| Read the existing binary-cache format and resolution registry | `binary_cache.py`, `multires_cache.py` |
| Validate arrays, split boundaries and masks | `cache_contract.py` |
| Reusable station/segment pool and sampler resume state | `temporal_pool_v1.py` |
| Fast legal-cutoff queries for actual C/H | `temporal_quality_index.py` |
| C/H distribution and variable-length batch assembly | `window_shapes.py`, `collate.py` |
| Region identity, probability normalization and supply counts | `regions.py` |
| Calibrate and load average-supply region probabilities | `temporal_average_supply.py` |
| Select source/C/H/region and draw legal windows | `sampler.py` |

The active order is source, actual C/H, feasible region using the calibrated
profile, and legal station/cutoff. The eligibility index excludes windows
crossing cadence breaks; final runtime mask and boundary checks remain.
Sampling without replacement within a batch is retained where supply permits;
smaller supports retain the documented replacement behavior and counters.

Removed from this package: the old indexed region-balanced batch sampler,
its shuffle-cycle/audit dependencies, origin thinning, conditional short-horizon
replay, special H=6 context selection and canonical-context override branches.
Unsupported experiment keywords raise an error, not a silent fallback.
No station-equal alternative is provided by the sampler base class.

Some **fixed full-data checkpoint fields** keep their historical names
(`pretrain_data_fraction=1.0`, an origin fingerprint and empty subset counters).
These preserve existing formal-run sampler fingerprints/resume state; they
are not configurable experiment features. Pool/QC/profile fingerprints and
pending-prefetch queues still protect against incompatible resumes.

The binary-cache wrappers retain their serialized format and metadata/index
reading paths. They are not raw-CSV preprocessing tools. Their nominal task
indices identify available station membership; temporal-pool candidates are
built from full station arrays, not the union of task-window origins.

Cleanup equivalence is recorded in `verification/loader_cleanup.json`.
Its synthetic tests are functional checks, not a new full-corpus training run.
