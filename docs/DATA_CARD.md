# Data scope: Australian target stations only

The supplied forecast data is limited to the **11 Australian unseen stations**
listed in `baselines/australia_stations.json`. Their PV, weather and metadata
are in `datasets/australia/`, including the target-site training, validation
and test periods required for full-shot baseline retraining.

No Hong Kong or UK demonstration windows and no PVFM pretraining corpus are
distributed. The synthetic training/test fixtures contain no observations.

The Australian prepared evaluation NPZ arrays are **not included**. Frozen
window manifests and reference metrics are retained as protocol metadata;
they do not by themselves make checkpoint evaluation runnable. Evaluation
fails explicitly if its prepared arrays are absent. The retained raw CSVs
cannot be silently substituted because normalization and input contracts
must remain identical to the archived experiment.

See [AUSTRALIA_DATA.md](AUSTRALIA_DATA.md) for sources, provider attribution,
five-task definitions and the difference between raw and prepared data.
See [METRICS.md](METRICS.md) for the common-Q9 scoring convention.
