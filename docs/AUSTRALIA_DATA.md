# Australian unseen cohort: full-shot training and frozen evaluation

This package includes **11 physical Australian unseen stations**, with the
station's available PV, historical weather, future weather, and metadata
needed to construct **train / validation / test** splits. It does not include
the 746-station PVFM pretraining corpus. The exported cohort was checked to
have zero intersection with that pretraining manifest.

**Current distribution:** raw station CSVs and metadata remain included.
Prepared evaluation NPZ arrays are not included; only their frozen manifests
and hourly origin lists remain. Thus full-shot baseline training can read its
raw inputs, but PVFM and external-TSFM frozen-array evaluation requires separate
replay assets. The evaluator stops explicitly before model loading when these
assets are absent. No replacement preprocessing is silently applied.

The station list is `baselines/australia_stations.json`: Bannerton, Bomen,
Columboola, Gannawarra, Haughton Stage 1, Karadoc, Mugga Lane, Stubbo 1,
Wemen, Wollar, and Woolooga. Do not interpret these as a random sample of all
Australian PV stations or as a newly designed benchmark.

## Data lineage and redistribution

PV measurements were obtained from public **Australian Energy Market
Operator (AEMO) NEM** dispatch/SCADA data. Retain AEMO attribution and the
station-level source notes shipped beside the data. AEMO's
[copyright permissions](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions)
apply to AEMO-owned public material, not confidential or third-party material.
Weather is sourced through **Open-Meteo**, whose data attribution/CC-BY-4.0
conditions are described in its [terms](https://open-meteo.com/en/terms).
Access/API service conditions and redistribution of downloaded data are not
the same thing. These are processed research exports, not untouched upstream
downloads, and providers do not endorse this package.

Source histories are in each `README.md`, `other_data.json`, and, where
present, `future_nwp_1h_linear_info.json`. Those records retain historical
source paths for provenance only. Runtime data locations are portable.
For example, Bannerton records use AEMO dispatch/SCADA material, a station
capacity entry and hourly-to-native weather processing; other stations' exact
date ranges and metadata are kept separately rather than assumed identical.

## Two representations

- `datasets/australia/<version>/<station>/`: station-level CSVs and metadata
  for the original full-shot loader. Different version folders preserve
  genuine differences in time-axis padding/weather files; they do not
  represent additional physical stations. Each job points to its own version.
- `datasets/australia_evaluation/`: retained frozen manifests and hourly test
  cutoff lists. `manifest.json` records the expected optional prepared-array
  hashes, station, C/H, weather columns, role, and window count. The referenced
  NPZ files are not shipped. The prepared-array descriptions below specify the
  archived replay contract, not the present files in the distribution.

Raw CSV time series contain all splits. Only the original loader's train
partition is used for training/scaler fitting; validation selects checkpoints
and test is reserved for final scoring. Split ratios and original command
flags remain in the copied job/parser. This package does not relabel dropped
pretraining stations as zero-shot stations, re-sample test origins, or
interpolate missing PV to manufacture additional windows.

Prepared arrays contain test labels as well as inputs, because metrics need
labels. `model_inputs()` excludes `future_target`; external adapters follow
the same separation. Missingness is represented by masks (and NaNs in raw
native time-axis-padded files). Preserve these, capacity-factor scaling, and
the original per-station scaler treatment.

## Five tasks and contracts

| Manifest task | C/H in data points | C → H in hours |
|---|---|---|
| `1h_C24_H6` | 24 / 6 | 24 → 6 |
| `1h_C72_H24` | 72 / 24 | 72 → 24 |
| `1h_C336_H168` | 336 / 168 | 336 → 168 |
| `15min_C64_H16` | 64 / 16 | 16 → 4 |
| `15min_C96_H24` | 96 / 24 | 24 → 6 |

No 15min 96h→24h task is included. The prepared `pvfm` role preserves the
released PVFM-3.9M checkpoint's hourly full-shot-origin-aligned cache inputs and
native-15min verified replay inputs. The `fullshot` role preserves verified
per-station full-shot prepared arrays for the external-model runner. The
hourly weather normalization can differ between roles; they are explicitly
named, not silently treated as interchangeable.

The Australia hold-out profile is separately bound in
`configs/evaluation_profiles.json`. Its manifest is
`datasets/australia_holdout_evaluation/manifest.json`. All 55 station/task
array sets were checked against the original hourly hold-out loader and
the native-15min full-shot replay. They happen to be identical to the
existing PVFM prepared arrays and are referenced without duplication;
this was verified, not assumed from the model architecture. The checkpoint
differs and is strictly bound by SHA256. The old native cache-only
evaluation is NOT used for the two 15-minute tasks.

The current main paper's 15min/4h common-68 cohort contains only four of these
Australian sites. This package's **11-site regional** average must not be
labelled that common-68 table's Australian slice. Baseline families with
shorter-context interfaces retain their original archived commands; the
canonical evaluation C/H does not retroactively change those architectures.

## What can and cannot be reproduced

You can train the included full-shot baselines from scratch on these 11
targets and score their predictions in capacity-factor units on common Q9.
Running the released PVFM checkpoint through the frozen evaluation protocol
requires the separate prepared arrays. All six external TSFM adapters are
included but require both those arrays and upstream packages/weights.

Four complete baseline groups (11 Australian sites each, seed 2021) were
retrained after packaging: three match all five station metrics exactly;
iTransformer-TC 15min/4h differs slightly and remains explicitly unresolved.
See `reference/baseline_reproduction/`. Earlier bounded audits are in
`verification/australia_report.json`; that earlier report predates the
44 full runs. Some external historical hourly weather contracts differ;
see `BASELINES.md` before comparing numbers.
GPU training is not promised bitwise deterministic across environments.

The NWP files are archived **valid-time** covariates. The package does not
prove they were available at each operational forecast issue time. Original
1h→15min weather handling is retained, not relabelled as native 15-minute
forecast production. No new data-quality or operational-vintage claim is
introduced by exporting these experiments.
