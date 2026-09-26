"""Frozen Australian station arrays and explicit test cutoffs (no training)."""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def records(role="pvfm", task=None, station=None, manifest=None):
    path = ROOT / (manifest or "datasets/australia_evaluation/manifest.json")
    rows = json.loads(path.read_text())["records"]
    return [
        r
        for r in rows
        if r["role"] == role
        and (task is None or r["task"] == task)
        and (station is None or station in r["station_dir"])
    ]


def require_prepared_arrays(rows):
    """Fail before loading weights or creating outputs when replay assets are absent."""
    missing = [row["file"] for row in rows if not (ROOT / row["file"]).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} prepared Australia evaluation arrays. "
            "These optional replay arrays are not distributed in this package; "
            "the retained raw CSVs are for baseline training, not a drop-in replacement. "
            "See docs/AUSTRALIA_DATA.md. No test windows were regenerated."
        )


def load(row):
    require_prepared_arrays([row])
    with np.load(ROOT / row["file"], allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def item(row, arrays, index):
    c, h = row["C"], row["H"]
    cut = int(arrays["cutoffs"][index])
    a, b = cut - c, cut + h
    target = arrays["target"]
    history = arrays["history"]
    future = arrays["future"]
    result = dict(
        past_target=target[a:cut].reshape(c, 1),
        future_target=target[cut:b].reshape(h, 1),
        historical_covariates=history[a:cut],
        future_covariates=future[cut:b],
        past_observed_mask=arrays["target_mask"][a:cut].reshape(c, 1),
        future_observed_mask=arrays["target_mask"][cut:b].reshape(h, 1),
        historical_covariates_mask=arrays["history_mask"][a:cut],
        future_covariates_mask=arrays["future_mask"][cut:b],
        seq_len=c,
        pred_len=h,
        metadata=dict(
            timestamps=arrays["timestamps_ns"][a:b].astype("datetime64[ns]"),
            history_covariate_cols=row["weather_columns"],
            future_covariate_cols=row["weather_columns"],
        ),
    )
    if "time" in arrays:
        result.update(
            past_time_features=arrays["time"][a:cut],
            future_time_features=arrays["time"][cut:b],
            static_features=arrays["static"],
            site_features=arrays["site"],
        )
    return result


def model_batch(row, arrays, indices):
    from pvfm.runtime import INPUT_FIELDS

    samples = [item(row, arrays, int(i)) for i in indices]
    return {
        **{k: np.stack([s[k] for s in samples]) for k in INPUT_FIELDS},
        "resolutions": [row["resolution"]] * len(samples),
    }
