"""Aggregate reproduced seeds within station, then give each station equal weight."""

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["R2", "MAE", "RMSE", "CRPS", "AQL"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    a = p.parse_args()
    files = sorted(a.runs.rglob("common_q9_metrics.csv"))
    if not files:
        p.error("No common_q9_metrics.csv results")
    data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    keys = ["model", "mode", "task", "station_dir"]
    if data.duplicated(keys + ["seed"]).any():
        p.error("Duplicate station/seed runs; select one result per run")
    coverage = (
        data.groupby(keys, dropna=False).agg(seeds=("seed", "nunique")).reset_index()
    )
    counts = coverage.groupby(["model", "mode", "task"], dropna=False).agg(
        stations=("station_dir", "nunique"),
        min_seeds=("seeds", "min"),
        max_seeds=("seeds", "max"),
    )
    if not a.allow_partial and (
        (counts.stations != 11).any() or (counts.min_seeds != counts.max_seeds).any()
    ):
        p.error(
            "Incomplete station/seed coverage; inspect runs or explicitly use --allow-partial"
        )
    stations = data.groupby(keys, dropna=False)[METRICS].mean().reset_index()
    summary = (
        stations.groupby(["model", "mode", "task"], dropna=False)[METRICS]
        .mean()
        .join(counts)
        .reset_index()
    )
    out = a.output.resolve()
    if not out.is_relative_to(ROOT):
        p.error("Output must be inside the package")
    out.mkdir(parents=True, exist_ok=False)
    stations.to_csv(out / "station_seed_means.csv", index=False)
    summary.to_csv(out / "australia_station_equal.csv", index=False)
    coverage.to_csv(out / "coverage.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
