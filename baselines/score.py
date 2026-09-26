"""Rescore raw full-shot quantiles using the release's common-Q9 metrics."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pvfm.metrics import station_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    launch = json.loads((args.run / "launch.json").read_text())
    job = launch["job"]
    found = list((args.run / "results").rglob("pred_quantiles.npy"))
    if len(found) != 1:
        raise ValueError(f"Expected exactly one quantile output: {found}")
    path = found[0]
    pred = np.load(path)
    target = np.load(path.with_name("quantile_true.npy"))
    levels = np.load(path.with_name("quantile_levels.npy"))
    scores = station_metrics(pred, target, np.isfinite(target), levels)
    pd.DataFrame([{**job, **scores}]).to_csv(
        args.run / "common_q9_metrics.csv", index=False
    )
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
