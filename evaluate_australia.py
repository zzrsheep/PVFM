"""Evaluate released PVFM-3.9M on all frozen Australia unseen windows, five tasks."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from australia import records, load, model_batch, require_prepared_arrays
from pvfm.runtime import load_model, model_inputs, safe_output
from pvfm.metrics import station_metrics, station_equal

ROOT = Path(__file__).resolve().parent


def evaluation_profile(name):
    profiles = json.loads((ROOT / "configs/evaluation_profiles.json").read_text())
    return profiles[name]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task")
    p.add_argument("--station")
    p.add_argument(
        "--model-profile",
        choices=["all_region", "australia_holdout"],
        default="all_region",
        help="all_region: unseen stations; australia_holdout: entire Australia excluded from pretraining",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--limit-windows",
        type=int,
        default=0,
        help="Smoke only; 0 means complete station test set",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    profile = evaluation_profile(args.model_profile)
    selected = records("pvfm", args.task, args.station, manifest=profile["manifest"])
    if not selected:
        p.error("No matching test contracts")
    try:
        require_prepared_arrays(selected)
    except FileNotFoundError as error:
        p.error(str(error))
    checkpoint = ROOT / profile["checkpoint"]
    with checkpoint.open("rb") as f:
        checkpoint_sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    if checkpoint_sha256 != profile["checkpoint_sha256"]:
        raise ValueError("Evaluation checkpoint fingerprint mismatch")
    out = safe_output(args.output)
    model, config = load_model(checkpoint, args.device)
    scores = []
    with torch.inference_mode():
        for row in selected:
            arrays = load(row)
            n = (
                row["windows"]
                if args.limit_windows <= 0
                else min(row["windows"], args.limit_windows)
            )
            preds = []
            ys = []
            ms = []
            for start in range(0, n, args.batch_size):
                ids = np.arange(start, min(n, start + args.batch_size))
                batch = model_batch(row, arrays, ids)
                preds.append(
                    model(**model_inputs(batch, args.device))[1]["quantiles"]
                    .float()
                    .cpu()
                    .numpy()
                )
                cuts = arrays["cutoffs"][ids]
                ix = cuts[:, None] + np.arange(row["H"])
                ys.append(arrays["target"][ix])
                ms.append(arrays["target_mask"][ix])
            q, y, m = map(np.concatenate, [preds, ys, ms])
            score = dict(
                task=row["task"],
                station_dir=row["station_dir"],
                windows=n,
                **station_metrics(q, y, m, model.probabilistic_quantiles)
            )
            scores.append(score)
            print(score, flush=True)
            np.savez_compressed(
                out / (row["task"] + "_" + row["station_key"] + ".npz"),
                quantiles=q,
                target=y,
                mask=m,
                cutoffs=arrays["cutoffs"][:n],
            )
    pd.DataFrame(scores).to_csv(out / "station_metrics.csv", index=False)
    pd.DataFrame(
        [
            dict(task=task, **station_equal([r for r in scores if r["task"] == task]))
            for task in sorted({r["task"] for r in scores})
        ]
    ).to_csv(out / "task_metrics.csv", index=False)
    (out / "protocol.json").write_text(
        json.dumps(
            dict(
                common_q9=True,
                model_profile=args.model_profile,
                checkpoint=profile["checkpoint"],
                checkpoint_sha256=checkpoint_sha256,
                manifest=profile["manifest"],
                window_limit=args.limit_windows,
                partial_smoke=bool(args.limit_windows),
                stations=len({r["station_dir"] for r in scores}),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
