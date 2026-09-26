"""Evaluate satellite ON/OFF using the exact same prepared test windows."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .checkpoint import DEFAULT_CHECKPOINT, load_model
from .data import WindowDataset, model_inputs, synthetic_batch, file_hash
from .metrics import PointMetrics, station_equal


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    totals = {arm: PointMetrics() for arm in ("on", "off")}
    stations = {arm: defaultdict(PointMetrics) for arm in totals}
    sample_count = frame_count = valid_frames = 0
    for batch in batches:
        inputs = model_inputs(batch, device)
        predictions = {"on": model(**inputs)}
        predictions["off"] = model(
            **dict(
                inputs,
                satellite_frame_mask=torch.zeros_like(inputs["satellite_frame_mask"]),
            )
        )
        target = batch["future_target"].numpy()
        mask = batch["future_target_mask"].numpy()
        for arm, prediction in predictions.items():
            values = prediction.cpu().numpy()
            totals[arm].update(values, target, mask)
            for i, station in enumerate(batch["station_id"]):
                stations[arm][station].update(values[i], target[i], mask[i])
        sample_count += len(target)
        frame_count += inputs["satellite_frame_mask"].numel()
        valid_frames += int(inputs["satellite_frame_mask"].sum())
    if not sample_count or not totals["on"].count:
        raise ValueError("No valid evaluation targets")
    return {
        "samples": sample_count,
        "metric_space": "capacity_factor",
        "satellite_frame_coverage": valid_frames / frame_count,
        "pooled": {k: v.result() for k, v in totals.items()},
        "station_equal": {
            k: station_equal([a.result() for a in v.values()])
            for k, v in stations.items()
        },
        "by_station": {
            s: {arm: stations[arm][s].result() for arm in totals}
            for s in stations["on"]
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--manifest", type=Path)
    data.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    device = torch.device(args.device)
    model, payload = load_model(args.checkpoint, device)
    if args.synthetic:
        batches = [synthetic_batch(2)]
        fingerprint = None
    else:
        dataset = WindowDataset(args.manifest, "test")
        batches = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        fingerprint = dataset.fingerprint
    report = evaluate(model, batches, device)
    report.update(
        synthetic=args.synthetic,
        checkpoint_sha256=file_hash(args.checkpoint),
        dataset_fingerprint=fingerprint,
        adaptation_step=payload["adaptation_step"],
        quantile_calibration_claimed=False,
    )
    args.output.mkdir(parents=True)
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report["station_equal"], indent=2))


if __name__ == "__main__":
    main()
