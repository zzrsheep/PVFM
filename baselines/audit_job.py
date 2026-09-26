"""Bounded audit using the copied parser, dataset and model; no full training."""

import ast
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.run import load_config


def parse_original():
    path = Path.cwd() / "run.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test):
            body = []
            for child in node.body:
                if isinstance(child, ast.If) and ast.unparse(child.test).startswith(
                    "args.task_name =="
                ):
                    break
                body.append(child)
            node.body = body
    ns = {"__name__": "__main__", "__file__": str(path)}
    exec(compile(tree, str(path), "exec"), ns)
    return ns["args"]


def main():
    torch.set_num_threads(2)
    args = parse_original()
    args.use_gpu = False
    args.use_amp = False
    args.num_workers = 0
    args.batch_size = 2
    args.eval_batch_size = 2
    args.enable_station_cache = False
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from torch.utils.data import DataLoader, Subset

    experiment = Exp_Long_Term_Forecast(args)
    report = {"model": args.model, "task": args.task_spec_name, "splits": {}}
    for split in ["train", "val", "test"]:
        data, _ = experiment._get_data(split)
        batch = next(
            iter(DataLoader(Subset(data, list(range(min(2, len(data))))), batch_size=2))
        )
        x, y, xm, ym, metadata, extras = experiment._unpack_batch(batch)
        x, y, xm, ym = [t.float() for t in [x, y, xm, ym]]
        extras = experiment._move_model_extras(extras, "audit", metadata)
        experiment.model.train(split == "train")
        result = experiment._model_forward(
            experiment.model,
            x,
            xm,
            experiment._build_decoder_input(y, metadata),
            ym,
            extras,
            y,
        )
        loss, *_ = experiment._forecast_loss(result, y, metadata)
        auxiliary = experiment._model_aux_loss()
        if auxiliary is not None:
            loss = loss + auxiliary
        assert torch.isfinite(loss), loss
        if split == "train":
            loss.backward()
            assert all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in experiment.model.parameters()
            )
            experiment._select_optimizer().step()
            experiment.model.zero_grad(set_to_none=True)
        report["splits"][split] = {
            "windows": len(data),
            "sample_loss": float(loss.detach()),
            "input_shape": list(x.shape),
            "target_shape": list(y.shape),
        }
        # Check the full ordered evaluation cutoff set, not just two samples.
        if split == "test":
            launch = Path(os.environ["PVFM_RESULTS_ROOT"]).parent / "launch.json"
            job = json.loads(launch.read_text())["job"]
            config = load_config(job)
            if config.get("reference_eval_origins"):
                frozen = (
                    Path(__file__).resolve().parents[1]
                    / config["reference_eval_origins"]
                )
                expected = pd.read_csv(frozen)
                actual = [
                    str(data[i][4]["future_start_time"]) for i in range(len(data))
                ]
                np.testing.assert_array_equal(
                    pd.to_datetime(actual).to_numpy(),
                    pd.to_datetime(expected.future_start_time).to_numpy(),
                )
                report["all_test_origins_exact"] = True
    report["parameters"] = sum(p.numel() for p in experiment.model.parameters())
    output = Path(os.environ["PVFM_RESULTS_ROOT"]).parent / "audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
