"""Six upstream TSFM adapters on frozen Australia windows; isolated envs required."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE / "runtime"), str(ROOT)]
from australia import records, load, item, require_prepared_arrays
from pvfm.metrics import EVAL_LEVELS, station_metrics, station_equal


def predictor(name, checkpoint, c, h, device, batch_size):
    """Use original adapters, not reimplementations of upstream architectures."""
    if name == "chronos2":
        from chronos import Chronos2Pipeline
        from foundation.tsfm_zero_shot.chronos2_covariate import (
            build_chronos2_input,
            quantile_forecasts,
        )

        model = Chronos2Pipeline.from_pretrained(checkpoint, device_map=device)

        def predict(items):
            result = model.predict(
                [build_chronos2_input(x) for x in items],
                prediction_length=h,
                context_length=c,
                batch_size=batch_size,
                cross_learning=False,
            )
            return quantile_forecasts(result, list(model.quantiles), EVAL_LEVELS, h)

    elif name == "citras":
        import torch
        from foundation.tsfm_zero_shot.citras_fm_covariate import (
            build_citras_input,
            quantile_forecasts,
            model_quantile_levels,
        )
        from scripts.eval_citras_fm_v2_balanced_zero_shot import (
            _load_model,
            _forecast_batch,
        )

        opts = SimpleNamespace(
            model_id=checkpoint,
            revision=None,
            local_files_only=True,
            device=device,
            hf_cache_dir=None,
        )
        model = _load_model(opts).to(device).eval()

        def predict(items):
            built = [build_citras_input(x) for x in items]
            target = torch.as_tensor(
                np.stack([x["target"] for x in built]), device=device
            )
            known = torch.as_tensor(
                np.stack([x["known_cov"] for x in built]), device=device
            )
            with torch.inference_mode():
                _, values = _forecast_batch(model, target, known, h)
            return quantile_forecasts(
                values, model_quantile_levels(model), EVAL_LEVELS, h
            )

    elif name == "moirai2":
        from uni2ts.model.moirai2 import Moirai2Module
        from foundation.tsfm_zero_shot.moirai2_covariate import (
            build_moirai2_dataset,
            forecast_quantiles,
        )
        from scripts.eval_moirai2_v2_balanced_zero_shot import _make_predictor

        module = Moirai2Module.from_pretrained(checkpoint).eval()
        model = _make_predictor(
            module,
            context_length=c,
            prediction_length=h,
            covariate_dim=6,
            batch_size=batch_size,
            device=device,
        )

        def predict(items):
            data, ids, _ = build_moirai2_dataset(items)
            forecasts = list(model.predict(data))
            if [str(x.item_id) for x in forecasts] != list(map(str, ids)):
                raise ValueError("Moirai reordered the forecast windows")
            return forecast_quantiles(forecasts, EVAL_LEVELS, h)

    elif name == "timesfm3":
        from foundation.tsfm_zero_shot.timesfm3_covariate import (
            build_timesfm3_input,
            quantile_forecasts,
            model_quantile_levels,
        )
        from scripts.eval_timesfm3_v2_balanced_zero_shot import (
            _load_timesfm3,
            _predict_batch,
        )

        opts = SimpleNamespace(
            model_id=checkpoint,
            model_revision=None,
            hf_cache_dir=None,
            local_files_only=True,
            per_core_batch_size=batch_size,
            device=device,
            use_symmetric_averaging=False,
            make_positive=False,
            use_znorm=False,
            padding_mode="none",
        )
        model, _ = _load_timesfm3(opts)

        def predict(items):
            outputs = _predict_batch(
                model, [build_timesfm3_input(x) for x in items], opts, h
            )
            return quantile_forecasts(
                outputs, model_quantile_levels(model), EVAL_LEVELS, h
            )

    elif name == "tirex2":
        from tirex_official import TiRex2Model
        from foundation.tsfm_zero_shot.fev_batched_windows import (
            make_batched_fev_window,
        )
        from foundation.tsfm_zero_shot.fev_frozen_windows import FrozenWindowTask
        from scripts.eval_fev_strict_zero_shot_shard import extract_forecasts

        model = TiRex2Model(
            model_path=checkpoint,
            batch_size=batch_size,
            device=device,
            as_univariate=False,
        )

        def predict(items):
            window = make_batched_fev_window(
                items, [str(i) for i in range(len(items))], include_covariates=True
            )
            result = model.fit_predict(FrozenWindowTask([window]))
            _, quantiles = extract_forecasts(result[0], len(items), h)
            return np.stack([quantiles[str(float(q))] for q in EVAL_LEVELS], axis=-1)

    elif name == "tabpfn3":
        from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline
        from foundation.tsfm_zero_shot.fev_frozen_windows import (
            FrozenWindowTask,
            make_fev_window,
        )
        from scripts.eval_tabpfn_ts3_v2_balanced_optimized_shard import (
            extract_forecasts,
        )

        model = TabPFNTSPipeline(
            tabpfn_mode=TabPFNMode.LOCAL, tabpfn_model_config={"model_path": checkpoint}
        )

        def predict(items):
            values = []
            for i, sample in enumerate(items):
                result, _ = model.predict_fev(
                    FrozenWindowTask([make_fev_window(sample, str(i))]),
                    use_covariates=True,
                )
                _, q = extract_forecasts(result[0], h)
                values.append(q)
            return np.stack(values)

    else:
        raise ValueError(name)
    return predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        choices=["chronos2", "citras", "moirai2", "timesfm3", "tirex2", "tabpfn3"],
        required=True,
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Downloaded upstream weights; TabPFN uses its .ckpt file",
    )
    p.add_argument("--task", required=True)
    p.add_argument("--station")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--limit-windows", type=int, default=0)
    p.add_argument(
        "--input-role",
        choices=["fullshot", "pvfm"],
        default="fullshot",
        help="Explicit weather-preprocessing contract; see docs/BASELINES.md",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.batch_size < 1 or a.limit_windows < 0:
        p.error("Invalid batch size/window limit")
    checkpoint = a.checkpoint.resolve()
    if not checkpoint.exists():
        p.error("Download upstream weights first; checkpoint path does not exist")
    selected = records(a.input_role, a.task, a.station)
    if not selected:
        p.error("No matching windows")
    try:
        require_prepared_arrays(selected)
    except FileNotFoundError as error:
        p.error(str(error))
    out = a.output.resolve()
    if not out.is_relative_to(ROOT):
        p.error("Output must be inside the package")
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(out / "mplcache"))
    (out / "protocol.json").write_text(
        json.dumps(
            {
                **vars(a),
                "checkpoint": str(checkpoint),
                "quantiles": EVAL_LEVELS.tolist(),
                "station_equal": True,
                "partial_smoke": bool(a.limit_windows),
                "historical_score_parity_claimed": False,
            },
            default=str,
            indent=2,
        )
    )
    predict = predictor(
        a.model,
        str(checkpoint),
        selected[0]["C"],
        selected[0]["H"],
        a.device,
        a.batch_size,
    )
    scores = []
    for row in selected:
        arrays = load(row)
        n = min(row["windows"], a.limit_windows) if a.limit_windows else row["windows"]
        predictions = []
        for start in range(0, n, a.batch_size):
            samples = [
                item(row, arrays, i) for i in range(start, min(n, start + a.batch_size))
            ]
            q = np.asarray(predict(samples))
            if q.shape != (len(samples), row["H"], 9) or not np.isfinite(q).all():
                raise ValueError(f"Invalid upstream forecast shape/values: {q.shape}")
            predictions.append(q)
        q = np.concatenate(predictions)
        ix = arrays["cutoffs"][:n, None] + np.arange(row["H"])
        target, mask = arrays["target"][ix], arrays["target_mask"][ix]
        score = dict(
            model=a.model,
            task=a.task,
            station_dir=row["station_dir"],
            windows=n,
            **station_metrics(q, target, mask, EVAL_LEVELS),
        )
        scores.append(score)
        np.savez_compressed(
            out / (row["station_key"] + ".npz"),
            quantiles=q,
            target=target,
            mask=mask,
            cutoffs=arrays["cutoffs"][:n],
        )
        pd.DataFrame(scores).to_csv(out / "station_metrics.csv", index=False)
        print(score, flush=True)
    pd.DataFrame([dict(model=a.model, task=a.task, **station_equal(scores))]).to_csv(
        out / "task_metrics.csv", index=False
    )


if __name__ == "__main__":
    main()
