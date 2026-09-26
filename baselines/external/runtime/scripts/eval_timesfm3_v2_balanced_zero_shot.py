# Mechanically extracted; original definitions and internal namespaces retained.
from __future__ import annotations

import argparse

import importlib.metadata

import inspect

from typing import Any

import numpy as np

def _load_timesfm3(args: argparse.Namespace) -> tuple[Any, str]:
    """Load the official package lazily so CPU bookkeeping tests need no dependency."""
    try:
        timesfm3 = importlib.import_module("timesfm3")
    except ImportError as error:
        raise RuntimeError(
            "TimesFM-3 is not installed in this Python environment. Install the isolated "
            "TimesFM-3 environment from requirements_timesfm3.txt; do not replace the "
            "project's pinned old timesfm package."
        ) from error

    model_config_cls = getattr(timesfm3, "ModelConfig", None)
    if model_config_cls is None:
        raise RuntimeError("Installed timesfm3 package does not expose ModelConfig.")
    model_cls = getattr(timesfm3, "TimesFM3Evaluator", None)
    api_name = "TimesFM3Evaluator"
    if model_cls is None:
        model_cls = getattr(timesfm3, "TimesFM3Forecaster", None)
        api_name = "TimesFM3Forecaster"
    if model_cls is None:
        raise RuntimeError("Installed timesfm3 package exposes neither TimesFM3Evaluator nor TimesFM3Forecaster.")

    config_kwargs: dict[str, Any] = {
        "checkpoint_path": args.model_id,
        "per_core_batch_size": args.per_core_batch_size,
        "device": args.device,
        "revision": args.model_revision,
        "cache_dir": str(args.hf_cache_dir) if args.hf_cache_dir else None,
        "local_files_only": bool(args.local_files_only),
    }
    # Keep compatibility with an early release that may not expose all Hub options.
    supported = set(inspect.signature(model_config_cls).parameters)
    config_kwargs = {key: value for key, value in config_kwargs.items() if key in supported and value is not None}
    config = model_config_cls(**config_kwargs)
    return model_cls(config), api_name


def _predict_batch(model: Any, inputs: list[dict[str, np.ndarray | None]], args: argparse.Namespace, horizon: int) -> list[Any]:
    contexts = [item["context"] for item in inputs]
    past_only = [item["past_only_covariates"] for item in inputs]
    past_future = [item["past_future_covariates"] for item in inputs]
    outputs = model.predict_batch(
        contexts=contexts,
        horizon=horizon,
        past_only_covariates=past_only,
        past_future_covariates=past_future,
        return_quantiles=True,
        use_symmetric_averaging=args.use_symmetric_averaging,
        make_positive=args.make_positive,
        sort_quantiles=True,
        use_znorm=args.use_znorm,
        padding_mode=args.padding_mode,
    )
    return list(outputs)

