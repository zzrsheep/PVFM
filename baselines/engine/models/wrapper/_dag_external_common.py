"""Small helpers for loading the unmodified DAG benchmark sources."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "external_dag"


def external_root(configs) -> Path:
    return Path(getattr(configs, "dag_source_root", DEFAULT_ROOT)).expanduser().resolve()


def load_external(module_name: str, configs):
    root = external_root(configs)
    if not root.is_dir():
        raise FileNotFoundError(f"DAG benchmark source root is missing: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return importlib.import_module(module_name)


def covariate_dims(configs):
    hist = [x.strip() for x in str(getattr(configs, "history_covariate_cols", "")).split(",") if x.strip()]
    future = [x.strip() for x in str(getattr(configs, "future_covariate_cols", "")).split(",") if x.strip()]
    if not hist or not future:
        raise ValueError("This adapter requires both history_covariate_cols and future_covariate_cols")
    if len(hist) != len(future):
        raise ValueError(
            "External DAG benchmark models require equal history/future covariate widths; "
            f"got history={len(hist)} future={len(future)}"
        )
    return len(hist), len(future)


def base_config(configs, **overrides):
    """Translate PVFM argparse names to the original model config names."""
    hist_dim, future_dim = covariate_dims(configs)
    values = dict(
        task_name="long_term_forecast",
        features="MS",
        freq=getattr(configs, "freq", "h"),
        seq_len=int(configs.seq_len),
        label_len=int(getattr(configs, "label_len", configs.pred_len)),
        pred_len=int(configs.pred_len),
        horizon=int(configs.pred_len),
        enc_in=1 + hist_dim,
        dec_in=future_dim,
        c_out=1,
        series_dim=1,
        d_model=int(getattr(configs, "d_model", 256)),
        d_ff=int(getattr(configs, "d_ff", 512)),
        e_layers=int(getattr(configs, "e_layers", 1)),
        d_layers=int(getattr(configs, "d_layers", 1)),
        n_heads=int(getattr(configs, "n_heads", 4)),
        factor=int(getattr(configs, "factor", 1)),
        patch_len=int(getattr(configs, "patch_len", 24)),
        stride=int(getattr(configs, "stride", getattr(configs, "patch_len", 24))),
        dropout=float(getattr(configs, "dropout", 0.1)),
        activation=getattr(configs, "activation", "gelu"),
        use_norm=bool(int(getattr(configs, "use_norm", 1))),
        use_future_exog=True,
        use_future=1,
        covariate_dim=hist_dim + future_dim,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def reorder_target_first(x):
    """PVFM stores PV in the final channel; the benchmark cores use it first."""
    return x[..., [-1] + list(range(x.shape[-1] - 1))]


def future_covariates(x_dec, pred_len):
    if x_dec is None or x_dec.shape[-1] <= 1:
        raise ValueError("PVFM decoder input does not contain future covariates")
    return x_dec[:, -pred_len:, :-1]
