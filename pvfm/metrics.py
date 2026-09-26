"""Common-Q9 evaluation in capacity-factor units; station-equal aggregation."""

import numpy as np
from .probabilistic import pinball_loss_numpy, crps_from_quantiles_numpy

EVAL_LEVELS = np.arange(1, 10, dtype=np.float64) / 10
METRICS = ("R2", "MAE", "RMSE", "CRPS", "AQL")


def common_q9(prediction, levels):
    levels = np.asarray(levels, dtype=np.float64)
    if prediction.shape[-1] != len(levels):
        raise ValueError("Quantile axis does not match levels")
    indices = []
    for q in EVAL_LEVELS:
        hits = np.flatnonzero(np.isclose(levels, q, atol=1e-7, rtol=0))
        if len(hits) != 1:
            raise ValueError(
                f"Exactly one output for q={q} is required; no silent interpolation"
            )
        indices.append(int(hits[0]))
    return np.asarray(prediction, dtype=np.float64)[..., indices]


def station_metrics(prediction, target, mask, levels):
    q = common_q9(prediction, levels)
    y = np.asarray(target, dtype=np.float64)
    valid = np.asarray(mask) > 0
    if y.shape == q.shape[:-1] + (1,):
        y, valid = y[..., 0], valid[..., 0]
    if y.shape != q.shape[:-1] or valid.shape != y.shape:
        raise ValueError("Target/mask/prediction shape mismatch")
    valid &= np.isfinite(y)
    if not np.isfinite(q[valid]).all():
        raise ValueError("Non-finite predictions on valid targets")
    y, q = y[valid], q[valid]
    if not y.size:
        raise ValueError("No valid targets")
    error = q[:, 4] - y
    sse = np.square(error).sum()
    sst = np.square(y).sum() - y.sum() ** 2 / y.size
    return {
        "R2": float(1 - sse / sst) if sst > 1e-6 else float("nan"),
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(sse / y.size)),
        "CRPS": float(crps_from_quantiles_numpy(q, y, EVAL_LEVELS).mean()),
        "AQL": float(pinball_loss_numpy(q, y, EVAL_LEVELS).mean()),
        "valid_targets": int(y.size),
    }


def station_equal(rows):
    result = {}
    for metric in METRICS:
        values = np.asarray([r[metric] for r in rows], dtype=float)
        finite = values[np.isfinite(values)]
        result[metric] = float(finite.mean()) if len(finite) else float("nan")
        result[metric + "_stations"] = int(len(finite))
    return result
