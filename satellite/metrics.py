"""Masked point metrics in capacity-factor units, with station-equal means."""

import math
import numpy as np
import torch


def masked_mse_components(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Prediction/target/mask shape mismatch")
    valid = mask.bool()
    if not valid.any():
        raise ValueError("Batch has no valid targets")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target[valid]).all():
        raise ValueError("Nonfinite predictions or observed targets")
    residual = prediction.float()[valid] - target.float()[valid]
    return residual.square().sum(), valid.sum()


class PointMetrics:
    def __init__(self):
        self.count = 0
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.target_sum = 0.0
        self.target_square_sum = 0.0

    def update(self, prediction, target, mask):
        prediction = np.asarray(prediction, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        if prediction.shape != target.shape or target.shape != mask.shape:
            raise ValueError("Metric array shape mismatch")
        prediction, target = prediction[mask], target[mask]
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError("Nonfinite valid prediction/target")
        error = prediction - target
        self.count += target.size
        self.absolute_error += float(np.abs(error).sum())
        self.squared_error += float(np.square(error).sum())
        self.target_sum += float(target.sum())
        self.target_square_sum += float(np.square(target).sum())

    def result(self):
        if not self.count:
            return {"mae": None, "rmse": None, "r2": None, "count": 0}
        variance = self.target_square_sum - self.target_sum**2 / self.count
        return {
            "mae": self.absolute_error / self.count,
            "rmse": math.sqrt(self.squared_error / self.count),
            "r2": 1 - self.squared_error / variance if variance > 1e-6 else None,
            "count": int(self.count),
        }


def station_equal(rows):
    result = {
        "station_count": len(rows),
        "target_points": sum(x["count"] for x in rows),
    }
    for key in ("mae", "rmse", "r2"):
        values = [r[key] for r in rows if r[key] is not None and math.isfinite(r[key])]
        result[key] = float(np.mean(values)) if values else None
        result[key + "_stations"] = len(values)
    return result
