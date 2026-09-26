"""Streaming metrics for multi-quantile forecasts.

The training objective uses the usual pinball loss.  This module keeps the
evaluation definition in one place so that a checkpoint's reported AQL is
the same quantity as its masked training loss (apart from aggregation over
the evaluated windows).

CRPS is approximated from the reported quantiles using

    CRPS(y, F) = 2 * integral_0^1 rho_q(y - Q(q)) dq.

The finite quantile grid is linearly integrated with the trapezoid rule.  We
hold the first and last reported quantile constant in the two tails, and
monotonically rearrange the reported values before CRPS integration.  The
latter makes the numerical CDF approximation valid even when a model has a
small quantile crossing.
"""

from __future__ import annotations

import math

import numpy as np


def _align_quantile_target_shape(prediction, truth):
    """Accept the common scalar-target ``[..., 1]`` representation.

    Quantile heads return ``[..., Q]`` while collators often preserve the
    scalar target channel as ``[..., 1]``.  Those arrays have the same rank,
    but their final axes have different meanings.  Normalize this one
    unambiguous case before applying strict shape checks; multi-output
    targets are still rejected rather than guessed.
    """

    if (
        prediction.ndim == truth.ndim
        and truth.ndim >= 1
        and truth.shape[-1] == 1
        and prediction.shape[:-1] == truth.shape[:-1]
    ):
        truth = truth[..., 0]
    return truth


def _validate_levels(quantile_levels):
    levels = tuple(float(value) for value in quantile_levels)
    if len(levels) < 2:
        raise ValueError("At least two quantile levels are required.")
    if any(not 0.0 < value < 1.0 for value in levels):
        raise ValueError(f"Quantile levels must lie strictly between 0 and 1: {levels}")
    if tuple(sorted(levels)) != levels or len(set(levels)) != len(levels):
        raise ValueError(f"Quantile levels must be strictly increasing: {levels}")
    return levels


def pinball_loss_numpy(pred_quantiles, target, quantile_levels):
    """Return pinball loss per target point, averaged over quantiles.

    ``pred_quantiles`` has shape ``[..., Q]`` and ``target`` has the same
    leading shape without the final quantile axis.  NaN handling is left to
    the caller so masks can be applied without silently changing counts.
    """

    levels = np.asarray(_validate_levels(quantile_levels), dtype=np.float64)
    prediction = np.asarray(pred_quantiles, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    truth = _align_quantile_target_shape(prediction, truth)
    if prediction.shape[:-1] != truth.shape:
        raise ValueError(
            f"Expected pred shape [..., Q] and target shape [...], got "
            f"{prediction.shape} and {truth.shape}."
        )
    error = truth[..., None] - prediction
    losses = np.maximum(levels * error, (levels - 1.0) * error)
    return np.mean(losses, axis=-1)


def crps_from_quantiles_numpy(pred_quantiles, target, quantile_levels):
    """Approximate CRPS per target point from a quantile function.

    Quantile crossings are monotonically rearranged before integration.  The
    returned value is in the same units squared-root as the target (the
    standard CRPS units), while ``pinball_loss_numpy`` returns the average
    pinball loss over the supplied levels.
    """

    levels = np.asarray(_validate_levels(quantile_levels), dtype=np.float64)
    prediction = np.asarray(pred_quantiles, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    truth = _align_quantile_target_shape(prediction, truth)
    if prediction.shape[:-1] != truth.shape:
        raise ValueError(
            f"Expected pred shape [..., Q] and target shape [...], got "
            f"{prediction.shape} and {truth.shape}."
        )

    # A quantile crossing does not define a valid quantile function.  The
    # increasing rearrangement is the least surprising projection for CRPS.
    monotone_prediction = np.maximum.accumulate(prediction, axis=-1)
    integration_levels = np.concatenate(([0.0], levels, [1.0]))
    tail_low = monotone_prediction[..., :1]
    tail_high = monotone_prediction[..., -1:]
    integration_prediction = np.concatenate(
        (tail_low, monotone_prediction, tail_high), axis=-1
    )
    error = truth[..., None] - integration_prediction
    losses = np.maximum(
        integration_levels * error,
        (integration_levels - 1.0) * error,
    )
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return 2.0 * trapezoid(losses, integration_levels, axis=-1)


class ProbabilisticMetricAccumulator:
    """Accumulate probabilistic metrics without retaining all windows.

    In addition to AQL/CRPS/crossing, the accumulator reports empirical
    coverage and mean width for the standard P10--P90 and P05--P95 intervals
    whenever those levels are present.  Coverage is computed from the
    reported endpoints (using their ordered min/max if a crossing occurs),
    while the crossing rates remain separate diagnostics.  The historical
    ``quantile_crossing_rate`` is the sample-level any-pair rate; the
    pairwise rate is also exported under the short
    ``pairwise_crossing_rate`` alias.
    """

    def __init__(self, quantile_levels):
        self.quantile_levels = _validate_levels(quantile_levels)
        self.quantile_count = len(self.quantile_levels)
        self.count = 0
        self.pinball_sum = 0.0
        self.crps_sum = 0.0
        # Keep the historical ``quantile_crossing_rate`` definition (the
        # fraction of target points with at least one adjacent crossing),
        # while also tracking the denominator needed for the pairwise rate.
        # A row with Q quantiles contributes Q-1 adjacent comparisons.
        self.crossing_count = 0  # legacy alias for any-crossing rows
        self.crossing_any_count = 0
        self.crossing_pair_count = 0
        self.crossing_pair_total = 0
        self.interval_stats = {}
        for low, high, label in (
            (0.10, 0.90, "p10_p90"),
            (0.05, 0.95, "p05_p95"),
        ):
            # Check numerically rather than relying on exact float equality;
            # checkpoints serialized through JSON/configs can carry a tiny
            # representation error (e.g. 0.100000001).
            low_index = next(
                (idx for idx, value in enumerate(self.quantile_levels)
                 if np.isclose(value, low, atol=1e-6, rtol=0.0)),
                None,
            )
            high_index = next(
                (idx for idx, value in enumerate(self.quantile_levels)
                 if np.isclose(value, high, atol=1e-6, rtol=0.0)),
                None,
            )
            if low_index is not None and high_index is not None:
                self.interval_stats[label] = {
                    "low_index": low_index,
                    "high_index": high_index,
                    "inside_count": 0,
                    "width_sum": 0.0,
                }

    def update(self, pred_quantiles, target, mask=None):
        prediction = np.asarray(pred_quantiles, dtype=np.float64)
        truth = np.asarray(target, dtype=np.float64)
        truth = _align_quantile_target_shape(prediction, truth)
        if prediction.shape[-1] != self.quantile_count:
            raise ValueError(
                f"Expected {self.quantile_count} quantiles, got shape {prediction.shape}."
            )
        if prediction.ndim != truth.ndim + 1:
            raise ValueError(
                f"Expected prediction ndim={truth.ndim + 1}, got {prediction.ndim}."
            )
        prediction = prediction.reshape(-1, self.quantile_count)
        truth = truth.reshape(-1)
        finite = np.isfinite(truth) & np.all(np.isfinite(prediction), axis=-1)
        if mask is not None:
            finite &= np.asarray(mask).reshape(-1).astype(bool)
        if not np.any(finite):
            return
        prediction = prediction[finite]
        truth = truth[finite]
        pinball = pinball_loss_numpy(prediction, truth, self.quantile_levels)
        crps = crps_from_quantiles_numpy(prediction, truth, self.quantile_levels)
        self.count += int(truth.size)
        self.pinball_sum += float(np.sum(pinball))
        self.crps_sum += float(np.sum(crps))
        pair_crossings = prediction[:, :-1] > prediction[:, 1:]
        any_crossings = np.any(pair_crossings, axis=-1)
        any_count = int(any_crossings.sum())
        pair_count = int(pair_crossings.sum())
        self.crossing_count += any_count
        self.crossing_any_count += any_count
        self.crossing_pair_count += pair_count
        self.crossing_pair_total += int(prediction.shape[0] * max(self.quantile_count - 1, 0))
        for state in self.interval_stats.values():
            lower = prediction[:, state["low_index"]]
            upper = prediction[:, state["high_index"]]
            ordered_low = np.minimum(lower, upper)
            ordered_high = np.maximum(lower, upper)
            state["inside_count"] += int(np.count_nonzero((truth >= ordered_low) & (truth <= ordered_high)))
            state["width_sum"] += float(np.sum(ordered_high - ordered_low))

    def metrics(self):
        if self.count <= 0:
            result = {
                "aql": math.nan,
                "crps": math.nan,
                "quantile_crossing_rate": math.nan,
                "quantile_crossing_rate_any": math.nan,
                "quantile_crossing_rate_pairwise": math.nan,
                "pairwise_crossing_rate": math.nan,
                "quantile_count": 0,
            }
            for label in self.interval_stats:
                result[f"{label}_coverage"] = math.nan
                result[f"{label}_mean_width"] = math.nan
            return result
        result = {
            "aql": float(self.pinball_sum / self.count),
            "crps": float(self.crps_sum / self.count),
            # ``quantile_crossing_rate`` remains the backwards-compatible
            # any-crossing metric.  The explicit names remove ambiguity in
            # new reports and allow both definitions to be compared.
            "quantile_crossing_rate": float(self.crossing_any_count / self.count),
            "quantile_crossing_rate_any": float(self.crossing_any_count / self.count),
            "quantile_crossing_rate_pairwise": (
                float(self.crossing_pair_count / self.crossing_pair_total)
                if self.crossing_pair_total > 0
                else math.nan
            ),
            # Short alias used by the post-hoc regime report.
            "pairwise_crossing_rate": (
                float(self.crossing_pair_count / self.crossing_pair_total)
                if self.crossing_pair_total > 0
                else math.nan
            ),
            "quantile_count": int(self.count),
        }
        for label, state in self.interval_stats.items():
            result[f"{label}_coverage"] = float(state["inside_count"] / self.count)
            result[f"{label}_mean_width"] = float(state["width_sum"] / self.count)
        return result


def station_equal_probabilistic_metrics(accumulators):
    """Return arithmetic station-equal averages of probabilistic metrics.

    ``accumulators`` may be an iterable of ``ProbabilisticMetricAccumulator``
    objects or payload dictionaries containing an ``"acc"`` entry (the
    representation used by the evaluation scripts).  Each station first
    contributes one metric row, regardless of how many valid windows it has;
    empty stations are excluded from the averages and reported separately.

    ``quantile_count`` is a valid-point count rather than a metric, so the
    station-equal row reports its sum and adds explicit station counts.  All
    metric fields are averaged over eligible stations using finite values.
    """

    if isinstance(accumulators, dict):
        # Evaluation code normally passes ``mapping.values()``; accepting the
        # mapping itself makes the helper less error-prone for callers.
        accumulators = accumulators.values()

    rows = []
    total_count = 0
    total_stations = 0
    for item in accumulators:
        total_stations += 1
        accumulator = item.get("acc") if isinstance(item, dict) else item
        if accumulator is None:
            continue
        metrics = accumulator.metrics()
        count = int(metrics.get("quantile_count", 0) or 0)
        total_count += count
        if count > 0:
            rows.append(metrics)

    result = {
        "station_count": int(len(rows)),
        "station_count_total": int(total_stations),
        "station_valid_points": int(total_count),
        "quantile_count": int(total_count),
    }
    metric_names = (
        "aql",
        "crps",
        "quantile_crossing_rate",
        "quantile_crossing_rate_any",
        "quantile_crossing_rate_pairwise",
        "pairwise_crossing_rate",
        "p10_p90_coverage",
        "p10_p90_mean_width",
        "p05_p95_coverage",
        "p05_p95_mean_width",
    )
    for name in metric_names:
        values = [float(row[name]) for row in rows if np.isfinite(row.get(name, math.nan))]
        result[name] = float(np.mean(values)) if values else math.nan
    return result
