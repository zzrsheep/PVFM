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
