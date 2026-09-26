import torch
import math


def masked_mse_loss_components(pred, target, mask=None):
    loss = (pred - target) ** 2
    if mask is None:
        return loss.sum(), torch.tensor(float(loss.numel()), device=loss.device, dtype=loss.dtype)
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(loss.dtype)
    weighted = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return weighted.sum(), denom


def masked_mse_loss(pred, target, mask=None):
    loss_sum, denom = masked_mse_loss_components(pred, target, mask=mask)
    return loss_sum / denom


def masked_pinball_loss_components(pred_quantiles, target, quantiles, mask=None, point_weights=None):
    """Return summed multi-quantile pinball loss and its effective weight.

    ``pred_quantiles`` is ``[B, T, Q]`` and ``target`` is ``[B, T, 1]`` (or
    broadcastable).  Keeping the sum/denominator contract identical to the
    point losses makes validation and checkpoint selection comparable.
    """
    if pred_quantiles.ndim != 3:
        raise ValueError(f"Expected quantile predictions [B,T,Q], got {tuple(pred_quantiles.shape)}")
    q = torch.as_tensor(quantiles, dtype=pred_quantiles.dtype, device=pred_quantiles.device).view(1, 1, -1)
    if pred_quantiles.shape[-1] != q.shape[-1]:
        raise ValueError("Quantile level count does not match prediction width.")
    # Accept the two target layouts emitted by the point and quantile
    # trainers: ``[B, T]`` and ``[B, T, 1]``.  Slicing a rank-2 target with
    # ``[..., :1]`` would produce ``[B, 1]`` and accidentally broadcast the
    # time axis, silently assigning the wrong target to every horizon.
    target = target.to(device=pred_quantiles.device, dtype=pred_quantiles.dtype)
    if target.ndim == pred_quantiles.ndim - 1:
        target = target.unsqueeze(-1)
    elif target.ndim == pred_quantiles.ndim:
        if target.shape[-1] != 1:
            raise ValueError(
                "Quantile targets must have shape [B,T] or [B,T,1]; "
                f"got {tuple(target.shape)}."
            )
        target = target[..., :1]
    else:
        raise ValueError(
            "Quantile targets must have shape [B,T] or [B,T,1]; "
            f"got {tuple(target.shape)}."
        )
    if target.shape[:-1] != pred_quantiles.shape[:-1]:
        raise ValueError(
            "Quantile target and prediction batch/time dimensions differ: "
            f"target={tuple(target.shape)}, pred={tuple(pred_quantiles.shape)}."
        )
    error = target - pred_quantiles
    loss = torch.maximum(q * error, (q - 1.0) * error)
    if mask is None:
        mask = torch.ones_like(loss[..., :1])
    else:
        mask = mask[..., :1] if mask.ndim == 3 else mask
        while mask.ndim < loss.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.to(loss.dtype)
    if point_weights is None:
        point_weights = torch.ones_like(mask, dtype=loss.dtype, device=loss.device)
    else:
        while point_weights.ndim < loss.ndim:
            point_weights = point_weights.unsqueeze(-1)
        point_weights = point_weights.to(dtype=loss.dtype, device=loss.device)
    weighted_mask = mask * point_weights
    weighted = loss * weighted_mask
    return weighted.sum(), (weighted_mask.sum() * pred_quantiles.shape[-1]).clamp_min(1.0)


def normalize_time_feature_schema(schema):
    """Map dataset-level time-schema names to the geometry helper contract.

    Datasets persist descriptive versioned names (for example
    ``legacy_right_pad_v1`` and ``minute_hour_weekday_day_dayofyear_v1``),
    whereas the loss helper only needs to know whether minute is the first
    channel.  Keeping the mapping here prevents an opt-in regime loss from
    rejecting an otherwise valid existing dataset/checkpoint.
    """
    text = str(schema or "legacy").strip().lower()
    if text in {"legacy", "legacy_right_pad_v1"}:
        return "legacy"
    if text in {"canonical", "minute_hour_weekday_day_dayofyear_v1"}:
        return "canonical"
    if text == "auto":
        return "auto"
    raise ValueError(f"Unsupported time-feature schema: {schema!r}")


def solar_elevation_from_time_features(time_features, site_features, schema="legacy", clamp=True):
    """Return a solar-elevation proxy from normalized time features.

    The original 1h contract is ``[hour, weekday, day, dayofyear]``.  The
    canonical multi-resolution contract prepends minute and is therefore
    ``[minute, hour, weekday, day, dayofyear]``.  Existing callers retain the
    legacy behavior by default; callers that handle mixed-resolution batches
    should use ``schema='auto'``.
    By default the result is clamped at zero, preserving the historical
    daytime-weighting behavior.  Regime classifiers that need to distinguish
    night from twilight can request ``clamp=False`` to retain the signed
    elevation proxy.
    """
    if time_features is None or time_features.shape[-1] < 4:
        raise ValueError("time_features must include at least 4 time dimensions")
    if site_features is None or site_features.shape[-1] < 2:
        raise ValueError("site_features must include lat/lon in the first two dimensions")
    schema = normalize_time_feature_schema(schema)
    use_canonical = schema == "canonical" or (schema == "auto" and time_features.shape[-1] >= 5)
    if use_canonical and time_features.shape[-1] < 5:
        raise ValueError(
            "canonical time_features must use "
            "[minute, hour, weekday, day, dayofyear] (width >= 5); "
            f"got width={time_features.shape[-1]}."
        )
    minute_index = 0 if use_canonical else None
    hour_index = 1 if use_canonical else 0
    # The legacy contract is four channels.  Some mixed-resolution legacy
    # collators right-pad those four channels to width five, however; the
    # padding is at the end, so day-of-year remains channel three.  Using
    # ``-1`` here would silently read the padding as January and corrupt all
    # seasonal/night classification for those batches.
    dayofyear_index = 4 if use_canonical else 3
    hour_norm = time_features[..., hour_index]
    dayofyear_norm = time_features[..., dayofyear_index]
    hour_of_day = (hour_norm + 0.5) * 23.0
    if minute_index is not None:
        # Canonical multiresolution features encode minute-of-hour in channel
        # zero. Include it in the physical clock so phase emphasis does not
        # collapse every sub-hour sample to the same solar position.
        minute_of_hour = (time_features[..., minute_index] + 0.5) * 59.0
        hour_of_day = hour_of_day + minute_of_hour / 60.0
    day_of_year = (dayofyear_norm + 0.5) * 365.0 + 1.0
    lat = site_features[..., 0:1]
    lon = site_features[..., 1:2]
    timezone_offset_hours = (
        site_features[..., 4:5] if site_features.shape[-1] > 4 else torch.zeros_like(lat)
    )
    solar_hour = hour_of_day - timezone_offset_hours.squeeze(-1).unsqueeze(1) + lon.squeeze(-1).unsqueeze(1) / 15.0
    delta = 23.45 * torch.sin(2 * math.pi * (284.0 + day_of_year) / 365.0)
    hour_angle = 15.0 * (solar_hour - 12.0)
    lat_rad = torch.deg2rad(lat)
    delta_rad = torch.deg2rad(delta)
    hour_angle_rad = torch.deg2rad(hour_angle)
    sin_alpha = (
        torch.sin(lat_rad) * torch.sin(delta_rad)
        + torch.cos(lat_rad) * torch.cos(delta_rad) * torch.cos(hour_angle_rad)
    )
    if clamp:
        return torch.clamp(sin_alpha, min=0.0).float()
    return sin_alpha.float()


def daytime_weight_map(
    future_time_features,
    site_features,
    day_weight=1.0,
    night_weight=0.2,
    mode="daytime_weighted_mse",
    schema="legacy",
):
    elevation = solar_elevation_from_time_features(
        future_time_features, site_features, schema=schema
    )
    if mode == "daytime_weighted_mse":
        return torch.where(
            elevation > 0.0,
            elevation.new_full(elevation.shape, float(day_weight)),
            elevation.new_full(elevation.shape, float(night_weight)),
        )
    if mode == "solar_weighted_mse":
        base = elevation.clamp(0.0, 1.0)
        return float(night_weight) + (float(day_weight) - float(night_weight)) * base
    raise ValueError(f"Unsupported loss mode: {mode}")


def regime_point_weight_map(
    future_target,
    future_target_mask,
    future_time_features=None,
    site_features=None,
    resolutions=None,
    ramp_weight=0.0,
    ramp_threshold=0.10,
    twilight_weight=0.0,
    twilight_elevation=0.15,
    peak_weight=0.0,
    peak_threshold=0.80,
    night_weight=0.0,
    max_weight=4.0,
    default_step_hours=1.0,
    target_space="capacity_factor",
    time_feature_schema="auto",
    collect_stats=True,
):
    """Build optional training-only weights for PV difficulty regimes.

    The map is point-wise ``[B, T, 1]`` and starts at one.  A configured
    regime contributes an additive multiplier, so ``ramp_weight=1`` doubles
    the loss on points adjacent to a ramp.  Ramp thresholds are expressed per
    physical hour; callers using standardized targets must explicitly set
    ``target_space='standardized'`` and provide thresholds in that space.

    ``time_feature_schema`` should be supplied by a dataset when phase
    weighting is enabled (``legacy`` for the historical four-channel layout,
    ``canonical`` for the minute-prefixed five-channel layout).  ``auto`` is
    retained for standalone callers.

    This function deliberately uses the observed future target only as a
    *training supervision signal*.  The trainer gates it on ``model.training``
    so validation/checkpoint selection remains the ordinary unweighted loss.
    """
    if future_target is None or future_target.ndim < 2:
        return None, {}
    strengths = tuple(
        float(value) for value in (ramp_weight, twilight_weight, peak_weight, night_weight)
    )
    # Validate before checking whether weighting is enabled: NaN would make
    # ``value > 0`` false and silently disable a malformed configuration,
    # while +inf would otherwise be clipped to ``max_weight`` and hide the
    # typo.  Direct callers need the same fail-closed behavior as the CLI and
    # FoundationTrainer constructor.
    if any(not math.isfinite(value) or value < 0.0 for value in strengths):
        raise ValueError("regime weights must be finite and non-negative.")
    enabled = any(value > 0.0 for value in strengths)
    if not enabled:
        return None, {}
    if not math.isfinite(float(max_weight)) or float(max_weight) < 1.0:
        raise ValueError("regime max_weight must be >= 1.")
    if not math.isfinite(float(default_step_hours)) or float(default_step_hours) <= 0.0:
        raise ValueError("regime default_step_hours must be positive.")
    if any(
        not math.isfinite(float(value)) or float(value) < 0.0
        for value in (ramp_threshold, twilight_elevation, peak_threshold)
    ):
        raise ValueError("regime thresholds must be non-negative.")
    target_space = str(target_space or "capacity_factor").strip().lower()
    if target_space not in {"capacity_factor", "standardized"}:
        raise ValueError("regime target_space must be capacity_factor|standardized.")

    target = future_target
    if target.ndim == 2:
        target = target.unsqueeze(-1)
    elif target.ndim >= 3:
        target = target[..., :1]
    else:  # defensive; the rank check above should make this unreachable
        return None, {}
    target = target.float()
    if future_target_mask is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    else:
        # Collators normally return ``[B,T,1]`` masks, but a few lightweight
        # callers use ``[B,T]``.  Slicing a 2-D mask with ``[..., :1]`` would
        # silently reduce it to ``[B,1]`` and make the last T-1 points appear
        # invalid, so normalize the rank explicitly.
        mask = future_target_mask
        if mask.ndim == target.ndim - 1:
            mask = mask.unsqueeze(-1)
        elif mask.ndim == target.ndim:
            mask = mask[..., :1]
        else:
            while mask.ndim < target.ndim:
                mask = mask.unsqueeze(-1)
            mask = mask[..., :1]
        valid = mask > 0
        while valid.ndim < target.ndim:
            valid = valid.unsqueeze(-1)
    weights = torch.ones_like(target)
    stats = {}

    # A dynamic C/H batch is homogeneous by resolution.  Infer the physical
    # interval from the batch metadata when available, with a safe 1h default.
    batch_size = int(target.shape[0])
    step_hours = target.new_full((batch_size, 1, 1), float(default_step_hours))
    # Keep this in sync with ``foundation.task_specs.pv_tasks``. Native
    # dynamic-resolution batches can be 10sec/1min/5min/10min/15min/30min or
    # 1h; falling back to one hour for a known cadence would distort ramp
    # thresholds by up to 360x.
    resolution_map = {
        "1h": 1.0,
        "60min": 1.0,
        "30min": 0.5,
        "15min": 0.25,
        "10min": 1.0 / 6.0,
        "5min": 1.0 / 12.0,
        "1min": 1.0 / 60.0,
        "10sec": 1.0 / 360.0,
    }
    if resolutions is not None:
        try:
            resolution_values = (
                [resolutions] * batch_size
                if isinstance(resolutions, str)
                else list(resolutions)[:batch_size]
            )
            for row, resolution in enumerate(resolution_values):
                resolution_text = str(resolution).strip().lower()
                if resolution_text in resolution_map:
                    step_hours[row, 0, 0] = resolution_map[resolution_text]
        except TypeError:
            pass

    valid_2d = valid[..., 0]
    if float(ramp_weight) > 0.0 and target.shape[1] > 1:
        delta = (target[:, 1:, 0] - target[:, :-1, 0]).abs()
        delta_per_hour = delta / step_hours[:, 0, 0].unsqueeze(-1).clamp_min(1e-6)
        pair_valid = valid_2d[:, 1:] & valid_2d[:, :-1]
        ramp_pair = pair_valid & (delta_per_hour >= float(ramp_threshold))
        ramp = torch.zeros_like(valid_2d)
        ramp[:, 1:] |= ramp_pair
        ramp[:, :-1] |= ramp_pair
        ramp = ramp.unsqueeze(-1)
        weights = weights + float(ramp_weight) * ramp.to(weights.dtype)
        if collect_stats:
            stats["ramp_point_fraction"] = float(
                (ramp & valid).sum().detach().cpu().item()
                / valid.sum().clamp_min(1).detach().cpu().item()
            )
    elif collect_stats:
        stats["ramp_point_fraction"] = 0.0

    daylight = twilight = night = None
    phase_requested = float(twilight_weight) > 0.0 or float(night_weight) > 0.0
    if phase_requested and (future_time_features is None or site_features is None):
        raise ValueError(
            "Solar-phase regime weighting was enabled, but future_time_features "
            "and site_features were not supplied."
        )
    if future_time_features is not None and site_features is not None:
        try:
            elevation = solar_elevation_from_time_features(
                future_time_features,
                site_features,
                schema=time_feature_schema,
                clamp=False,
            ).unsqueeze(-1)
            geometry_valid = torch.isfinite(elevation)
            # ``site_features[..., 2]`` is the collator's geo-reliability
            # flag. Missing coordinates are filled with zeros so that models
            # can still run; treating those finite zeros as a real equatorial
            # site would incorrectly label points as day/night and apply a
            # phase weight.  Preserve compatibility with lightweight callers
            # that provide only [lat, lon] by requiring this check only when
            # the reliability channel is present.
            if site_features.shape[-1] > 2:
                reliability = site_features[..., 2:3]
                while reliability.ndim < elevation.ndim:
                    reliability = reliability.unsqueeze(1)
                geometry_valid = geometry_valid & torch.isfinite(reliability) & (reliability > 0.0)
            # Sunrise/sunset is a band around the geometric horizon, not only
            # the low-positive (post-sunrise) half.  Keeping twilight and
            # deep night disjoint makes a combined twilight+night experiment
            # interpretable and gives pre-dawn and post-sunset ramps the same
            # treatment. ``daylight`` remains the positive-elevation mask
            # used by the optional high-output/peak regime.
            phase_threshold = float(twilight_elevation)
            daylight = geometry_valid & (elevation > 0.0)
            twilight = geometry_valid & (elevation.abs() <= phase_threshold)
            night = geometry_valid & (elevation < -phase_threshold)
        except (TypeError, ValueError, RuntimeError) as exc:
            # If a caller explicitly requests a solar-phase weight, silently
            # dropping it because a collator supplied an incompatible time or
            # site tensor would make the experiment irreproducible.  Ramp or
            # peak-only weighting can still operate without geometry.
            if phase_requested:
                raise ValueError(
                    "Solar-phase regime weighting was enabled, but solar "
                    "geometry could not be computed from future_time_features "
                    "and site_features. Check the time_feature_schema and "
                    "site feature dimensions."
                ) from exc
            daylight = twilight = night = None

    if float(twilight_weight) > 0.0 and twilight is not None:
        weights = weights + float(twilight_weight) * twilight.to(weights.dtype)
        if collect_stats:
            stats["twilight_point_fraction"] = float(
                (twilight & valid).sum().detach().cpu().item()
                / valid.sum().clamp_min(1).detach().cpu().item()
            )
    elif collect_stats:
        stats["twilight_point_fraction"] = 0.0

    if float(peak_weight) > 0.0:
        peak = target >= float(peak_threshold)
        if daylight is not None:
            peak = peak & daylight
        weights = weights + float(peak_weight) * peak.to(weights.dtype)
        if collect_stats:
            stats["peak_point_fraction"] = float(
                (peak & valid).sum().detach().cpu().item()
                / valid.sum().clamp_min(1).detach().cpu().item()
            )
    elif collect_stats:
        stats["peak_point_fraction"] = 0.0

    if float(night_weight) > 0.0 and night is not None:
        weights = weights + float(night_weight) * night.to(weights.dtype)
        if collect_stats:
            stats["night_point_fraction"] = float(
                (night & valid).sum().detach().cpu().item()
                / valid.sum().clamp_min(1).detach().cpu().item()
            )
    elif collect_stats:
        stats["night_point_fraction"] = 0.0

    weights = weights.clamp(max=float(max_weight))
    # Invalid/masked target points never contribute to the loss. Keep their
    # multiplier at one so a later caller cannot accidentally reuse a regime
    # map with a different mask and over-weight those points.
    weights = torch.where(valid, weights, torch.ones_like(weights))
    if collect_stats:
        stats["regime_weight_mean"] = (
            float(weights[valid].mean().detach().cpu().item())
            if bool(valid.any())
            else 1.0
        )
        stats["regime_weight_max"] = float(weights.max().detach().cpu().item())
        stats["regime_target_space"] = target_space
    return weights, stats


def masked_weighted_mse_loss_components(pred, target, mask=None, point_weights=None):
    loss = (pred - target) ** 2
    if mask is None:
        mask = torch.ones_like(loss[..., 0] if loss.ndim > 2 else loss, dtype=loss.dtype, device=loss.device)
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(loss.dtype)
    if point_weights is None:
        point_weights = torch.ones_like(mask, dtype=loss.dtype, device=loss.device)
    while point_weights.ndim < loss.ndim:
        point_weights = point_weights.unsqueeze(-1)
    point_weights = point_weights.to(loss.dtype)
    weighted_mask = mask * point_weights
    weighted = loss * weighted_mask
    denom = weighted_mask.sum().clamp_min(1.0)
    return weighted.sum(), denom


def masked_diff_mse_loss_components(pred, target, mask=None, point_weights=None):
    if pred.shape[1] <= 1 or target.shape[1] <= 1:
        zero = pred.sum() * 0.0
        return zero, zero + 1.0

    pred_diff = pred[:, 1:, ...] - pred[:, :-1, ...]
    target_diff = target[:, 1:, ...] - target[:, :-1, ...]

    diff_mask = None
    if mask is not None:
        while mask.ndim < pred.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.to(pred.dtype)
        diff_mask = mask[:, 1:, ...] * mask[:, :-1, ...]

    diff_weights = None
    if point_weights is not None:
        while point_weights.ndim < pred.ndim:
            point_weights = point_weights.unsqueeze(-1)
        point_weights = point_weights.to(pred.dtype)
        diff_weights = 0.5 * (point_weights[:, 1:, ...] + point_weights[:, :-1, ...])

    return masked_weighted_mse_loss_components(
        pred_diff,
        target_diff,
        mask=diff_mask,
        point_weights=diff_weights,
    )


def masked_peak_mse_loss_components(pred, target, mask=None):
    """MSE between valid-window peak amplitudes for each sample/output channel."""
    if pred.shape[1] <= 0 or target.shape[1] <= 0:
        zero = pred.sum() * 0.0
        return zero, zero + 1.0

    if mask is None:
        peak_pred = pred.max(dim=1).values
        peak_target = target.max(dim=1).values
        loss = (peak_pred - peak_target) ** 2
        return loss.sum(), torch.tensor(float(loss.numel()), device=loss.device, dtype=loss.dtype)

    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=torch.bool, device=pred.device)
    valid_series = mask.any(dim=1)
    neg_inf = torch.finfo(pred.dtype).min
    peak_pred = pred.masked_fill(~mask, neg_inf).max(dim=1).values
    peak_target = target.masked_fill(~mask, neg_inf).max(dim=1).values
    loss = (peak_pred - peak_target) ** 2
    loss = torch.where(valid_series, loss, torch.zeros_like(loss))
    denom = valid_series.to(loss.dtype).sum().clamp_min(1.0)
    return loss.sum(), denom


def masked_negative_mse_loss_components(pred, mask=None):
    loss = torch.relu(-pred) ** 2
    if mask is None:
        return loss.sum(), torch.tensor(float(loss.numel()), device=loss.device, dtype=loss.dtype)
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(loss.dtype)
    weighted = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return weighted.sum(), denom
