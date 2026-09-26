import torch


def masked_pinball_loss_components(
    pred_quantiles, target, quantiles, mask=None, point_weights=None
):
    """Return summed multi-quantile pinball loss and its effective weight.

    ``pred_quantiles`` is ``[B, T, Q]`` and ``target`` is ``[B, T, 1]`` (or
    broadcastable).  Keeping the sum/denominator contract identical to the
    point losses makes validation and checkpoint selection comparable.
    """
    if pred_quantiles.ndim != 3:
        raise ValueError(
            f"Expected quantile predictions [B,T,Q], got {tuple(pred_quantiles.shape)}"
        )
    q = torch.as_tensor(
        quantiles, dtype=pred_quantiles.dtype, device=pred_quantiles.device
    ).view(1, 1, -1)
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
    return weighted.sum(), (weighted_mask.sum() * pred_quantiles.shape[-1]).clamp_min(
        1.0
    )
