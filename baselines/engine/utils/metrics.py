import numpy as np


def _flatten_valid_pairs(pred, true):
    pred_arr = np.asarray(pred, dtype=np.float64)
    true_arr = np.asarray(true, dtype=np.float64)
    valid = np.isfinite(pred_arr) & np.isfinite(true_arr)
    return pred_arr[valid], true_arr[valid]


def RSE(pred, true):
    pred, true = _flatten_valid_pairs(pred, true)
    if true.size == 0:
        return np.nan
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    pred, true = _flatten_valid_pairs(pred, true)
    if true.size == 0:
        return np.nan
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    pred, true = _flatten_valid_pairs(pred, true)
    if true.size == 0:
        return np.nan
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    pred, true = _flatten_valid_pairs(pred, true)
    if true.size == 0:
        return np.nan
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    pred, true = _flatten_valid_pairs(pred, true)
    if true.size == 0:
        return np.nan
    denom = np.where(np.abs(true) < 1e-6, np.nan, true)
    value = np.abs((true - pred) / denom)
    return np.nanmean(value)


def MSPE(pred, true):
    pred, true = _flatten_valid_pairs(pred, true)
    if true.size == 0:
        return np.nan
    denom = np.where(np.abs(true) < 1e-6, np.nan, true)
    value = np.square((true - pred) / denom)
    return np.nanmean(value)


def SMAPE(pred, true):
    pred, true = _flatten_valid_pairs(pred, true)
    if true.size == 0:
        return np.nan
    denom = (np.abs(pred) + np.abs(true)) / 2.0
    denom = np.where(denom < 1e-6, np.nan, denom)
    value = np.abs(pred - true) / denom
    return np.nanmean(value)


def R2(pred, true):
    pred, true = _flatten_valid_pairs(pred, true)
    if true.size == 0:
        return np.nan
    true_mean = np.mean(true)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true_mean) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    smape = SMAPE(pred, true)
    r2 = R2(pred, true)

    return mae, mse, rmse, mape, mspe, smape, r2
