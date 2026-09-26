import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_or_identity_scaler(train_values, use_scale=True):
    scaler = StandardScaler()
    if use_scale:
        scaler.fit(train_values)
    else:
        scaler.mean_ = np.zeros(train_values.shape[-1], dtype=np.float64)
        scaler.scale_ = np.ones(train_values.shape[-1], dtype=np.float64)
        scaler.var_ = np.ones(train_values.shape[-1], dtype=np.float64)
        scaler.n_features_in_ = train_values.shape[-1]
    return scaler


def build_time_features(timestamps, resolution):
    dates = pd.to_datetime(timestamps)
    # Every sub-hourly cadence uses the canonical minute-prefixed layout.  The
    # task specs and native-covariate slicer already reserve five channels for
    # 10sec/10min (in addition to 1/5/15/30min), so omitting those two
    # resolutions here would yield a rank-compatible but width-inconsistent
    # tensor and break mixed native-covariate collation.
    resolution = str(resolution or "").strip().lower()
    stamp_df = pd.DataFrame(
        {
            "hour": dates.hour / 23.0 - 0.5,
            "weekday": dates.dayofweek / 6.0 - 0.5,
            "day": (dates.day - 1) / 30.0 - 0.5,
            "dayofyear": (dates.dayofyear - 1) / 365.0 - 0.5,
        }
    )
    if resolution in {"10sec", "1min", "5min", "10min", "15min", "30min"}:
        stamp_df.insert(0, "minute", dates.minute / 59.0 - 0.5)
    return stamp_df.values.astype(np.float32)
