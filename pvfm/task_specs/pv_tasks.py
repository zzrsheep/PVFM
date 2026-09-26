from __future__ import annotations

from dataclasses import dataclass

PV_STEPS_PER_HOUR = {
    "10sec": 360,
    "1min": 60,
    "5min": 12,
    "10min": 6,
    "15min": 4,
    "30min": 2,
    "1h": 1,
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_family: str
    resolution: str
    horizon_hours: float
    context_ratio: float
    max_context_hours: float | None = None
    history_choices_hours: tuple = ()
    compatible_granularities: tuple = ()
    target_col: str = "target"
    modalities: tuple = ("target", "time", "historical_covariates", "future_covariates")
    default_history_covariate_source: str = "era5"
    default_future_covariate_source: str = "nwp"
    business_tags: tuple = ("pv",)
    loss_weight_scheme: str = "uniform"

    @property
    def context_hours(self):
        if self.max_context_hours is not None:
            return float(self.max_context_hours)
        return float(self.horizon_hours) * float(self.context_ratio)

    @property
    def prediction_length(self):
        return int(float(self.horizon_hours) * PV_STEPS_PER_HOUR[self.resolution])

    @property
    def context_length(self):
        return int(float(self.context_hours) * PV_STEPS_PER_HOUR[self.resolution])


PV_TASK_SPECS = {
    "pv_15min_1h_ahead": TaskSpec(
        name="pv_15min_1h_ahead",
        task_family="pv_forecast",
        resolution="15min",
        horizon_hours=1,
        context_ratio=3.0,
        max_context_hours=3.0,
        history_choices_hours=(1.0, 2.0, 3.0),
        compatible_granularities=("15min",),
        business_tags=("pv", "forecast", "15min", "1h_ahead"),
    ),
    "pv_15min_3h_ahead": TaskSpec(
        name="pv_15min_3h_ahead",
        task_family="pv_forecast",
        resolution="15min",
        horizon_hours=3,
        context_ratio=2.0,
        max_context_hours=6.0,
        history_choices_hours=(3.0, 4.5, 6.0),
        compatible_granularities=("15min",),
        business_tags=("pv", "forecast", "15min", "3h_ahead"),
    ),
    "pv_15min_6h_ahead": TaskSpec(
        name="pv_15min_6h_ahead",
        task_family="pv_forecast",
        resolution="15min",
        horizon_hours=6,
        context_ratio=2.0,
        max_context_hours=12.0,
        history_choices_hours=(6.0, 9.0, 12.0),
        compatible_granularities=("15min",),
        business_tags=("pv", "forecast", "15min", "6h_ahead"),
    ),
    "pv_15min_24h_ahead": TaskSpec(
        name="pv_15min_24h_ahead",
        task_family="pv_forecast",
        resolution="15min",
        horizon_hours=24,
        context_ratio=2.0,
        max_context_hours=48.0,
        history_choices_hours=(24, 36, 48),
        compatible_granularities=("15min",),
        business_tags=("pv", "forecast", "15min", "24h_ahead"),
    ),
    "pv_1h_ahead": TaskSpec(
        name="pv_1h_ahead",
        task_family="pv_forecast",
        resolution="1h",
        horizon_hours=1,
        context_ratio=72.0,
        max_context_hours=72.0,
        history_choices_hours=(8, 16, 24),
        compatible_granularities=(
            "1h",
            "30min",
            "15min",
            "10min",
            "5min",
            "1min",
            "10sec",
        ),
        business_tags=("pv", "forecast", "1h_ahead"),
    ),
    "pv_4h_ahead": TaskSpec(
        name="pv_4h_ahead",
        task_family="pv_forecast",
        resolution="1h",
        horizon_hours=4,
        context_ratio=18.0,
        max_context_hours=72.0,
        history_choices_hours=(16, 24, 48),
        compatible_granularities=(
            "1h",
            "30min",
            "15min",
            "10min",
            "5min",
            "1min",
            "10sec",
        ),
        business_tags=("pv", "forecast", "4h_ahead"),
    ),
    "pv_6h_ahead": TaskSpec(
        name="pv_6h_ahead",
        task_family="pv_forecast",
        resolution="1h",
        horizon_hours=6,
        context_ratio=4.0,
        max_context_hours=24.0,
        history_choices_hours=(12, 18, 24),
        compatible_granularities=(
            "1h",
            "30min",
            "15min",
            "10min",
            "5min",
            "1min",
            "10sec",
        ),
        business_tags=("pv", "forecast", "6h_ahead"),
    ),
    "pv_24h_ahead": TaskSpec(
        name="pv_24h_ahead",
        task_family="pv_forecast",
        resolution="1h",
        horizon_hours=24,
        context_ratio=3.0,
        max_context_hours=72.0,
        history_choices_hours=(24, 48, 72),
        compatible_granularities=(
            "1h",
            "30min",
            "15min",
            "10min",
            "5min",
            "1min",
            "10sec",
        ),
        business_tags=("pv", "forecast", "24h_ahead"),
    ),
    "pv_48h_ahead": TaskSpec(
        name="pv_48h_ahead",
        task_family="pv_forecast",
        resolution="1h",
        horizon_hours=48,
        context_ratio=3.0,
        max_context_hours=144.0,
        history_choices_hours=(48, 96, 144),
        compatible_granularities=(
            "1h",
            "30min",
            "15min",
            "10min",
            "5min",
            "1min",
            "10sec",
        ),
        business_tags=("pv", "forecast", "48h_ahead", "two_day_ahead"),
    ),
    "pv_72h_ahead": TaskSpec(
        name="pv_72h_ahead",
        task_family="pv_forecast",
        resolution="1h",
        horizon_hours=72,
        context_ratio=2.0,
        max_context_hours=144.0,
        history_choices_hours=(72, 96, 144),
        compatible_granularities=(
            "1h",
            "30min",
            "15min",
            "10min",
            "5min",
            "1min",
            "10sec",
        ),
        business_tags=("pv", "forecast", "72h_ahead"),
    ),
    "pv_168h_ahead": TaskSpec(
        name="pv_168h_ahead",
        task_family="pv_forecast",
        resolution="1h",
        horizon_hours=168,
        context_ratio=2.0,
        max_context_hours=336.0,
        history_choices_hours=(168, 240, 336),
        compatible_granularities=(
            "1h",
            "30min",
            "15min",
            "10min",
            "5min",
            "1min",
            "10sec",
        ),
        business_tags=("pv", "forecast", "168h_ahead", "one_week_ahead"),
    ),
}


def resolve_task_spec(task_name):
    return PV_TASK_SPECS.get(task_name)


def resolve_task_lengths(task_name_or_spec):
    if isinstance(task_name_or_spec, TaskSpec):
        spec = task_name_or_spec
    else:
        spec = resolve_task_spec(task_name_or_spec)
    if spec is None:
        raise ValueError(f"Unknown task spec: {task_name_or_spec}")

    steps_per_hour = PV_STEPS_PER_HOUR[spec.resolution]
    pred_len = int(float(spec.horizon_hours) * steps_per_hour)
    seq_len = int(float(spec.context_hours) * steps_per_hour)
    label_len = max(1, pred_len)
    return seq_len, label_len, pred_len
