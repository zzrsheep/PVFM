"""Production temporal-pool loader, using extracted original implementations.

Only external, explicitly training-role data may be supplied here. The bundled
Australian unseen cohort is not a PVFM pretraining dataset. Target-site
full-shot baseline training is a separate workflow.
"""

import json
from pathlib import Path
import random

from torch.utils.data import DataLoader

from .data.multires_cache import MultiResolutionDataset
from .data.temporal_pool_v1 import (
    TemporalPoolDataset,
    TemporalPoolSampler,
    load_temporal_pool,
    validate_temporal_pool,
    build_temporal_pool,
)
from .data.sampler import AverageSupplyTemporalSampler
from .data.temporal_average_supply import build_or_load_average_supply

ROOT = Path(__file__).resolve().parents[1]


def policy():
    return json.loads((ROOT / "configs/training_80k.json").read_text())


def sampler_kwargs(cfg, batch_size, rank, world_size):
    return dict(
        batch_size=batch_size,
        seed=cfg["seed"],
        rank=rank,
        world_size=world_size,
        batches_per_epoch=cfg["train_steps"],
        source_weights={"1h": 0.75, "15min": 0.25},
        region_balance_alpha=0.25,
        region_balance_max_prob=0.15,
        context_max_hours=672,
        context_min_ratio=0.5,
        context_max_ratio=4.0,
        typical_probability=0.6,
        typical_weights=(0.333333, 0.333333, 0.333334),
        native_typical_probability=0.6,
        native_typical_horizons=(1.0, 4.0, 6.0, 24.0),
        native_typical_weights=(0.25,) * 4,
        native_horizon_min_hours=0.25,
        native_horizon_max_hours=24.0,
        native_context_min_ratio=0.5,
        native_context_max_ratio=4.0,
        native_context_max_hours=96.0,
        dynamic_native_task_names=("pv_15min_24h_ahead",),
        max_retries=64,
        enable_resume_state=True,
    )


def read_data_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg.get("role") != "pretrain_train":
        raise ValueError(
            "Training requires role=pretrain_train, not bundled unseen_test data"
        )
    for key in [
        "binary_cache_registry",
        "manifest_path",
        "pool_dir",
        "quality_dir",
        "region_profile",
    ]:
        if key not in cfg or not Path(cfg[key]).is_absolute():
            raise ValueError(
                f"{key} must be an explicit absolute path in your data config"
            )
    # Prevent an accidental direct reuse of any bundled evaluation station.
    import pandas as pd

    excluded = set(json.loads((ROOT / "baselines/australia_stations.json").read_text()))
    stations = set(pd.read_csv(cfg["manifest_path"]).station_dir)
    if stations & excluded:
        raise ValueError(
            "The PVFM pretraining manifest overlaps the Australian unseen cohort"
        )
    return cfg


def base_dataset(cfg):
    return MultiResolutionDataset(
        binary_cache_registry=cfg["binary_cache_registry"],
        binary_index_overlay_dir="",
        manifest_path=cfg["manifest_path"],
        task_names=[
            "pv_6h_ahead",
            "pv_24h_ahead",
            "pv_168h_ahead",
            "pv_15min_24h_ahead",
        ],
        split="train",
        fixed_seq_len=672,
        batch_size=256,
        shuffle=True,
        metadata_mode="minimal",
        time_feature_schema="minute_hour_weekday_day_dayofyear_v1",
        region_balance_alpha=0.25,
        region_balance_max_prob=0.15,
        region_balance_max_repeat_per_epoch=20.0,
        region_loss_alpha=0.0,
        min_history_covariate_valid_ratio=1.0,
        min_future_covariate_valid_ratio=1.0,
    )


def load_training(path, batch_size, rank, world_size, workers=0):
    cfg = read_data_config(path)
    for directory in ["pool_dir", "quality_dir"]:
        if not (Path(cfg[directory]) / "metadata.json").is_file():
            raise ValueError(
                f"Prebuild {directory} in one CPU process; training never silently rebuilds it"
            )
    base = base_dataset(cfg)
    pool = load_temporal_pool(cfg["pool_dir"])
    validate_temporal_pool(pool, base)
    ds = TemporalPoolDataset(
        base, pool, 672, 168, dynamic_native_task_names=("pv_15min_24h_ahead",)
    )
    ds.ensure_quality_index(cfg["quality_dir"])
    sampler = AverageSupplyTemporalSampler(
        ds,
        **sampler_kwargs(policy(), batch_size, rank, world_size),
        region_profile_path=cfg["region_profile"],
        average_max_oversampling=20.0,
    )
    # A dedicated generator keeps DataLoader iterator creation from perturbing
    # model/dropout RNG during resume. This is also necessary in the original trainer.
    import torch

    generator = torch.Generator().manual_seed(policy()["seed"] + rank)
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        collate_fn=ds.collate_fn,
        num_workers=workers,
        persistent_workers=workers > 0,
        generator=generator,
    )
    binding = {
        "pool": sampler.pool_fingerprint,
        "quality": ds.quality_index.fingerprint,
        "sampling": sampler.sampler_config_fingerprint,
        "region_profile": sampler.region_profile_fingerprint,
    }
    return loader, sampler, binding


def prepare(path):
    cfg = read_data_config(path)
    for name in ["pool_dir", "quality_dir"]:
        if not Path(cfg[name]).resolve().is_relative_to(ROOT):
            raise ValueError(
                "New pool/QC outputs must be inside this independent package"
            )
    if not Path(cfg["region_profile"]).resolve().is_relative_to(ROOT):
        raise ValueError("New profile must be inside this independent package")
    base = base_dataset(cfg)
    qc = {
        "sample_nan_ratio_threshold": 0.0,
        "min_past_target_valid_ratio": 1.0,
        "min_future_target_valid_ratio": 1.0,
        "min_history_covariate_valid_ratio": 1.0,
        "min_future_covariate_valid_ratio": 1.0,
        "min_history_covariate_std": 0.0,
        "min_future_covariate_std": 0.0,
        "min_future_target_std": 0.0,
        "max_future_zero_run_hours": 0.0,
        "max_future_constant_run_hours": 0.0,
        "future_zero_tolerance": 1e-6,
        "future_constant_tolerance": 1e-6,
        "min_future_target_range": 0.0,
        "decoder_covariate_context_len": 0,
    }
    config = {
        "source_weights": {"1h": 0.75, "15min": 0.25},
        "region_balance_alpha": 0.25,
        "train_window_sample_stride": 1,
        "context_max_hours": 672,
        "horizon_max_hours": 168,
        "native_context_max_hours": 96.0,
        "native_horizon_max_hours": 24.0,
        "qc_config": qc,
        "registry": cfg["binary_cache_registry"],
        "manifest": cfg["manifest_path"],
    }
    pool = build_temporal_pool(base, cfg["pool_dir"], config=config)
    ds = TemporalPoolDataset(
        base, pool, 672, 168, dynamic_native_task_names=("pv_15min_24h_ahead",)
    )
    ds.ensure_quality_index(cfg["quality_dir"])
    sampler = TemporalPoolSampler(ds, **sampler_kwargs(policy(), 256, 0, 1))
    build_or_load_average_supply(
        ds,
        sampler,
        Path(cfg["region_profile"]).parent,
        samples_per_source=5000,
        calibration_seed=20260913,
        alpha=0.25,
        max_oversampling=20.0,
        max_region_probability=0.15,
        log=print,
    )
