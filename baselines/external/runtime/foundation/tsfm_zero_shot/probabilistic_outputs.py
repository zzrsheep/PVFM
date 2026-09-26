# Mechanically extracted; original definitions and internal namespaces retained.
from __future__ import annotations

from typing import Iterable

import numpy as np

DEFAULT_QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def validate_quantile_levels(levels: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(level) for level in levels)
    if len(values) < 2:
        raise ValueError("At least two quantile levels are required.")
    if any(not 0.0 < level < 1.0 for level in values):
        raise ValueError(f"Quantile levels must be strictly between 0 and 1: {values}")
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError(f"Quantile levels must be strictly increasing and unique: {values}")
    if not any(np.isclose(level, 0.5, rtol=0.0, atol=1e-8) for level in values):
        raise ValueError("The probability forecast contract must include q0.5 for point metrics.")
    return values


def quantile_key(level: float) -> str:
    return f"{float(level):g}"


def quantile_index(levels: Iterable[float], requested: float) -> int:
    values = tuple(float(level) for level in levels)
    matches = [index for index, level in enumerate(values) if np.isclose(level, requested, rtol=0.0, atol=1e-8)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one q{requested:g} in quantile levels {values}.")
    return matches[0]

