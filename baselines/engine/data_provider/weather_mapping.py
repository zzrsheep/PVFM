"""PVTC_V2 weather-feature registry used by the PVFM adapter.

The IDs intentionally match PV_benchmark/data_provider/weather_mapping.py.
PVTC_V2 consumes the IDs as metadata tokens; this module is adapter-side data
plumbing and does not change the frozen PVTC_V2 model.
"""

from __future__ import annotations

import fnmatch


FEATURE_GROUP_VOCAB = {
    "radiation": 0,
    "temperature": 1,
    "moisture": 2,
    "other": 3,
    "unknown": 4,
}


FEATURE_REGISTRY = {
    "ghi": {
        "feature_id": 0,
        "group_id": FEATURE_GROUP_VOCAB["radiation"],
        "aliases": ("ghi", "shortwave_radiation", "ssr", "ssrd", "surface solar radiation downwards"),
    },
    "dni": {
        "feature_id": 1,
        "group_id": FEATURE_GROUP_VOCAB["radiation"],
        "aliases": ("dni", "direct_normal_irradiance", "direct solar radiation"),
    },
    "dhi": {
        "feature_id": 2,
        "group_id": FEATURE_GROUP_VOCAB["radiation"],
        "aliases": ("dhi", "diffuse_radiation"),
    },
    "temperature": {
        "feature_id": 3,
        "group_id": FEATURE_GROUP_VOCAB["temperature"],
        "aliases": ("temperature", "temperature_2m", "2 metre temperature", "t2m"),
    },
    "humidity": {
        "feature_id": 4,
        "group_id": FEATURE_GROUP_VOCAB["moisture"],
        "aliases": ("humidity", "relative_humidity", "relative_humidity_2m", "rh"),
    },
    "precipitation": {
        "feature_id": 5,
        "group_id": FEATURE_GROUP_VOCAB["moisture"],
        "aliases": ("precipitation", "rain"),
    },
    "cloudcover": {
        "feature_id": 6,
        "group_id": FEATURE_GROUP_VOCAB["other"],
        "aliases": ("cloud_cover", "total cloud cover", "tcc"),
    },
    "pressure": {
        "feature_id": 7,
        "group_id": FEATURE_GROUP_VOCAB["other"],
        "aliases": ("surface_pressure", "pressure", "sp", "msl"),
    },
    "windspeed": {
        "feature_id": 8,
        "group_id": FEATURE_GROUP_VOCAB["other"],
        "aliases": ("wind_speed_10m", "wind_speed", "ws10m"),
    },
    "winddirection": {
        "feature_id": 9,
        "group_id": FEATURE_GROUP_VOCAB["other"],
        "aliases": ("wind_direction", "wind_direction_10m", "wd10m"),
    },
}

UNKNOWN_FEATURE_ID = len(FEATURE_REGISTRY)
UNKNOWN_GROUP_ID = FEATURE_GROUP_VOCAB["unknown"]


def get_feature_spec(name: str | None) -> dict[str, int | str | None]:
    """Return the frozen PVTC feature/group IDs for a PVFM column name."""
    text = str(name or "").strip().lower()
    for canonical_name, spec in FEATURE_REGISTRY.items():
        if text == canonical_name:
            return {"canonical_name": canonical_name, **spec}
        for alias in spec["aliases"]:
            alias_text = str(alias).lower()
            if ("*" in alias_text and fnmatch.fnmatch(text, alias_text)) or text == alias_text:
                return {"canonical_name": canonical_name, **spec}
    return {
        "canonical_name": None,
        "feature_id": UNKNOWN_FEATURE_ID,
        "group_id": UNKNOWN_GROUP_ID,
    }
