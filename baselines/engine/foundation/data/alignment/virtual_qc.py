"""Read-only application of a versioned virtual QC release.

The release produced by ``build_v2_clean_common_cohort.py`` is deliberately
an overlay: source CSV files and the frozen binary cache remain untouched.
This module applies only the recorded boundary crop and *new* point masks to
an in-memory hourly frame before the ordinary window-QC code runs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def _normalise_station_dir(value: object) -> str:
    return str(value or "").strip().replace("\\", "/")


def _normalise_times(values: pd.Series | list[object]) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(parsed, pd.DatetimeIndex):
        parsed = pd.Series(parsed)
    return parsed.dt.tz_convert(None)


class VirtualQCRelease:
    """Load and apply a data-only QC release without mutating its inputs."""

    BOUNDARY_FILE = "station_boundaries.csv"
    INTERVAL_FILE = "virtual_qc_mask_intervals.csv.gz"

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.boundary_path = self.root / self.BOUNDARY_FILE
        self.interval_path = self.root / self.INTERVAL_FILE
        if not self.boundary_path.exists():
            raise FileNotFoundError(f"virtual QC boundary file not found: {self.boundary_path}")

        boundaries = pd.read_csv(
            self.boundary_path,
            dtype={"station_dir": str},
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        if "station_dir" not in boundaries.columns:
            raise ValueError("station_boundaries.csv must contain station_dir")
        boundaries = boundaries.copy()
        boundaries["station_dir"] = boundaries["station_dir"].map(_normalise_station_dir)
        if boundaries["station_dir"].eq("").any():
            raise ValueError("station_boundaries.csv contains an empty station_dir")
        if boundaries["station_dir"].duplicated().any():
            duplicated = boundaries.loc[
                boundaries["station_dir"].duplicated(keep=False), "station_dir"
            ].tolist()
            raise ValueError(f"duplicate virtual QC boundary rows: {duplicated[:5]}")
        for column in ("boundary_start", "boundary_end"):
            if column not in boundaries.columns:
                boundaries[column] = ""
            boundaries[column] = _normalise_times(boundaries[column])
        invalid_bounds = (
            boundaries["boundary_start"].notna()
            & boundaries["boundary_end"].notna()
            & (boundaries["boundary_start"] > boundaries["boundary_end"])
        )
        if invalid_bounds.any():
            raise ValueError("virtual QC boundary_start is after boundary_end")
        self.boundaries = boundaries.set_index("station_dir", drop=False)

        if self.interval_path.exists():
            intervals = pd.read_csv(
                self.interval_path,
                dtype={"station_dir": str},
                keep_default_na=False,
                encoding="utf-8-sig",
            )
        else:
            intervals = pd.DataFrame()
        if intervals.empty:
            intervals = pd.DataFrame(
                columns=["station_dir", "start_time", "end_time", "hours"]
            )
        if "station_dir" not in intervals.columns:
            raise ValueError("virtual_qc_mask_intervals.csv.gz must contain station_dir")
        for column in ("start_time", "end_time"):
            if column not in intervals.columns:
                raise ValueError(
                    f"virtual_qc_mask_intervals.csv.gz must contain {column}"
                )
        intervals = intervals.copy()
        intervals["station_dir"] = intervals["station_dir"].map(_normalise_station_dir)
        intervals["start_time"] = _normalise_times(intervals["start_time"])
        intervals["end_time"] = _normalise_times(intervals["end_time"])
        if intervals[["start_time", "end_time"]].isna().any().any():
            raise ValueError("virtual QC interval contains an invalid timestamp")
        invalid_intervals = intervals["start_time"] > intervals["end_time"]
        if invalid_intervals.any():
            raise ValueError("virtual QC interval start_time is after end_time")
        # Older releases may contain source-QC-only intervals.  They are not
        # part of this overlay; source QC remains inherited from the source.
        # Require the explicit scope written by the corrected builder so an
        # empty/legacy scope cannot silently widen the source QC mask.
        if "mask_scope" in intervals.columns:
            allowed = intervals["mask_scope"].astype(str).str.strip().isin(
                {"new_virtual_candidate", "virtual_candidate"}
            )
            intervals = intervals.loc[allowed].copy()
        self.intervals = intervals.sort_values(
            ["station_dir", "start_time", "end_time"], kind="mergesort"
        ).reset_index(drop=True)
        self._intervals_by_station = {
            station_dir: group.reset_index(drop=True)
            for station_dir, group in self.intervals.groupby("station_dir", sort=False)
        }
        self._signature = self._build_signature()

    def _build_signature(self) -> str:
        digest = hashlib.sha256()
        for path in (self.boundary_path, self.interval_path):
            digest.update(path.name.encode("utf-8"))
            if not path.exists():
                digest.update(b"<missing>")
                continue
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
        return digest.hexdigest()

    @property
    def release_signature(self) -> str:
        return self._signature

    def __contains__(self, station_dir: object) -> bool:
        return _normalise_station_dir(station_dir) in self.boundaries.index

    def boundary_for(self, station_dir: object) -> pd.Series:
        key = _normalise_station_dir(station_dir)
        if key not in self.boundaries.index:
            raise KeyError(f"station has no virtual QC boundary row: {key}")
        return self.boundaries.loc[key]

    def apply(
        self,
        frame: pd.DataFrame,
        station_dir: object,
        *,
        target_columns: tuple[str, ...] | list[str] = ("target",),
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        """Apply the recorded crop and point mask to an in-memory frame.

        ``frame`` must contain an hourly ``timestamp`` column.  The returned
        frame is a copy; no caller-owned object is changed.  Covariates are
        intentionally untouched, so the normal strict aligned-QC procedure
        decides whether a resulting window remains usable.
        """
        if "timestamp" not in frame.columns:
            raise ValueError("virtual QC overlay requires a timestamp column")
        key = _normalise_station_dir(station_dir)
        boundary = self.boundary_for(key)
        out = frame.copy()
        timestamps = _normalise_times(out["timestamp"])
        if timestamps.isna().any():
            raise ValueError(f"invalid timestamp in virtual QC frame for {key}")
        out["timestamp"] = timestamps

        start = boundary.get("boundary_start")
        end = boundary.get("boundary_end")
        keep = np.ones(len(out), dtype=bool)
        if pd.notna(start):
            keep &= timestamps.to_numpy() >= pd.Timestamp(start).to_datetime64()
        if pd.notna(end):
            keep &= timestamps.to_numpy() <= pd.Timestamp(end).to_datetime64()
        cropped = int((~keep).sum())
        out = out.loc[keep].copy()

        interval_group = self._intervals_by_station.get(key)
        masked = np.zeros(len(out), dtype=bool)
        if interval_group is not None and not interval_group.empty and len(out):
            retained_times = out["timestamp"]
            for interval in interval_group.itertuples(index=False):
                masked |= (
                    (retained_times >= pd.Timestamp(interval.start_time))
                    & (retained_times <= pd.Timestamp(interval.end_time))
                ).to_numpy(dtype=bool)
            columns = [column for column in dict.fromkeys(target_columns) if column in out.columns]
            for column in columns:
                values = pd.to_numeric(out[column], errors="coerce").astype(float)
                values.loc[masked] = np.nan
                out[column] = values

        identity = bool(cropped == 0 and not masked.any())
        return out.reset_index(drop=True), {
            "station_dir": key,
            "boundary_start": "" if pd.isna(start) else str(pd.Timestamp(start)),
            "boundary_end": "" if pd.isna(end) else str(pd.Timestamp(end)),
            "cropped_rows": cropped,
            "masked_rows": int(masked.sum()),
            "identity_transform": identity,
            "release_signature": self.release_signature,
        }
