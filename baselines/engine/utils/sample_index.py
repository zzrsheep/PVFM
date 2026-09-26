import os

import pandas as pd


def _normalize_time(value):
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return text
    if getattr(parsed, "tzinfo", None) is not None:
        parsed = parsed.tz_convert(None)
    return pd.Timestamp(parsed).strftime("%Y-%m-%d %H:%M:%S")


def _normalize_times(values):
    values = list(values)
    if not values:
        return []
    parsed = pd.to_datetime(pd.Series(values), errors="coerce", utc=True)
    if parsed.notna().all():
        parsed = parsed.dt.tz_convert(None)
        return parsed.dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    normalized = []
    for value, parsed_value in zip(values, parsed):
        if pd.notna(parsed_value):
            normalized.append(pd.Timestamp(parsed_value).tz_convert(None).strftime("%Y-%m-%d %H:%M:%S"))
        else:
            normalized.append(_normalize_time(value))
    return normalized


def _normalize_station_dir(value):
    return str(value or "").strip().replace("\\", "/")


class SampleIndexFilter:
    """Whitelist sample windows by station and input/future timestamps."""

    def __init__(self, csv_path="", split=""):
        self.csv_path = str(csv_path or "")
        self.split = str(split or "").strip().lower()
        self.input_end_by_station_dir = {}
        self.input_end_by_station_id = {}
        self.input_end_global = set()
        self.future_start_by_station_dir = {}
        self.future_start_by_station_id = {}
        self.future_start_global = set()
        self.strict = False
        self.expected_keys = set()
        self.matched_keys = set()
        self.num_rows = 0
        if self.csv_path:
            self._load(self.csv_path)

    @property
    def enabled(self):
        return bool(self.csv_path)

    def _load(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"sample_index_csv not found: {csv_path}")
        frame = pd.read_csv(csv_path, low_memory=False)
        if self.split and "split" in frame.columns:
            split_values = frame["split"].fillna("").astype(str).str.strip().str.lower()
            frame = frame[(split_values == self.split) | (split_values == "all")].copy()
        self.num_rows = int(len(frame))
        if frame.empty:
            return

        if "strict" in frame.columns:
            strict_values = (
                frame["strict"]
                .fillna(False)
                .astype(str)
                .str.strip()
                .str.lower()
            )
            self.strict = bool(strict_values.isin({"1", "true", "yes", "y"}).any())

        station_dir_col = "station_dir" if "station_dir" in frame.columns else ""
        station_id_col = "station_id" if "station_id" in frame.columns else ""
        input_col = "input_end_time" if "input_end_time" in frame.columns else ""
        if not input_col and "input_end" in frame.columns:
            input_col = "input_end"
        future_col = ""
        for candidate in ("future_start_time", "target_start_time"):
            if candidate in frame.columns:
                future_col = candidate
                break
        if not input_col and not future_col:
            raise ValueError(
                "sample_index_csv must contain input_end_time/input_end or "
                "future_start_time/target_start_time"
            )

        station_dirs = (
            frame[station_dir_col].map(_normalize_station_dir).tolist()
            if station_dir_col else [""] * len(frame)
        )
        station_ids = (
            frame[station_id_col].fillna("").astype(str).str.strip().tolist()
            if station_id_col else [""] * len(frame)
        )
        input_times = _normalize_times(frame[input_col].tolist()) if input_col else [""] * len(frame)
        future_times = _normalize_times(frame[future_col].tolist()) if future_col else [""] * len(frame)

        for station_dir, station_id, input_end_time, future_start_time in zip(
            station_dirs, station_ids, input_times, future_times
        ):
            if input_end_time:
                self._add_time(
                    self.input_end_by_station_dir,
                    self.input_end_by_station_id,
                    self.input_end_global,
                    station_dir,
                    station_id,
                    input_end_time,
                )
            if future_start_time:
                self._add_time(
                    self.future_start_by_station_dir,
                    self.future_start_by_station_id,
                    self.future_start_global,
                    station_dir,
                    station_id,
                    future_start_time,
                )
            if self.strict:
                station_key = self._station_key(station_dir, station_id)
                key = (station_key, input_end_time, future_start_time)
                if key in self.expected_keys:
                    raise ValueError(f"duplicate strict sample-index row: {key}")
                self.expected_keys.add(key)

    @staticmethod
    def _station_key(station_dir, station_id):
        station_dir = _normalize_station_dir(station_dir)
        station_id = str(station_id or "").strip()
        if station_dir:
            return f"dir:{station_dir}"
        if station_id:
            return f"id:{station_id}"
        return "global:"

    def _strict_match_key(self, station_dir, station_id, input_end_time, future_start_time):
        station_keys = []
        normalized_dir = _normalize_station_dir(station_dir)
        normalized_id = str(station_id or "").strip()
        if normalized_dir:
            station_keys.append(f"dir:{normalized_dir}")
        if normalized_id:
            station_keys.append(f"id:{normalized_id}")
        station_keys.append("global:")

        for station_key in station_keys:
            for candidate in (
                (station_key, input_end_time, future_start_time),
                (station_key, input_end_time, ""),
                (station_key, "", future_start_time),
            ):
                if candidate in self.expected_keys:
                    return candidate
        return None

    @staticmethod
    def _add_time(by_dir, by_id, global_set, station_dir, station_id, timestamp):
        if station_dir:
            by_dir.setdefault(station_dir, set()).add(timestamp)
        if station_id:
            by_id.setdefault(station_id, set()).add(timestamp)
        if not station_dir and not station_id:
            global_set.add(timestamp)

    @staticmethod
    def _station_match(by_dir, by_id, global_set, station_dir, station_id, timestamp):
        if not timestamp:
            return False
        if timestamp in global_set:
            return True
        if station_dir and timestamp in by_dir.get(station_dir, set()):
            return True
        if station_id and timestamp in by_id.get(str(station_id), set()):
            return True
        return False

    def accepts(self, *, station_dir, station_id, input_end_time="", future_start_time=""):
        if not self.enabled:
            return True
        station_dir = _normalize_station_dir(station_dir)
        station_id = str(station_id or "").strip()
        input_end_time = _normalize_time(input_end_time)
        future_start_time = _normalize_time(future_start_time)
        return self.accepts_normalized(
            station_dir=station_dir,
            station_id=station_id,
            input_end_time=input_end_time,
            future_start_time=future_start_time,
        )

    def accepts_normalized(self, *, station_dir, station_id, input_end_time="", future_start_time=""):
        if not self.enabled:
            return True
        station_dir = _normalize_station_dir(station_dir)
        station_id = str(station_id or "").strip()
        if self.strict:
            match = self._strict_match_key(
                station_dir,
                station_id,
                input_end_time,
                future_start_time,
            )
            if match is None:
                return False
            self.matched_keys.add(match)
            return True
        has_input_filter = bool(
            self.input_end_global or self.input_end_by_station_dir or self.input_end_by_station_id
        )
        has_future_filter = bool(
            self.future_start_global or self.future_start_by_station_dir or self.future_start_by_station_id
        )
        input_match = (
            self._station_match(
                self.input_end_by_station_dir,
                self.input_end_by_station_id,
                self.input_end_global,
                station_dir,
                station_id,
                input_end_time,
            )
            if has_input_filter
            else False
        )
        future_match = (
            self._station_match(
                self.future_start_by_station_dir,
                self.future_start_by_station_id,
                self.future_start_global,
                station_dir,
                station_id,
                future_start_time,
            )
            if has_future_filter
            else False
        )
        return input_match or future_match

    def assert_complete(self):
        """Raise when a strict manifest row did not map to a valid dataset window."""
        if not self.strict:
            return
        missing = self.expected_keys - self.matched_keys
        if not missing:
            return
        examples = sorted(missing)[:5]
        raise ValueError(
            "strict sample-index manifest is incomplete: "
            f"matched={len(self.matched_keys)} expected={len(self.expected_keys)} "
            f"missing={len(missing)} examples={examples}"
        )


def filter_valid_starts_by_sample_index(
    valid_start_indices,
    timestamps,
    seq_len,
    pred_len,
    station_dir,
    station_id,
    sample_index_filter,
):
    if sample_index_filter is None or not sample_index_filter.enabled:
        return list(valid_start_indices)
    timestamps = list(timestamps)
    normalized_timestamps = _normalize_times(timestamps)
    kept = []
    for start in valid_start_indices:
        input_end_idx = int(start) + int(seq_len) - 1
        future_start_idx = int(start) + int(seq_len)
        future_end_idx = future_start_idx + int(pred_len) - 1
        if input_end_idx < 0 or future_end_idx >= len(timestamps):
            continue
        if sample_index_filter.accepts_normalized(
            station_dir=station_dir,
            station_id=station_id,
            input_end_time=normalized_timestamps[input_end_idx],
            future_start_time=normalized_timestamps[future_start_idx],
        ):
            kept.append(int(start))
    return kept
