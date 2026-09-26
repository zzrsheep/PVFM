import csv


def load_station_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def filter_manifest_rows(rows, resolution=None, allowed_granularities=None, station_dirs=None, regions=None, max_stations=0):
    filtered = rows
    if allowed_granularities:
        allowed = {item.strip() for item in allowed_granularities if item and item.strip()}
        filtered = [
            row
            for row in filtered
            if not row.get("granularity", "") or row.get("granularity", "") in allowed
        ]
    elif resolution:
        filtered = [row for row in filtered if row.get("granularity", "") == resolution]
    if regions:
        allowed_regions = {item.strip() for item in regions if item and item.strip()}
        filtered = [row for row in filtered if row.get("region", "") in allowed_regions]
    if station_dirs:
        allowed = {item.strip() for item in station_dirs if item and item.strip()}
        filtered = [row for row in filtered if row.get("station_dir", "") in allowed]
    if max_stations:
        filtered = filtered[:max(0, int(max_stations))]
    return filtered
