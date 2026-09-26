import math
import random
from collections import Counter, defaultdict

from torch.utils.data import Sampler

from foundation.data.pool_sampling import (
    SamplingAudit,
    ShuffleCyclePoolDrawer,
    normalize_pool_draw_mode,
)


_CHINA_REGION_MAP = {
    "安徽": "Anhui",
    "广西": "Guangxi",
    "河北": "Hebei",
    "新疆": "Xinjiang",
    "广东": "Guangdong",
    "湖北": "Hubei",
    "云南": "Yunnan",
}


def _clean_text(value):
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _normalize_region_name(region):
    region = _clean_text(region)
    return _CHINA_REGION_MAP.get(region, region) or "UNKNOWN"


def canonical_region_key(station_record):
    """Return the official region key used for multi-region balancing.

    Manifest rows for Chinese provinces often have region=China_Mainland, so
    station_dir/source_region are preferred to avoid collapsing Anhui/Guangxi
    etc. into one bucket.
    """

    row = station_record or {}
    station_dir = _clean_text(row.get("station_dir")).replace("\\", "/")
    parts = [part for part in station_dir.split("/") if part]

    split_region = _clean_text(row.get("split_region"))
    if split_region:
        return _normalize_region_name(split_region)

    official_region = _clean_text(row.get("official_region"))
    if official_region:
        return _normalize_region_name(official_region)

    source_region = _clean_text(row.get("source_region"))
    if source_region and source_region not in {"China_Mainland", "China"}:
        return _normalize_region_name(source_region)

    if len(parts) >= 3 and parts[1] == "China_Mainland":
        return _normalize_region_name(parts[2])
    if len(parts) >= 2:
        return _normalize_region_name(parts[1])

    region = _clean_text(row.get("region"))
    if region and region not in {"China_Mainland", "China"}:
        return _normalize_region_name(region)
    if source_region:
        return _normalize_region_name(source_region)
    if region:
        return _normalize_region_name(region)
    return "UNKNOWN"


def sample_weights_from_region_keys(region_keys, alpha):
    alpha = max(0.0, float(alpha or 0.0))
    if not region_keys:
        return []
    if alpha <= 0.0:
        return [1.0 for _ in region_keys]
    counts = Counter(region_keys)
    raw_by_region = {
        region: float(count) ** (-alpha)
        for region, count in counts.items()
        if count > 0
    }
    weighted_total = sum(float(counts[region]) * raw for region, raw in raw_by_region.items())
    scale = float(len(region_keys)) / max(weighted_total, 1e-12)
    return [raw_by_region.get(region, 1.0) * scale for region in region_keys]


class RegionBalancedBatchSampler(Sampler):
    """Indexed batch sampler that samples each batch from one canonical region."""

    def __init__(
        self,
        dataset,
        batch_size,
        drop_last=False,
        shuffle=True,
        region_balance_alpha=0.5,
        region_balance_max_prob=0.0,
        region_balance_max_repeat_per_epoch=0.0,
        seed=0,
        pool_draw_mode="replacement",
        rank=0,
        world_size=1,
        sampling_audit_dir="",
        sampling_audit_trace_batches=16,
        sampling_audit_condition_samples_per_pool=256,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.region_balance_alpha = max(0.0, float(region_balance_alpha or 0.0))
        self.region_balance_max_prob = max(0.0, float(region_balance_max_prob or 0.0))
        self.region_balance_max_repeat_per_epoch = max(
            0.0, float(region_balance_max_repeat_per_epoch or 0.0)
        )
        self.seed = int(seed or 0)
        self.pool_draw_mode = normalize_pool_draw_mode(pool_draw_mode)
        self.rank = int(rank or 0)
        self.world_size = max(1, int(world_size or 1))
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"rank={self.rank} must be in [0, {self.world_size}).")
        self.sampling_audit_dir = str(sampling_audit_dir or "")
        self.epoch = 0
        self._buckets = self._build_buckets()
        self._num_batches = self._compute_num_batches()
        self.region_summaries = self._build_region_summaries()
        self._region_keys = [item["region"] for item in self.region_summaries]
        self._region_probs = [item["batch_probability"] for item in self.region_summaries]
        self._cycle_drawer = ShuffleCyclePoolDrawer() if self.pool_draw_mode == "shuffle_cycle" else None
        self._sampling_audit = SamplingAudit(
            dataset,
            mode=self.pool_draw_mode,
            trace_batches=sampling_audit_trace_batches,
            condition_samples_per_pool=sampling_audit_condition_samples_per_pool,
        ) if self.sampling_audit_dir else None

    def _region_key_for_index_item(self, index_item):
        dataset_idx = int(index_item[0])
        station_index = int(index_item[2])
        station_record = self.dataset.task_datasets[dataset_idx].station_records[station_index]
        return canonical_region_key(station_record)

    def _build_buckets(self):
        buckets = defaultdict(list)
        for sample_index, index_item in enumerate(getattr(self.dataset, "index", [])):
            region = self._region_key_for_index_item(index_item)
            buckets[region].append(int(sample_index))
        if not buckets:
            raise ValueError("RegionBalancedBatchSampler received an empty indexed dataset.")
        return {key: list(indices) for key, indices in sorted(buckets.items())}

    def _pool_key(self, region):
        task_dataset = self.dataset.task_datasets[0]
        return (
            str(task_dataset.task_spec.resolution),
            str(task_dataset.task_name),
            int(task_dataset.seq_len),
            int(task_dataset.pred_len),
            str(region),
        )

    def _compute_num_batches(self):
        total_samples = sum(len(indices) for indices in self._buckets.values())
        if self.drop_last:
            return total_samples // self.batch_size
        return int(math.ceil(float(total_samples) / float(self.batch_size)))

    @staticmethod
    def _normalize_weights(weights):
        total_weight = sum(weights.values())
        if total_weight <= 0.0:
            raise ValueError("RegionBalancedBatchSampler region weights are all zero.")
        return {region: weight / total_weight for region, weight in weights.items()}

    @staticmethod
    def _apply_probability_caps(raw_probs, upper_bounds):
        eps = 1e-12
        regions = list(raw_probs)
        if not regions:
            return {}
        if not any(math.isfinite(upper_bounds.get(region, math.inf)) for region in regions):
            return dict(raw_probs)
        finite_upper_sum = sum(
            upper_bounds[region]
            for region in regions
            if math.isfinite(upper_bounds.get(region, math.inf))
        )
        infinite_regions = [
            region for region in regions if not math.isfinite(upper_bounds.get(region, math.inf))
        ]
        if finite_upper_sum < 1.0 - eps and not infinite_regions:
            raise ValueError(
                "RegionBalancedBatchSampler caps are infeasible: "
                f"sum of region upper probabilities is {finite_upper_sum:.6f} < 1.0."
            )

        remaining = set(regions)
        capped_probs = {}
        remaining_mass = 1.0
        while remaining:
            raw_total = sum(raw_probs[region] for region in remaining)
            if raw_total <= 0.0:
                share = remaining_mass / float(len(remaining))
                tentative = {region: share for region in remaining}
            else:
                tentative = {
                    region: remaining_mass * raw_probs[region] / raw_total
                    for region in remaining
                }
            violators = [
                region
                for region in remaining
                if tentative[region] > upper_bounds.get(region, math.inf) + eps
            ]
            if not violators:
                capped_probs.update(tentative)
                break
            for region in violators:
                upper = upper_bounds.get(region, math.inf)
                if not math.isfinite(upper):
                    continue
                capped_probs[region] = upper
                remaining_mass -= upper
                remaining.remove(region)
            if remaining_mass < -eps:
                raise ValueError("RegionBalancedBatchSampler caps produced negative remaining mass.")
            remaining_mass = max(0.0, remaining_mass)

        total_prob = sum(capped_probs.values())
        if total_prob <= 0.0:
            raise ValueError("RegionBalancedBatchSampler capped probabilities are all zero.")
        return {region: prob / total_prob for region, prob in capped_probs.items()}

    def _region_upper_probability(self, sample_count):
        upper = math.inf
        if self.region_balance_max_prob > 0.0:
            upper = min(upper, self.region_balance_max_prob)
        if self.region_balance_max_repeat_per_epoch > 0.0:
            denom = max(1.0, float(self._num_batches * self.batch_size))
            repeat_upper = self.region_balance_max_repeat_per_epoch * float(sample_count) / denom
            upper = min(upper, max(0.0, repeat_upper))
        return upper

    def _build_region_summaries(self):
        weights = {}
        for region, indices in self._buckets.items():
            count = len(indices)
            weights[region] = float(count) ** (-self.region_balance_alpha) if count > 0 else 0.0
        raw_probs = self._normalize_weights(weights)
        upper_bounds = {
            region: self._region_upper_probability(len(indices))
            for region, indices in self._buckets.items()
        }
        batch_probs = self._apply_probability_caps(raw_probs, upper_bounds)
        summaries = []
        default_batches = max(1, int(self._num_batches))
        for region in sorted(self._buckets):
            sample_count = len(self._buckets[region])
            raw_probability = raw_probs[region]
            probability = batch_probs[region]
            upper_probability = upper_bounds[region]
            cap_active = math.isfinite(upper_probability)
            expected_batches = probability * default_batches
            expected_repeat = expected_batches * float(self.batch_size) / max(1.0, float(sample_count))
            summaries.append(
                {
                    "region": region,
                    "sample_count": sample_count,
                    "raw_batch_probability": raw_probability,
                    "batch_probability": probability,
                    "cap_upper_probability": upper_probability if cap_active else 1.0,
                    "cap_active": cap_active,
                    "probability_capped": probability < raw_probability - 1e-12,
                    "expected_batches_per_epoch": expected_batches,
                    "expected_repeat_exposure": expected_repeat,
                }
            )
        return summaries

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _sample_batch_from_region(self, region, rng):
        candidates = self._buckets[region]
        if self.pool_draw_mode == "shuffle_cycle":
            return self._cycle_drawer.draw(
                self._pool_key(region), candidates, self.batch_size, rng, shuffle=self.shuffle
            )
        if len(candidates) >= self.batch_size:
            if self.shuffle:
                return rng.sample(candidates, self.batch_size), {"cycles_completed": 0, "forced_duplicates": 0}
            return list(candidates[: self.batch_size]), {"cycles_completed": 0, "forced_duplicates": 0}
        if not candidates:
            return [], {"cycles_completed": 0, "forced_duplicates": 0}
        return [rng.choice(candidates) for _ in range(self.batch_size)], {"cycles_completed": 0, "forced_duplicates": 0}

    def write_sampling_audit(self):
        if self._sampling_audit is None:
            return None
        return self._sampling_audit.write(self.sampling_audit_dir, rank=self.rank, world_size=self.world_size)

    def __iter__(self):
        # Keep legacy replacement bit-for-bit compatible: it has independent
        # rank RNG streams and one local logical epoch.  shuffle_cycle instead
        # advances one global sequence before assigning batches to ranks.
        if self.pool_draw_mode == "shuffle_cycle":
            yield from self._iter_shuffle_cycle_global()
            return
        rng = random.Random(self.seed + int(self.epoch) * 1000003)
        if self._num_batches <= 0:
            return
        for batch_idx in range(self._num_batches):
            if self.shuffle:
                region = rng.choices(self._region_keys, weights=self._region_probs, k=1)[0]
            else:
                region = self._region_keys[batch_idx % len(self._region_keys)]
            batch, metadata = self._sample_batch_from_region(region, rng)
            if self._sampling_audit is not None:
                self._sampling_audit.record(
                    self._pool_key(region), batch,
                    forced_duplicates=metadata["forced_duplicates"],
                    completed_cycles=metadata["cycles_completed"],
                )
            if len(batch) < self.batch_size and self.drop_last:
                continue
            if batch:
                yield batch
        self.epoch += 1

    def _iter_shuffle_cycle_global(self):
        rng = random.Random(self.seed + int(self.epoch) * 1000003)
        if self._num_batches <= 0:
            return
        for batch_idx in range(self._num_batches):
            if self.shuffle:
                region = rng.choices(self._region_keys, weights=self._region_probs, k=1)[0]
            else:
                region = self._region_keys[batch_idx % len(self._region_keys)]
            batch, metadata = self._sample_batch_from_region(region, rng)
            if self._sampling_audit is not None:
                self._sampling_audit.record(
                    self._pool_key(region), batch,
                    forced_duplicates=metadata["forced_duplicates"],
                    completed_cycles=metadata["cycles_completed"],
                )
            if batch_idx % self.world_size != self.rank:
                continue
            if len(batch) < self.batch_size and self.drop_last:
                continue
            if batch:
                yield batch
        self.epoch += 1

    def __len__(self):
        if self.pool_draw_mode != "shuffle_cycle":
            return self._num_batches
        local = len(range(self.rank, self._num_batches, self.world_size))
        return local if local > 0 else self._num_batches
