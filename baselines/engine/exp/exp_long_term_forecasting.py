from data_provider.data_factory import data_provider
from exp.exp_basic_ascii import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from models.ProbabilisticOutputWrapper import (
    QuantileOutputWrapper,
    parse_quantile_levels,
)
from foundation.trainer.losses import masked_pinball_loss_components
from foundation.trainer.probabilistic_metrics import (
    ProbabilisticMetricAccumulator,
    station_equal_probabilistic_metrics,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import re
import json
import numpy as np
import pandas as pd
from utils.augmentation import run_augmentation, run_augmentation_single

try:
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, broadcast, gather_object
except ImportError:  # pragma: no cover - optional dependency
    Accelerator = None
    DistributedDataParallelKwargs = None
    broadcast = None
    gather_object = None

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        requested_quantiles = getattr(args, 'quantiles', '')
        self.quantile_levels = (
            parse_quantile_levels(requested_quantiles)
            if requested_quantiles
            else ()
        )
        self.quantile_median_index = (
            min(
                range(len(self.quantile_levels)),
                key=lambda index: abs(self.quantile_levels[index] - 0.5),
            )
            if self.quantile_levels
            else -1
        )
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.use_accelerate = bool(getattr(args, 'use_accelerate', False))
        self.accelerator = None
        self._accelerate_prepared = False
        if self.use_accelerate:
            if Accelerator is None:
                raise ImportError("accelerate is required when --use_accelerate is enabled")
            ddp_kwargs = None
            if (
                DistributedDataParallelKwargs is not None
                and bool(getattr(self.args, 'ddp_find_unused_parameters', False))
            ):
                # Opt in only for models with conditional parameter paths.
                # Crossformer and the standard baseline wrappers do not need
                # the additional graph traversal on every iteration.
                ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            self.accelerator = Accelerator(
                mixed_precision='fp16' if self.args.use_amp else 'no',
                kwargs_handlers=[ddp_kwargs] if ddp_kwargs is not None else None,
            )
            self.device = self.accelerator.device

    def _maybe_prepare_for_accelerate(self, *objects):
        if not self.use_accelerate:
            return objects
        return self.accelerator.prepare(*objects)

    def _autocast_context(self):
        if self.use_accelerate:
            return self.accelerator.autocast()
        return torch.cuda.amp.autocast()

    def _backward(self, loss):
        if self.use_accelerate:
            self.accelerator.backward(loss)
        else:
            loss.backward()

    def _gather_scalar_loss(self, loss):
        if self.use_accelerate:
            reduced = self.accelerator.reduce(loss.detach().float(), reduction="mean")
            return float(reduced.item())
        return float(loss.item())

    def _broadcast_stop_signal(self, should_stop):
        """Share main-rank early stopping with every Accelerate process."""
        if not self.use_accelerate:
            return bool(should_stop)
        if broadcast is None:
            raise ImportError("accelerate.utils.broadcast is required for distributed early stopping")
        signal = torch.tensor(
            [int(bool(should_stop)) if self._is_main_process else 0],
            device=self.device,
            dtype=torch.int32,
        )
        signal = broadcast(signal, from_process=0)
        return bool(signal.item())

    def _save_model_state(self, model, path):
        model_to_save = self.accelerator.unwrap_model(model) if self.use_accelerate else model
        torch.save(model_to_save.state_dict(), path)

    def _load_model_state(self, model, path):
        state_dict = torch.load(path, map_location="cpu")
        target_model = self.accelerator.unwrap_model(model) if self.use_accelerate else model
        target_model.load_state_dict(state_dict)

    def _checkpoint_path(self, setting):
        return os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')

    @property
    def _should_use_single_process_test(self):
        if not self.use_accelerate:
            return False
        return getattr(self.accelerator, 'num_processes', 1) > 1

    @property
    def _is_main_process(self):
        if self.use_accelerate:
            return self.accelerator.is_main_process
        return True

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 8:
            batch_x, batch_y, batch_x_mark, batch_y_mark, metadata, batch_w, seq_w_nwp_hist, seq_x_hist = batch
            extras = {
                'batch_w': batch_w,
                'seq_w_nwp_hist': seq_w_nwp_hist,
                'seq_x_hist': seq_x_hist,
            }
            return batch_x, batch_y, batch_x_mark, batch_y_mark, metadata, extras
        if len(batch) == 5:
            batch_x, batch_y, batch_x_mark, batch_y_mark, metadata = batch
            return batch_x, batch_y, batch_x_mark, batch_y_mark, metadata, {}
        if len(batch) == 6 and isinstance(batch[5], dict):
            batch_x, batch_y, batch_x_mark, batch_y_mark, metadata, extras = batch
            return batch_x, batch_y, batch_x_mark, batch_y_mark, metadata, extras
        if len(batch) == 4:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            return batch_x, batch_y, batch_x_mark, batch_y_mark, None, {}
        raise ValueError(f"Unexpected batch structure length: {len(batch)}")

    def _is_cross_unet_native(self):
        return (
            getattr(self.args, 'model', '') == 'Cross_Unet'
            and (getattr(self.args, 'nwp_mode', '') or '') == 'cross_unet_nwp_direct'
        )

    def _is_pvtc_v2_direct(self):
        return (
            getattr(self.args, 'model', '') in {'PVTC_V2', 'PVTC_Ablation'}
            and (getattr(self.args, 'nwp_mode', '') or '') == 'pvtc_v2_direct'
        )

    def _is_tide_direct(self):
        return (
            getattr(self.args, 'model', '') == 'TiDEOfficial'
            and (getattr(self.args, 'nwp_mode', '') or '') == 'tide_direct'
        )

    def _is_dag_external(self):
        return getattr(self.args, 'model', '') in {
            'DAGWrapper', 'GCGNetWrapper', 'TimeXerWrapper', 'TiDEBenchmarkWrapper'
        }

    def _model_aux_loss(self):
        model = self.accelerator.unwrap_model(self.model) if self.use_accelerate else self.model
        model = getattr(model, 'module', model)
        aux_loss = getattr(model, 'last_aux_loss', None)
        if aux_loss is None:
            return None
        if not isinstance(aux_loss, torch.Tensor):
            raise TypeError('model.last_aux_loss must be a torch.Tensor or None')
        return aux_loss.to(device=self.device)

    def _move_model_extras(self, extras, name, metadata):
        if not extras:
            return {}
        moved = {}
        for key, value in extras.items():
            moved[key] = value.float().to(self.device)
            self._ensure_finite(moved[key], f'{name}.{key}', metadata)
        return moved

    def _model_forward(self, model, batch_x, batch_x_mark, dec_inp, batch_y_mark, extras, batch_y=None):
        if getattr(self.args, 'model', '') == 'PVFMV6FullShot':
            return model(batch_x, batch_x_mark, dec_inp, batch_y_mark, **extras)
        if self._is_cross_unet_native():
            required = {'batch_w', 'seq_w_nwp_hist', 'seq_x_hist'}
            missing = sorted(required.difference(extras))
            if missing:
                raise ValueError(f"Cross_Unet direct NWP batch missing: {', '.join(missing)}")
            batch_x_target = batch_x[:, :, -1:]
            return model(
                batch_x_target,
                batch_x_mark,
                extras['batch_w'],
                batch_y_mark,
                extras['seq_w_nwp_hist'],
                extras['seq_x_hist'],
            )
        if self._is_pvtc_v2_direct():
            required = {
                'pvtc_static_feat', 'pvtc_timestamps', 'pvtc_feature_mask',
                'pvtc_nwp_mask', 'pvtc_hist_feature_ids', 'pvtc_hist_group_ids',
                'pvtc_nwp_feature_ids', 'pvtc_nwp_group_ids',
            }
            missing = sorted(required.difference(extras))
            if missing:
                raise ValueError(f"PVTC direct batch missing: {', '.join(missing)}")
            return model(
                batch_x,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                static_feat=extras['pvtc_static_feat'],
                timestamps=extras['pvtc_timestamps'],
                feature_mask=extras['pvtc_feature_mask'],
                nwp_mask=extras['pvtc_nwp_mask'],
                hist_feature_ids=extras['pvtc_hist_feature_ids'],
                hist_group_ids=extras['pvtc_hist_group_ids'],
                nwp_feature_ids=extras['pvtc_nwp_feature_ids'],
                nwp_group_ids=extras['pvtc_nwp_group_ids'],
            )
        if self._is_tide_direct():
            required = {'tide_past_time_features', 'tide_future_time_features'}
            missing = sorted(required.difference(extras))
            if missing:
                raise ValueError(f"TiDEOfficial direct batch missing: {', '.join(missing)}")
            return model(
                batch_x,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                past_time_features=extras['tide_past_time_features'],
                future_time_features=extras['tide_future_time_features'],
            )
        if self._is_dag_external():
            if getattr(self.args, 'model', '') == 'GCGNetWrapper':
                if batch_y is None:
                    raise ValueError('GCGNetWrapper requires batch_y for its training graph target')
                return model(batch_x, batch_x_mark, dec_inp, batch_y_mark, target_future=batch_y)
            return model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    @property
    def probabilistic_enabled(self):
        return bool(self.quantile_levels)

    @staticmethod
    def _split_model_output(model_output):
        """Return the point tensor and optional FM-style quantile payload."""
        if (
            isinstance(model_output, (tuple, list))
            and len(model_output) == 2
            and torch.is_tensor(model_output[0])
            and isinstance(model_output[1], dict)
        ):
            return model_output[0], model_output[1].get('quantiles')
        if not torch.is_tensor(model_output):
            raise TypeError(
                'Forecast model must return a tensor or (point, auxiliary) tuple, '
                f'got {type(model_output).__name__}'
            )
        return model_output, None

    def _target_slice(self, batch_y, metadata=None):
        """Select the scalar target channel used by the q-head."""
        future_target = batch_y[:, -self.args.pred_len:, :]
        if self.args.features == 'M':
            if future_target.shape[-1] != 1:
                raise ValueError(
                    'Probabilistic lightweight baselines require a scalar target. '
                    f'features=M produced {future_target.shape[-1]} target channels.'
                )
            return future_target
        # ``-1:0`` is an empty slice, so handle the multi-sensor target
        # explicitly instead of constructing a slice from a negative index.
        if self.args.features == 'MS':
            return future_target[..., -1:]
        return future_target[..., :1]

    def _prepare_forecast_tensors(self, model_output, batch_y, metadata):
        """Normalize point/probabilistic outputs to a common training layout."""
        point_output, quantiles = self._split_model_output(model_output)
        if self.probabilistic_enabled:
            if quantiles is None:
                raise RuntimeError(
                    'Probabilistic mode was requested but the model did not return '
                    'auxiliary quantiles.'
                )
            if quantiles.ndim != 3 or quantiles.shape[-1] != len(self.quantile_levels):
                raise ValueError(
                    'Quantile output must have shape [B, H, Q] with Q='
                    f'{len(self.quantile_levels)}, got {tuple(quantiles.shape)}.'
                )
            quantiles = quantiles[:, -self.args.pred_len:, :]
            target = self._target_slice(batch_y, metadata).to(self.device)
            point_output = quantiles[
                ..., self.quantile_median_index:self.quantile_median_index + 1
            ]
            return point_output, target, quantiles

        f_dim = -1 if self.args.features == 'MS' else 0
        point_output = point_output[:, -self.args.pred_len:, f_dim:]
        target = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        return point_output, target, None

    def _forecast_loss(self, model_output, batch_y, metadata):
        outputs, target, quantiles = self._prepare_forecast_tensors(
            model_output, batch_y, metadata
        )
        target_safe, target_mask = self._build_target_and_mask(target, metadata)
        self._ensure_finite(outputs, 'forecast.outputs', metadata)
        if quantiles is not None:
            self._ensure_finite(quantiles, 'forecast.quantiles', metadata)
            loss_sum, denominator = masked_pinball_loss_components(
                quantiles,
                target_safe,
                self.quantile_levels,
                mask=target_mask,
            )
            loss = loss_sum / denominator
        else:
            loss = self._masked_mse_loss(outputs, target_safe, target_mask)
        return loss, outputs, target_safe, target_mask, quantiles

    @staticmethod
    def _normalize_metadata(metadata):
        if metadata is None:
            return None
        if isinstance(metadata, list):
            return metadata
        if isinstance(metadata, dict):
            keys = list(metadata.keys())
            if not keys:
                return []
            batch_size = len(metadata[keys[0]])
            normalized = []
            for idx in range(batch_size):
                normalized.append({key: metadata[key][idx] for key in keys})
            return normalized
        return None

    @staticmethod
    def _flatten_metadata_objects(metadata):
        if metadata is None:
            return None
        if isinstance(metadata, dict):
            return [metadata]
        if isinstance(metadata, list):
            flattened = []
            for item in metadata:
                nested = Exp_Long_Term_Forecast._flatten_metadata_objects(item)
                if nested is None:
                    continue
                flattened.extend(nested)
            return flattened
        return None

    @staticmethod
    def _metadata_scalar(value, default=None):
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            return value.detach().cpu().reshape(-1)[0].item()
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            return value.reshape(-1)[0].item()
        return value

    @staticmethod
    def _metadata_string(value, default=''):
        value = Exp_Long_Term_Forecast._metadata_scalar(value, default)
        if value is None:
            return default
        if isinstance(value, bytes):
            return value.decode('utf-8', errors='replace')
        return str(value)

    @staticmethod
    def _metadata_float(value, default=np.nan):
        value = Exp_Long_Term_Forecast._metadata_scalar(value, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _metadata_int(value, default=-1):
        value = Exp_Long_Term_Forecast._metadata_scalar(value, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _append_prediction_points(self, path, pred, true, metadata, header):
        if not getattr(self.args, 'save_prediction_points', False):
            return header
        normalized_metadata = self._normalize_metadata(metadata)
        if normalized_metadata is None:
            return header

        rows = []
        pred_len = pred.shape[1]
        value_idx = pred.shape[-1] - 1
        for sample_idx, sample_meta in enumerate(normalized_metadata):
            if not isinstance(sample_meta, dict):
                continue
            future_start_raw = self._metadata_string(sample_meta.get('future_start_time', ''), '')
            future_start = pd.to_datetime(future_start_raw, errors='coerce')
            step_minutes = self._metadata_float(sample_meta.get('target_step_minutes', 60.0), 60.0)
            if not np.isfinite(step_minutes) or step_minutes <= 0:
                step_minutes = 60.0
            region = self._metadata_string(sample_meta.get('region', ''), '')
            station_id = self._metadata_string(sample_meta.get('station_id', ''), '')
            station_name = self._metadata_string(sample_meta.get('station_name', ''), '')
            station_dir = self._metadata_string(sample_meta.get('station_dir', ''), '')
            input_end_time = self._metadata_string(sample_meta.get('input_end_time', ''), '')
            source_region = self._metadata_string(sample_meta.get('source_region', ''), '')
            sample_index = self._metadata_int(sample_meta.get('sample_index', sample_idx), sample_idx)
            capacity_used_kw = self._metadata_float(sample_meta.get('capacity_used_kw', np.nan), np.nan)
            target_to_power_scale = self._metadata_float(sample_meta.get('target_to_power_scale', 1.0), 1.0)

            for horizon_idx in range(pred_len):
                target_time = ''
                if not pd.isna(future_start):
                    target_time = (
                        future_start + pd.to_timedelta(horizon_idx * step_minutes, unit='m')
                    ).strftime('%Y-%m-%d %H:%M:%S')
                y_pred = float(pred[sample_idx, horizon_idx, value_idx])
                y_true = float(true[sample_idx, horizon_idx, value_idx])
                rows.append({
                    'region': region,
                    'source_region': source_region,
                    'station_id': station_id,
                    'station_name': station_name,
                    'station_dir': station_dir,
                    'sample_index': sample_index,
                    'input_end_time': input_end_time,
                    'target_time': target_time,
                    'horizon': horizon_idx + 1,
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'is_valid': bool(np.isfinite(y_true) and np.isfinite(y_pred)),
                    'capacity_used_kw': capacity_used_kw,
                    'target_to_power_scale': target_to_power_scale,
                })

        if rows:
            pd.DataFrame(rows).to_csv(
                path,
                mode='a',
                header=header,
                index=False,
                encoding='utf-8-sig' if header else None,
                compression='gzip',
            )
            header = False
        return header

    def _append_quantile_prediction_points(self, path, quantiles, true, metadata, header):
        """Append scalar-target q9 rows using the shared prediction schema."""
        if not getattr(self.args, 'save_prediction_points', False):
            return header
        normalized_metadata = self._normalize_metadata(metadata)
        if normalized_metadata is None:
            return header

        rows = []
        pred_len = quantiles.shape[1]
        median_index = self.quantile_median_index
        for sample_idx, sample_meta in enumerate(normalized_metadata):
            if not isinstance(sample_meta, dict):
                continue
            future_start_raw = self._metadata_string(sample_meta.get('future_start_time', ''), '')
            future_start = pd.to_datetime(future_start_raw, errors='coerce')
            step_minutes = self._metadata_float(sample_meta.get('target_step_minutes', 60.0), 60.0)
            if not np.isfinite(step_minutes) or step_minutes <= 0:
                step_minutes = 60.0
            region = self._metadata_string(sample_meta.get('region', ''), '')
            station_id = self._metadata_string(sample_meta.get('station_id', ''), '')
            station_name = self._metadata_string(sample_meta.get('station_name', ''), '')
            station_dir = self._metadata_string(sample_meta.get('station_dir', ''), '')
            input_end_time = self._metadata_string(sample_meta.get('input_end_time', ''), '')
            source_region = self._metadata_string(sample_meta.get('source_region', ''), '')
            sample_index = self._metadata_int(sample_meta.get('sample_index', sample_idx), sample_idx)
            capacity_used_kw = self._metadata_float(sample_meta.get('capacity_used_kw', np.nan), np.nan)
            target_to_power_scale = self._metadata_float(
                sample_meta.get('target_to_power_scale', 1.0), 1.0
            )

            for horizon_idx in range(pred_len):
                target_time = ''
                if not pd.isna(future_start):
                    target_time = (
                        future_start + pd.to_timedelta(horizon_idx * step_minutes, unit='m')
                    ).strftime('%Y-%m-%d %H:%M:%S')
                row = {
                    'region': region,
                    'source_region': source_region,
                    'station_id': station_id,
                    'station_name': station_name,
                    'station_dir': station_dir,
                    'sample_index': sample_index,
                    'input_end_time': input_end_time,
                    'target_time': target_time,
                    'horizon': horizon_idx + 1,
                    'y_true': float(true[sample_idx, horizon_idx, 0]),
                    'y_pred': float(quantiles[sample_idx, horizon_idx, median_index]),
                    'is_valid': bool(
                        np.isfinite(true[sample_idx, horizon_idx, 0])
                        and np.all(np.isfinite(quantiles[sample_idx, horizon_idx]))
                    ),
                    'capacity_used_kw': capacity_used_kw,
                    'target_to_power_scale': target_to_power_scale,
                }
                for level_idx, level in enumerate(self.quantile_levels):
                    row[f'q_{level:g}'] = float(quantiles[sample_idx, horizon_idx, level_idx])
                rows.append(row)

        if rows:
            pd.DataFrame(rows).to_csv(
                path,
                mode='a',
                header=header,
                index=False,
                encoding='utf-8-sig' if header else None,
                compression='gzip',
            )
            header = False
        return header

    def _inverse_pv_quantiles_batch(self, quantiles, target, metadata, dataset):
        """Inverse-transform scalar-target quantiles with station scalers."""
        quantiles = quantiles.copy()
        target = target.copy()
        normalized_metadata = self._normalize_metadata(metadata)
        if normalized_metadata is None or self.args.data != 'pv_multires':
            return quantiles, target

        metric_space = (getattr(self.args, 'metric_space', 'power') or 'power').strip().lower()
        restore_power_scale = metric_space != 'normalized'
        for sample_idx, sample_meta in enumerate(normalized_metadata):
            if not isinstance(sample_meta, dict):
                continue
            station_idx = self._metadata_int(sample_meta.get('station_idx'), -1)
            scaler = None
            if 0 <= station_idx < len(dataset.station_scalers):
                scaler = dataset.station_scalers[station_idx]
            if scaler is None:
                continue

            q_shape = quantiles[sample_idx].shape
            quantiles[sample_idx] = scaler.inverse_transform(
                quantiles[sample_idx].reshape(-1, 1)
            ).reshape(q_shape)
            target_shape = target[sample_idx].shape
            target[sample_idx] = scaler.inverse_transform(
                target[sample_idx].reshape(-1, 1)
            ).reshape(target_shape)
            if restore_power_scale:
                power_scale = self._metadata_float(
                    sample_meta.get('target_to_power_scale', 1.0), 1.0
                )
                if np.isfinite(power_scale) and power_scale != 1.0:
                    quantiles[sample_idx] *= power_scale
                    target[sample_idx] *= power_scale
        return quantiles, target

    def _inverse_generic_quantiles_batch(self, quantiles, target, dataset):
        """Best-effort inverse transform for non-PV callers of this runner."""
        if not (getattr(dataset, 'scale', False) and getattr(self.args, 'inverse', False)):
            return quantiles, target
        quantiles = quantiles.copy()
        target = target.copy()
        q_shape = quantiles.shape
        t_shape = target.shape
        try:
            quantiles = dataset.inverse_transform(quantiles.reshape(-1, 1)).reshape(q_shape)
            target = dataset.inverse_transform(target.reshape(-1, 1)).reshape(t_shape)
        except (ValueError, IndexError):
            # A multivariate legacy scaler may require all feature columns;
            # leave the scalar probability output in its native metric space
            # instead of guessing a column mapping.
            return quantiles, target
        return quantiles, target

    def _inverse_pv_multires_batch(self, outputs, batch_y, metadata, dataset):
        outputs = outputs.copy()
        batch_y = batch_y.copy()
        metadata = self._normalize_metadata(metadata)
        if metadata is None or self.args.data != 'pv_multires':
            return outputs, batch_y
        metric_space = (getattr(self.args, 'metric_space', 'power') or 'power').strip().lower()
        restore_power_scale = metric_space != 'normalized'

        for sample_idx, sample_meta in enumerate(metadata):
            station_idx = sample_meta.get('station_idx')
            scaler = None
            if station_idx is not None and 0 <= station_idx < len(dataset.station_scalers):
                scaler = dataset.station_scalers[station_idx]
            if scaler is None:
                continue
            power_scale = float(sample_meta.get('target_to_power_scale', 1.0) or 1.0)
            target_idx = sample_meta.get('target_idx')
            if target_idx is None or int(target_idx) < 0:
                outputs[sample_idx] = scaler.inverse_transform(outputs[sample_idx])
                batch_y[sample_idx] = scaler.inverse_transform(batch_y[sample_idx])
                if restore_power_scale and power_scale != 1.0:
                    outputs[sample_idx] = outputs[sample_idx] * power_scale
                    batch_y[sample_idx] = batch_y[sample_idx] * power_scale
            else:
                outputs[sample_idx, :, -1:] = scaler.inverse_transform(outputs[sample_idx, :, -1:])
                batch_y[sample_idx, :, -1:] = scaler.inverse_transform(batch_y[sample_idx, :, -1:])
                if restore_power_scale and power_scale != 1.0:
                    outputs[sample_idx, :, -1:] = outputs[sample_idx, :, -1:] * power_scale
                    batch_y[sample_idx, :, -1:] = batch_y[sample_idx, :, -1:] * power_scale
        return outputs, batch_y

    def _inverse_pv_multires_input_batch(self, batch_x, metadata, dataset):
        batch_x = batch_x.copy()
        metadata = self._normalize_metadata(metadata)
        if metadata is None or self.args.data != 'pv_multires':
            return batch_x
        metric_space = (getattr(self.args, 'metric_space', 'power') or 'power').strip().lower()
        restore_power_scale = metric_space != 'normalized'

        for sample_idx, sample_meta in enumerate(metadata):
            station_idx = sample_meta.get('station_idx')
            scaler = None
            if station_idx is not None and 0 <= station_idx < len(dataset.station_scalers):
                scaler = dataset.station_scalers[station_idx]
            if scaler is None:
                continue
            power_scale = float(sample_meta.get('target_to_power_scale', 1.0) or 1.0)
            target_idx = sample_meta.get('target_idx')
            if target_idx is None or int(target_idx) < 0:
                batch_x[sample_idx] = scaler.inverse_transform(batch_x[sample_idx])
                if restore_power_scale and power_scale != 1.0:
                    batch_x[sample_idx] = batch_x[sample_idx] * power_scale
            else:
                safe_idx = min(max(int(target_idx), 0), batch_x.shape[-1] - 1)
                batch_x[sample_idx, :, safe_idx:safe_idx + 1] = scaler.inverse_transform(
                    batch_x[sample_idx, :, safe_idx:safe_idx + 1]
                )
                if restore_power_scale and power_scale != 1.0:
                    batch_x[sample_idx, :, safe_idx:safe_idx + 1] = (
                        batch_x[sample_idx, :, safe_idx:safe_idx + 1] * power_scale
                    )
        return batch_x

    @staticmethod
    def _ensure_numpy_batch(batch_like):
        if torch.is_tensor(batch_like):
            return batch_like.detach().cpu().numpy()
        if isinstance(batch_like, np.ndarray):
            return batch_like
        if isinstance(batch_like, list):
            if not batch_like:
                return np.asarray(batch_like)
            if isinstance(batch_like[0], np.ndarray):
                return np.concatenate(batch_like, axis=0)
            return np.asarray(batch_like)
        return np.asarray(batch_like)

    @staticmethod
    def _safe_station_token(station_id, station_name):
        station_id = "" if station_id is None else str(station_id)
        station_name = "" if station_name is None else str(station_name)
        ascii_name = re.sub(r"[^0-9A-Za-z._-]+", "_", station_name).strip("_")
        ascii_id = re.sub(r"[^0-9A-Za-z._-]+", "_", station_id).strip("_")
        if ascii_name and ascii_id:
            return f"{ascii_id}__{ascii_name}"
        if ascii_id:
            return ascii_id
        if ascii_name:
            return ascii_name
        return "station"

    def _build_decoder_input(self, batch_y, metadata):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
        normalized_metadata = self._normalize_metadata(metadata)
        if self.args.data != 'pv_multires' or normalized_metadata is None:
            return dec_inp.to(self.device)

        for sample_idx, sample_meta in enumerate(normalized_metadata):
            target_idx = int(sample_meta.get('target_idx', -1))
            if target_idx < 0:
                continue
            dec_inp[sample_idx, -self.args.pred_len:, :] = batch_y[sample_idx, -self.args.pred_len:, :]
            dec_inp[sample_idx, -self.args.pred_len:, target_idx] = 0
        return dec_inp.to(self.device)

    @staticmethod
    def _tensor_stats(tensor):
        detached = tensor.detach()
        finite_mask = torch.isfinite(detached)
        finite_values = detached[finite_mask]
        if finite_values.numel() == 0:
            return {'shape': tuple(detached.shape), 'finite': 0, 'min': None, 'max': None}
        return {
            'shape': tuple(detached.shape),
            'finite': int(finite_values.numel()),
            'min': float(finite_values.min().item()),
            'max': float(finite_values.max().item()),
        }

    def _ensure_finite(self, tensor, name, metadata=None):
        if torch.isfinite(tensor).all():
            return

        metadata = self._normalize_metadata(metadata)
        station_dirs = []
        if metadata is not None:
            station_dirs = [sample.get('station_dir', '') for sample in metadata[:5]]
        raise RuntimeError(
            f"Non-finite values detected in {name}: "
            f"stats={self._tensor_stats(tensor)} "
            f"sample_stations={station_dirs}"
        )

    def _build_target_and_mask(self, target, metadata=None):
        target_mask = torch.isfinite(target)
        if self.args.use_masked_future_loss:
            target_safe = torch.where(target_mask, target, torch.zeros_like(target))
            return target_safe, target_mask

        self._ensure_finite(target, 'target', metadata)
        return target, target_mask

    def _masked_mse_loss(self, pred, target_safe, target_mask):
        if self.args.use_masked_future_loss:
            loss_raw = F.mse_loss(pred, target_safe, reduction='none')
            mask = target_mask.float()
            valid_count = mask.sum().clamp_min(1.0)
            return (loss_raw * mask).sum() / valid_count
        return F.mse_loss(pred, target_safe)

    @staticmethod
    def _print_metric_line(title, mae, mse, rmse, mape, smape, r2):
        print(
            f"{title}: "
            f"MAE={mae:.6f}, "
            f"MSE={mse:.6f}, "
            f"RMSE={rmse:.6f}, "
            f"MAPE={mape:.6f}, "
            f"SMAPE={smape:.6f}, "
            f"R2={r2:.6f}"
        )

    def _build_model(self):
        if self.quantile_levels and self.args.model in {
            'Crossformer', 'Cross_Unet', 'DLinear', 'iTransformer',
            'LightTS', 'PatchTST', 'TimeMixer',
        }:
            from models.native_q9.local import (
                CrossUnetQ9, CrossformerQ9, DLinearQ9, ITransformerQ9,
                LightTSQ9, PatchTSTQ9, TimeMixerQ9,
            )
            native_cls = {
                'Crossformer': CrossformerQ9,
                'Cross_Unet': CrossUnetQ9,
                'DLinear': DLinearQ9,
                'iTransformer': ITransformerQ9,
                'LightTS': LightTSQ9,
                'PatchTST': PatchTSTQ9,
                'TimeMixer': TimeMixerQ9,
            }[self.args.model]
            model = native_cls(self.args).float()
        elif self.quantile_levels and self.args.model in {
            'DAGWrapper', 'GCGNetWrapper', 'TimeXerWrapper', 'TiDEBenchmarkWrapper',
        }:
            from models.native_q9.external_dag import (
                DAGNativeQ9, GCGNetNativeQ9, TimeXerNativeQ9, TiDENativeQ9,
            )
            native_cls = {
                'DAGWrapper': DAGNativeQ9,
                'GCGNetWrapper': GCGNetNativeQ9,
                'TimeXerWrapper': TimeXerNativeQ9,
                'TiDEBenchmarkWrapper': TiDENativeQ9,
            }[self.args.model]
            model = native_cls(self.args).float()
        else:
            model = self.model_dict[self.args.model](self.args).float()

        if self.quantile_levels and not bool(getattr(model, 'supports_native_quantiles', False)):
            # Prefer the model's declared forecast width. Fall back to c_out before
            # enc_in: many MS adapters keep enc_in as (target + covariates) while
            # returning only the target channel (c_out=1).
            output_dim = getattr(model, 'output_dim', None)
            if output_dim is None:
                output_dim = getattr(model, 'channels', None)
            if output_dim is None:
                output_dim = getattr(model, 'c_out', None)
            if output_dim is None:
                output_dim = getattr(self.args, 'c_out', None)
            if output_dim is None:
                output_dim = getattr(model, 'enc_in', None)
            if output_dim is None:
                output_dim = getattr(self.args, 'enc_in', None)
            if output_dim is None:
                raise ValueError(
                    f'Unable to infer forecast output width for probabilistic model {self.args.model}'
                )
            model = QuantileOutputWrapper(
                model,
                output_dim=int(output_dim),
                pred_len=int(self.args.pred_len),
                quantile_levels=self.quantile_levels,
                quantile_parameterization=getattr(
                    self.args, 'quantile_parameterization', 'independent'
                ),
                quantile_increment_scale=float(
                    getattr(self.args, 'quantile_increment_scale', 0.1)
                ),
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, metadata, extras = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                extras = self._move_model_extras(extras, 'vali', metadata)

                # decoder input
                dec_inp = self._build_decoder_input(batch_y, metadata)
                self._ensure_finite(batch_x, 'vali.batch_x', metadata)
                self._ensure_finite(dec_inp, 'vali.dec_inp', metadata)
                # encoder - decoder
                if self.args.use_amp:
                    with self._autocast_context():
                        model_output = self._model_forward(
                            self.model, batch_x, batch_x_mark, dec_inp, batch_y_mark, extras, batch_y
                        )
                        loss, outputs, target_safe, target_mask, _ = self._forecast_loss(
                            model_output, batch_y, metadata
                        )
                else:
                    model_output = self._model_forward(
                        self.model, batch_x, batch_x_mark, dec_inp, batch_y_mark, extras, batch_y
                    )
                    loss, outputs, target_safe, target_mask, _ = self._forecast_loss(
                        model_output, batch_y, metadata
                    )
                self._ensure_finite(loss, 'vali.loss', metadata)

                total_loss.append(self._gather_scalar_loss(loss))
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        # All DDP ranks enter train(). Avoid a check-then-create race here.
        os.makedirs(path, exist_ok=True)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.use_accelerate:
            train_loader, vali_loader, test_loader, self.model, model_optim = self._maybe_prepare_for_accelerate(
                train_loader, vali_loader, test_loader, self.model, model_optim
            )
            self._accelerate_prepared = True
            # Accelerate shards the training loader by rank. Use the local
            # post-sharding length for progress and ETA reporting.
            train_steps = len(train_loader)
        elif self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, metadata, extras = self._unpack_batch(batch)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                extras = self._move_model_extras(extras, 'train', metadata)

                # decoder input
                dec_inp = self._build_decoder_input(batch_y, metadata)
                self._ensure_finite(batch_x, 'train.batch_x', metadata)
                self._ensure_finite(dec_inp, 'train.dec_inp', metadata)

                # encoder - decoder
                if self.args.use_amp:
                    with self._autocast_context():
                        model_output = self._model_forward(
                            self.model, batch_x, batch_x_mark, dec_inp, batch_y_mark, extras, batch_y
                        )
                        loss, outputs, target_safe, target_mask, _ = self._forecast_loss(
                            model_output, batch_y, metadata
                        )
                        aux_loss = self._model_aux_loss()
                        if aux_loss is not None:
                            loss = loss + aux_loss
                        self._ensure_finite(loss, 'train.loss', metadata)
                        train_loss.append(self._gather_scalar_loss(loss))
                else:
                    model_output = self._model_forward(
                        self.model, batch_x, batch_x_mark, dec_inp, batch_y_mark, extras, batch_y
                    )
                    loss, outputs, target_safe, target_mask, _ = self._forecast_loss(
                        model_output, batch_y, metadata
                    )
                    aux_loss = self._model_aux_loss()
                    if aux_loss is not None:
                        loss = loss + aux_loss
                    self._ensure_finite(loss, 'train.loss', metadata)
                    train_loss.append(self._gather_scalar_loss(loss))

                if (i + 1) % 100 == 0:
                    logged_loss = self._gather_scalar_loss(loss)
                if self._is_main_process and (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, logged_loss))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.use_accelerate:
                    self._backward(loss)
                    model_optim.step()
                elif self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    self._backward(loss)
                    model_optim.step()

            if self.use_accelerate:
                self.accelerator.wait_for_everyone()

            if self._is_main_process:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            should_stop = False
            if self._is_main_process:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.accelerator.unwrap_model(self.model) if self.use_accelerate else self.model, path)
                should_stop = early_stopping.early_stop
            if self.use_accelerate:
                should_stop = self._broadcast_stop_signal(should_stop)
                self.accelerator.wait_for_everyone()
            if should_stop:
                if self._is_main_process:
                    print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self._load_model_state(self.model, best_model_path)

        return self.model

    def _run_test_loop(self, model, test_data, test_loader, folder_path, gather_distributed, prediction_points_path=None):
        preds = []
        trues = []
        station_series = {}
        station_debug_samples = {}
        probabilistic_state = None
        if self.probabilistic_enabled:
            probabilistic_state = {
                'overall': ProbabilisticMetricAccumulator(self.quantile_levels),
                'stations': {},
                'regions': {},
                'horizons': {},
                'quantile_preds': [],
                'quantile_trues': [],
            }
        prediction_points_needs_header = True
        if prediction_points_path and os.path.exists(prediction_points_path):
            os.remove(prediction_points_path)

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, metadata, extras = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                extras = self._move_model_extras(extras, 'test', metadata)

                dec_inp = self._build_decoder_input(batch_y, metadata)
                self._ensure_finite(batch_x, 'test.batch_x', metadata)
                self._ensure_finite(dec_inp, 'test.dec_inp', metadata)
                if self.args.use_amp:
                    with self._autocast_context():
                        model_output = self._model_forward(
                            model, batch_x, batch_x_mark, dec_inp, batch_y_mark, extras, batch_y
                        )
                else:
                    model_output = self._model_forward(
                        model, batch_x, batch_x_mark, dec_inp, batch_y_mark, extras, batch_y
                    )

                if self.probabilistic_enabled:
                    _, target_tensor, quantile_tensor = self._prepare_forecast_tensors(
                        model_output, batch_y, metadata
                    )
                    self._ensure_finite(quantile_tensor, 'test.quantiles', metadata)
                    quantile_tensor = quantile_tensor.detach().float()
                    target_tensor = target_tensor.detach().float()
                    if gather_distributed:
                        quantile_tensor = self.accelerator.gather_for_metrics(quantile_tensor)
                        target_tensor = self.accelerator.gather_for_metrics(target_tensor)
                        if metadata is not None and gather_object is not None:
                            normalized_metadata = self._normalize_metadata(metadata)
                            metadata = self._flatten_metadata_objects(
                                gather_object(normalized_metadata)
                            )
                        else:
                            metadata = None

                    quantiles = self._ensure_numpy_batch(quantile_tensor)
                    true = self._ensure_numpy_batch(target_tensor)
                    if self.args.data == 'pv_multires':
                        quantiles, true = self._inverse_pv_quantiles_batch(
                            quantiles, true, metadata, test_data
                        )
                    else:
                        quantiles, true = self._inverse_generic_quantiles_batch(
                            quantiles, true, test_data
                        )
                    point = quantiles[
                        :, :, self.quantile_median_index:self.quantile_median_index + 1
                    ]
                    if true.ndim == 2:
                        true = true[:, :, None]

                    probabilistic_state['quantile_preds'].append(quantiles)
                    probabilistic_state['quantile_trues'].append(true)
                    probabilistic_state['overall'].update(
                        quantiles, true[..., 0]
                    )
                    for forecast_step in range(quantiles.shape[1]):
                        horizon_acc = probabilistic_state['horizons'].setdefault(
                            forecast_step + 1,
                            ProbabilisticMetricAccumulator(self.quantile_levels),
                        )
                        horizon_acc.update(
                            quantiles[:, forecast_step:forecast_step + 1, :],
                            true[:, forecast_step:forecast_step + 1, 0],
                        )

                    normalized_metadata = self._normalize_metadata(metadata)
                    if prediction_points_path is not None:
                        prediction_points_needs_header = self._append_quantile_prediction_points(
                            prediction_points_path,
                            quantiles,
                            true,
                            normalized_metadata,
                            prediction_points_needs_header,
                        )
                    if normalized_metadata is not None:
                        for sample_idx, sample_meta in enumerate(normalized_metadata):
                            if not isinstance(sample_meta, dict):
                                continue
                            station_key = (
                                sample_meta.get('region', ''),
                                sample_meta.get('station_id', ''),
                                sample_meta.get('station_name', ''),
                                sample_meta.get('station_dir', ''),
                            )
                            station_payload = probabilistic_state['stations'].setdefault(
                                station_key,
                                {
                                    'region': sample_meta.get('region', ''),
                                    'station_id': sample_meta.get('station_id', ''),
                                    'station_name': sample_meta.get('station_name', ''),
                                    'station_dir': sample_meta.get('station_dir', ''),
                                    'acc': ProbabilisticMetricAccumulator(self.quantile_levels),
                                },
                            )
                            station_payload['acc'].update(
                                quantiles[sample_idx], true[sample_idx, :, 0]
                            )
                            region = sample_meta.get('region', '')
                            region_payload = probabilistic_state['regions'].setdefault(
                                region,
                                {
                                    'region': region,
                                    'acc': ProbabilisticMetricAccumulator(self.quantile_levels),
                                },
                            )
                            region_payload['acc'].update(
                                quantiles[sample_idx], true[sample_idx, :, 0]
                            )
                            if station_key not in station_series:
                                station_series[station_key] = {'pred': [], 'true': []}
                            station_series[station_key]['pred'].append(point[sample_idx])
                            station_series[station_key]['true'].append(true[sample_idx])
                            if self.args.save_station_debug_arrays:
                                if station_key not in station_debug_samples:
                                    station_debug_samples[station_key] = []
                                if len(station_debug_samples[station_key]) < int(self.args.station_debug_max_samples):
                                    target_idx = int(sample_meta.get('target_idx', -1))
                                    batch_x_np = batch_x.detach().cpu().numpy()
                                    safe_idx = (
                                        batch_x_np.shape[-1] - 1
                                        if target_idx < 0
                                        else min(max(target_idx, 0), batch_x_np.shape[-1] - 1)
                                    )
                                    station_debug_samples[station_key].append({
                                        'history_target': batch_x_np[sample_idx, :, safe_idx].astype(np.float32),
                                        'future_true': true[sample_idx, :, -1].astype(np.float32),
                                        'future_pred': point[sample_idx, :, -1].astype(np.float32),
                                        'history_time_features': batch_x_mark[sample_idx].detach().cpu().numpy().astype(np.float32),
                                        'future_time_features': batch_y_mark[sample_idx].detach().cpu().numpy().astype(np.float32),
                                    })

                    preds.append(point)
                    trues.append(true)
                    if self._is_main_process and (not self.args.disable_test_visuals) and i % 20 == 0:
                        input = batch_x.detach().cpu().numpy()
                        sample_true = true[0]
                        sample_pred = point[0]
                        gt = np.concatenate((input[0, :, -1], sample_true[:, -1]), axis=0)
                        pred_plot = np.concatenate((input[0, :, -1], sample_pred[:, -1]), axis=0)
                        visual(gt, pred_plot, os.path.join(folder_path, str(i) + '.pdf'))
                    continue

                outputs = model_output

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                self._ensure_finite(outputs, 'test.outputs', metadata)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if self.args.data == 'pv_multires':
                    outputs, batch_y = self._inverse_pv_multires_batch(outputs, batch_y, metadata, test_data)
                elif test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                batch_x_np = batch_x.detach().cpu().numpy()
                if self.args.data == 'pv_multires':
                    batch_x_np = self._inverse_pv_multires_input_batch(batch_x_np, metadata, test_data)
                elif test_data.scale and self.args.inverse:
                    shape = batch_x_np.shape
                    batch_x_np = test_data.inverse_transform(batch_x_np.reshape(shape[0] * shape[1], -1)).reshape(shape)

                pred = outputs
                true = batch_y
                if gather_distributed:
                    pred = self.accelerator.gather_for_metrics(pred)
                    true = self.accelerator.gather_for_metrics(true)
                    if metadata is not None and gather_object is not None:
                        normalized_metadata = self._normalize_metadata(metadata)
                        metadata = self._flatten_metadata_objects(gather_object(normalized_metadata))
                    else:
                        metadata = None

                pred = self._ensure_numpy_batch(pred)
                true = self._ensure_numpy_batch(true)

                normalized_metadata = self._normalize_metadata(metadata)
                if prediction_points_path is not None:
                    prediction_points_needs_header = self._append_prediction_points(
                        prediction_points_path,
                        pred,
                        true,
                        normalized_metadata,
                        prediction_points_needs_header,
                    )
                if normalized_metadata is not None:
                    for sample_idx, sample_meta in enumerate(normalized_metadata):
                        if not isinstance(sample_meta, dict):
                            continue
                        station_key = (
                            sample_meta.get('region', ''),
                            sample_meta.get('station_id', ''),
                            sample_meta.get('station_name', ''),
                            sample_meta.get('station_dir', ''),
                        )
                        if station_key not in station_series:
                            station_series[station_key] = {'pred': [], 'true': []}
                        station_series[station_key]['pred'].append(pred[sample_idx])
                        station_series[station_key]['true'].append(true[sample_idx])
                        if self.args.save_station_debug_arrays:
                            if station_key not in station_debug_samples:
                                station_debug_samples[station_key] = []
                            if len(station_debug_samples[station_key]) < int(self.args.station_debug_max_samples):
                                target_idx = int(sample_meta.get('target_idx', -1))
                                safe_idx = batch_x_np.shape[-1] - 1 if target_idx < 0 else min(max(target_idx, 0), batch_x_np.shape[-1] - 1)
                                station_debug_samples[station_key].append({
                                    'history_target': batch_x_np[sample_idx, :, safe_idx].astype(np.float32),
                                    'future_true': true[sample_idx, :, -1].astype(np.float32),
                                    'future_pred': pred[sample_idx, :, -1].astype(np.float32),
                                    'history_time_features': batch_x_mark[sample_idx].detach().cpu().numpy().astype(np.float32),
                                    'future_time_features': batch_y_mark[sample_idx].detach().cpu().numpy().astype(np.float32),
                                })

                preds.append(pred)
                trues.append(true)
                if self._is_main_process and (not self.args.disable_test_visuals) and i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    sample_true = true[0] if true.ndim == 3 else true
                    sample_pred = pred[0] if pred.ndim == 3 else pred
                    gt = np.concatenate((input[0, :, -1], sample_true[:, -1]), axis=0)
                    pred_plot = np.concatenate((input[0, :, -1], sample_pred[:, -1]), axis=0)
                    visual(gt, pred_plot, os.path.join(folder_path, str(i) + '.pdf'))

        return preds, trues, station_series, station_debug_samples, probabilistic_state

    def test(self, setting, test=0):
        # Keep audit/evaluation artifacts inside the caller-selected run
        # directory.  The environment variables are optional so historical
        # experiments retain their original project-root layout.
        test_results_root = os.environ.get('PVFM_TEST_RESULTS_ROOT', './test_results')
        results_root = os.environ.get('PVFM_RESULTS_ROOT', './results')
        result_log_path = os.environ.get(
            'PVFM_RESULT_LOG',
            os.path.join(results_root, 'result_long_term_forecast.txt'),
        )
        folder_path = os.path.join(test_results_root, setting, '')
        if self._is_main_process and not os.path.exists(folder_path):
            os.makedirs(folder_path)
        result_folder_path = os.path.join(results_root, setting, '')
        prediction_points_path = None
        if self._is_main_process:
            if not os.path.exists(result_folder_path):
                os.makedirs(result_folder_path)
            if getattr(self.args, 'save_prediction_points', False):
                prediction_points_path = os.path.join(result_folder_path, 'prediction_points.csv.gz')
        if self._should_use_single_process_test:
            if not self._is_main_process:
                # Rank 0 evaluates with a fresh non-DDP model below. Waiting
                # here would hold the other ranks in a NCCL barrier for the
                # entire single-process evaluation and eventually time out.
                return
            if test:
                print('loading model')
            test_data, test_loader = self._get_data(flag='test')
            eval_model = self._build_model().to(self.device)
            if test:
                self._load_model_state(eval_model, self._checkpoint_path(setting))
            else:
                eval_model.load_state_dict(self.accelerator.unwrap_model(self.model).state_dict())
            preds, trues, station_series, station_debug_samples, probabilistic_state = self._run_test_loop(
                eval_model,
                test_data,
                test_loader,
                folder_path,
                gather_distributed=False,
                prediction_points_path=prediction_points_path,
            )
        else:
            test_data, test_loader = self._get_data(flag='test')
            if self.use_accelerate and not self._accelerate_prepared:
                test_loader, self.model = self._maybe_prepare_for_accelerate(test_loader, self.model)
                self._accelerate_prepared = True
            if test:
                if self._is_main_process:
                    print('loading model')
                self._load_model_state(self.model, self._checkpoint_path(setting))
            preds, trues, station_series, station_debug_samples, probabilistic_state = self._run_test_loop(
                self.model,
                test_data,
                test_loader,
                folder_path,
                gather_distributed=self.use_accelerate,
                prediction_points_path=prediction_points_path,
            )
            if self.use_accelerate:
                self.accelerator.wait_for_everyone()
            if not self._is_main_process:
                return

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        if self.args.data == 'pv_multires':
            print(f"metric_space={getattr(self.args, 'metric_space', 'power')}")

        # result save
        folder_path = result_folder_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, smape, r2 = metric(preds, trues)
        self._print_metric_line('Overall test metrics', mae, mse, rmse, mape, smape, r2)
        os.makedirs(os.path.dirname(result_log_path) or '.', exist_ok=True)
        f = open(result_log_path, 'a')
        f.write(setting + "  \n")
        f.write(
            'MAE:{}, MSE:{}, RMSE:{}, MAPE:{}, SMAPE:{}, R2:{}'.format(
                mae, mse, rmse, mape, smape, r2
            )
        )
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, smape, r2]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        if prediction_points_path is not None and os.path.exists(prediction_points_path):
            print(f"Saved prediction points to: {prediction_points_path}")

        probabilistic_overall_metrics = {}
        probabilistic_station_metrics = {}
        probabilistic_region_metrics = {}
        probabilistic_horizon_metrics = {}
        probabilistic_metric_names = (
            'aql',
            'crps',
            'quantile_crossing_rate',
            'quantile_crossing_rate_any',
            'quantile_crossing_rate_pairwise',
            'pairwise_crossing_rate',
            'quantile_count',
            'p10_p90_coverage',
            'p10_p90_mean_width',
            'p05_p95_coverage',
            'p05_p95_mean_width',
        )
        if probabilistic_state is not None:
            quantile_predictions = (
                np.concatenate(probabilistic_state['quantile_preds'], axis=0)
                if probabilistic_state['quantile_preds']
                else np.empty((0, self.args.pred_len, len(self.quantile_levels)), dtype=np.float32)
            )
            quantile_true = (
                np.concatenate(probabilistic_state['quantile_trues'], axis=0)
                if probabilistic_state['quantile_trues']
                else np.empty((0, self.args.pred_len, 1), dtype=np.float32)
            )
            np.save(folder_path + 'pred_quantiles.npy', quantile_predictions)
            np.save(folder_path + 'quantile_true.npy', quantile_true)
            np.save(folder_path + 'quantile_levels.npy', np.asarray(self.quantile_levels, dtype=np.float64))

            probabilistic_overall_metrics = probabilistic_state['overall'].metrics()
            for station_key, payload in probabilistic_state['stations'].items():
                probabilistic_station_metrics[station_key] = payload['acc'].metrics()
            for region, payload in probabilistic_state['regions'].items():
                probabilistic_region_metrics[region] = payload['acc'].metrics()
            station_equal = station_equal_probabilistic_metrics(
                probabilistic_state['stations'].values()
            )
            for forecast_step, accumulator in sorted(
                probabilistic_state['horizons'].items(), key=lambda item: int(item[0])
            ):
                probabilistic_horizon_metrics[int(forecast_step)] = accumulator.metrics()

            probabilistic_rows = [
                {
                    'scope': 'overall',
                    'task_name': self.args.task_name,
                    'split': 'test',
                    'metric_space': getattr(self.args, 'metric_space', 'power'),
                    **probabilistic_overall_metrics,
                }
            ]
            for station_key, payload in sorted(
                probabilistic_state['stations'].items(), key=lambda item: str(item[0])
            ):
                probabilistic_rows.append({
                    'scope': 'station',
                    'region': payload['region'],
                    'station_id': payload['station_id'],
                    'station_name': payload['station_name'],
                    'station_dir': payload['station_dir'],
                    'task_name': self.args.task_name,
                    'split': 'test',
                    'metric_space': getattr(self.args, 'metric_space', 'power'),
                    **payload['acc'].metrics(),
                })
            for region, payload in sorted(
                probabilistic_state['regions'].items(), key=lambda item: str(item[0])
            ):
                probabilistic_rows.append({
                    'scope': 'region',
                    'region': region,
                    'task_name': self.args.task_name,
                    'split': 'test',
                    'metric_space': getattr(self.args, 'metric_space', 'power'),
                    **payload['acc'].metrics(),
                })
            for forecast_step, metrics in sorted(probabilistic_horizon_metrics.items()):
                probabilistic_rows.append({
                    'scope': 'horizon',
                    'horizon': forecast_step,
                    'forecast_step': forecast_step,
                    'task_name': self.args.task_name,
                    'split': 'test',
                    'metric_space': getattr(self.args, 'metric_space', 'power'),
                    **metrics,
                })
            probabilistic_rows.append({
                'scope': 'station_equal',
                'aggregation': 'arithmetic_mean_over_stations',
                'task_name': self.args.task_name,
                'split': 'test',
                'metric_space': getattr(self.args, 'metric_space', 'power'),
                **station_equal,
            })
            pd.DataFrame(probabilistic_rows).to_csv(
                folder_path + 'probabilistic_metrics.csv', index=False, encoding='utf-8-sig'
            )
            horizon_rows = []
            for forecast_step, metrics in sorted(probabilistic_horizon_metrics.items()):
                horizon_rows.append({
                    'scope': 'horizon',
                    'horizon': forecast_step,
                    'forecast_step': forecast_step,
                    'task_name': self.args.task_name,
                    'split': 'test',
                    'metric_space': getattr(self.args, 'metric_space', 'power'),
                    **metrics,
                })
            pd.DataFrame(horizon_rows).to_csv(
                folder_path + 'probabilistic_metrics_by_horizon.csv',
                index=False,
                encoding='utf-8-sig',
            )
            with open(folder_path + 'probabilistic_config.json', 'w', encoding='utf-8') as handle:
                json.dump(
                    {
                        'quantile_levels': list(self.quantile_levels),
                        'quantile_parameterization': getattr(
                            self.args, 'quantile_parameterization', 'independent'
                        ),
                        'metric_space': getattr(self.args, 'metric_space', 'power'),
                        'point_forecast_quantile': 0.5,
                        'crps_method': 'trapezoid_quantile_interpolation_flat_tails_monotone_rearrangement',
                        'aggregation_note': 'overall is pooled over valid target points; station_equal is the arithmetic mean of per-station metrics',
                        'overall': probabilistic_overall_metrics,
                        'station_equal': station_equal,
                        'by_forecast_step': {
                            str(step): metrics
                            for step, metrics in probabilistic_horizon_metrics.items()
                        },
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=True,
                )
            print(f"Saved quantile predictions to: {folder_path + 'pred_quantiles.npy'}")
            print(f"Saved quantile truth to: {folder_path + 'quantile_true.npy'}")
            print(f"Saved quantile levels to: {folder_path + 'quantile_levels.npy'}")
            print(f"Saved probabilistic metrics to: {folder_path + 'probabilistic_metrics.csv'}")
            print(
                f"Saved probabilistic horizon metrics to: "
                f"{folder_path + 'probabilistic_metrics_by_horizon.csv'}"
            )
            print(f"Saved probabilistic config to: {folder_path + 'probabilistic_config.json'}")

        if station_series:
            station_rows = []
            for station_key, series_dict in station_series.items():
                region, station_id, station_name, station_dir = station_key
                station_pred = np.stack(series_dict['pred'], axis=0)
                station_true = np.stack(series_dict['true'], axis=0)
                mae_i, mse_i, rmse_i, mape_i, mspe_i, smape_i, r2_i = metric(station_pred, station_true)
                station_row = {
                    'region': region,
                    'station_id': station_id,
                    'station_name': station_name,
                    'station_dir': station_dir,
                    'mae': mae_i,
                    'mse': mse_i,
                    'rmse': rmse_i,
                    'mape': mape_i,
                    'mspe': mspe_i,
                    'smape': smape_i,
                    'r2': r2_i,
                }
                if station_key in probabilistic_station_metrics:
                    station_row.update(probabilistic_station_metrics[station_key])
                station_rows.append(station_row)

            station_metrics = pd.DataFrame(station_rows)
            station_metrics.to_csv(folder_path + 'station_metrics.csv', index=False, encoding='utf-8-sig')
            region_metrics = (
                station_metrics.groupby('region', as_index=False)[['mae', 'mse', 'rmse', 'mape', 'mspe', 'smape', 'r2']]
                .mean()
            )
            if probabilistic_region_metrics:
                for column in probabilistic_metric_names:
                    region_metrics[column] = [
                        probabilistic_region_metrics.get(region, {}).get(column, np.nan)
                        for region in region_metrics['region']
                    ]
            region_metrics.to_csv(folder_path + 'region_metrics.csv', index=False, encoding='utf-8-sig')
            station_metrics = station_metrics.sort_values(['region', 'station_id', 'station_name']).reset_index(drop=True)
            print('Station metrics summary:')
            if len(station_metrics) <= 20:
                print(station_metrics.to_string(index=False))
            else:
                print(station_metrics.head(20).to_string(index=False))
                print(f"... ({len(station_metrics)} stations total, full table saved to station_metrics.csv)")
            print('Region metrics summary:')
            print(region_metrics.to_string(index=False))
            print(f"Saved station metrics to: {folder_path + 'station_metrics.csv'}")
            print(f"Saved region metrics to: {folder_path + 'region_metrics.csv'}")

        if self.args.save_station_debug_arrays and station_debug_samples:
            debug_dir = os.path.join(folder_path, 'station_debug_samples')
            os.makedirs(debug_dir, exist_ok=True)
            debug_rows = []
            for station_key, sample_list in station_debug_samples.items():
                region, station_id, station_name, station_dir = station_key
                token = self._safe_station_token(station_id, station_name)
                out_path = os.path.join(debug_dir, f'{token}.npz')
                np.savez_compressed(
                    out_path,
                    region=region,
                    station_id=station_id,
                    station_name=station_name,
                    station_dir=station_dir,
                    history_target=np.stack([item['history_target'] for item in sample_list], axis=0),
                    future_true=np.stack([item['future_true'] for item in sample_list], axis=0),
                    future_pred=np.stack([item['future_pred'] for item in sample_list], axis=0),
                    history_time_features=np.stack([item['history_time_features'] for item in sample_list], axis=0),
                    future_time_features=np.stack([item['future_time_features'] for item in sample_list], axis=0),
                )
                debug_rows.append({
                    'region': region,
                    'station_id': station_id,
                    'station_name': station_name,
                    'station_dir': station_dir,
                    'saved_samples': len(sample_list),
                    'npz_path': out_path,
                })
            pd.DataFrame(debug_rows).sort_values(['region', 'station_id', 'station_name']).to_csv(
                os.path.join(debug_dir, 'index.csv'),
                index=False,
                encoding='utf-8-sig',
            )
            print(f"Saved station debug samples to: {debug_dir}")

        return
