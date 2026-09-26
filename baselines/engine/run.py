import argparse
import hashlib
import json
import os
import re
import torch
import torch.backends
from utils.print_args import print_args
from utils.model_audit import print_model_audit
import random
import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _seed_everything(seed):
    """Seed all RNGs used by the baseline entry point.

    This intentionally does not force cuDNN deterministic kernels.  The
    baseline protocol can request a seed for statistical replication without
    imposing the substantial performance/compatibility cost of bitwise
    determinism.
    """
    seed = int(seed)
    if seed <= 0:
        return
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

PV_TASK_SPECS = {
    '1min_intraday_1h': {
        'granularity': '1min',
        'horizon_hours': 1,
        'context_ratio': 3.0,
    },
    '5min_dayahead_24h': {
        'granularity': '5min',
        'horizon_hours': 24,
        'context_ratio': 3.0,
    },
    '15min_dayahead_24h': {
        'granularity': '15min',
        'horizon_hours': 24,
        'context_ratio': 3.0,
    },
    '1h_dayahead_24h': {
        'granularity': '1h',
        'horizon_hours': 24,
        'context_ratio': 3.0,
    },
    '1h_short_6h': {
        'granularity': '1h',
        'horizon_hours': 6,
        'context_ratio': 4.0,
    },
    '1h_week_168h': {
        'granularity': '1h',
        'horizon_hours': 168,
        'context_ratio': 2.0,
    },
}

PV_STEPS_PER_HOUR = {
    '1min': 60,
    '5min': 12,
    '15min': 4,
    '1h': 1,
}

NATIVE_EQUAL_CONTEXT_MODELS = {'Cross_Unet', 'FusionSFNoSpatial', 'FusionSFMasked'}

# A setting is used as one directory component below ``checkpoints`` and the
# result roots.  Linux limits a single component to NAME_MAX bytes (normally
# 255).  Keep headroom for future suffixes and make long q9 settings safe by
# replacing them with a readable prefix plus a hash of the complete setting.
SETTING_COMPONENT_MAX_BYTES = 240


def _filesystem_safe_setting(setting, max_bytes=SETTING_COMPONENT_MAX_BYTES):
    """Return a deterministic, filesystem-safe experiment directory name."""
    encoded = str(setting).encode('utf-8')
    if len(encoded) <= max_bytes and '/' not in str(setting) and '\x00' not in str(setting):
        return str(setting)

    original = str(setting)
    digest = hashlib.sha256(original.encode('utf-8')).hexdigest()[:16]
    prefix = re.sub(r'[^A-Za-z0-9_.-]+', '_', original)
    suffix = f'__h{digest}'
    prefix_budget = max(1, max_bytes - len(suffix.encode('utf-8')))
    prefix = prefix.encode('utf-8')[:prefix_budget].decode('utf-8', errors='ignore')
    prefix = prefix.rstrip('._-') or 'setting'
    safe = f'{prefix}{suffix}'

    # The prefix is ASCII after sanitization, but keep this guard in case the
    # implementation changes or a caller supplies an unusual max_bytes.
    if len(safe.encode('utf-8')) > max_bytes:
        safe = f'setting_{digest}'
    return safe


def _prepare_setting(args, setting, is_primary_process):
    """Shorten long path names while preserving the full name for provenance."""
    full_setting = str(setting)
    filesystem_setting = _filesystem_safe_setting(full_setting)
    if filesystem_setting == full_setting:
        return filesystem_setting

    # Keep backward compatibility for a legacy run that already has the full
    # directory.  New runs use the shortened name before the filesystem limit
    # can be hit; an existing checkpoint should remain addressable by its old
    # setting name.
    legacy_path = os.path.join(args.checkpoints, full_setting)
    try:
        if os.path.isdir(legacy_path):
            if is_primary_process:
                print(f'[run] reusing legacy filesystem_setting={full_setting}')
            return full_setting
    except OSError:
        # An overlong legacy component can itself raise on some filesystems;
        # in that case the deterministic shortened name is the only option.
        pass

    if is_primary_process:
        print(f'[run] full_setting={full_setting}')
        print(f'[run] filesystem_setting={filesystem_setting}')
        alias_path = os.path.join(args.checkpoints, 'setting_alias.json')
        try:
            os.makedirs(args.checkpoints, exist_ok=True)
            with open(alias_path, 'w', encoding='utf-8') as alias_file:
                json.dump(
                    {
                        'full_setting': full_setting,
                        'filesystem_setting': filesystem_setting,
                        'full_setting_sha256': hashlib.sha256(
                            full_setting.encode('utf-8')
                        ).hexdigest(),
                    },
                    alias_file,
                    indent=2,
                    sort_keys=True,
                )
                alias_file.write('\n')
        except OSError as exc:
            # The shortened setting still prevents the original checkpoint
            # failure; provenance logging should not make the run fail.
            print(f'[run] warning: could not write setting_alias.json: {exc}')
    return filesystem_setting

def resolve_pv_task_lengths(task_spec_name):
    task_spec = PV_TASK_SPECS.get(task_spec_name)
    if task_spec is None:
        return None

    granularity = task_spec['granularity']
    steps_per_hour = PV_STEPS_PER_HOUR[granularity]
    horizon_hours = float(task_spec['horizon_hours'])
    context_hours = horizon_hours * float(task_spec['context_ratio'])

    pred_len = int(horizon_hours * steps_per_hour)
    seq_len = int(context_hours * steps_per_hour)
    label_len = max(1, pred_len)
    return seq_len, label_len, pred_len

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--manifest_path', '--station-manifest', '--station_manifest',
                        dest='manifest_path', type=str,
                        default='./downloads/organized_by_station/station_manifest.csv',
                        help='path to the station manifest csv for pv_multires data')
    parser.add_argument('--station_data_root', type=str,
                        default='./downloads/organized_by_station',
                        help='root directory that contains per-station folders for pv_multires data')
    parser.add_argument('--task_spec_name', type=str, default='',
                        help='named task specification for pv_multires, e.g. 15min_dayahead_24h or 1h_dayahead_24h')
    parser.add_argument('--region_filter', type=str, default='',
                        help='comma-separated region list for pv_multires, e.g. 安徽,广西')
    parser.add_argument('--source_region_filter', type=str, default='',
                        help='comma-separated source_region list for pv_multires, e.g. 安徽,广西')
    parser.add_argument('--granularity_filter', type=str, default='',
                        help='comma-separated granularity list for pv_multires, e.g. 5min,15min,1h')
    parser.add_argument('--station_dir_filter', type=str, default='',
                        help='comma-separated station_dir list for pv_multires')
    parser.add_argument('--max_stations', type=int, default=0,
                        help='limit the number of stations loaded for pv_multires, 0 means no limit')
    parser.add_argument('--disable_test_visuals', action='store_true', default=False,
                        help='disable per-batch test visualization pdfs')
    parser.add_argument('--save_prediction_points', action='store_true', default=False,
                        help='save per-horizon prediction points for pv_multires test output')
    parser.add_argument('--save_station_debug_arrays', action='store_true', default=False,
                        help='save per-station debug arrays during test for inspecting history/true/pred windows')
    parser.add_argument('--station_debug_max_samples', type=int, default=32,
                        help='maximum number of debug samples to save per station when --save_station_debug_arrays is enabled')
    parser.add_argument('--data_file_name', type=str, default='',
                        help='override pv file name for pv_multires, e.g. pv.csv or pv_weather_merged.csv')
    parser.add_argument('--time_col_override', type=str, default='',
                        help='override time column for pv_multires')
    parser.add_argument('--target_col_override', type=str, default='',
                        help='override target column for pv_multires')
    parser.add_argument('--target_normalization', type=str, default='none',
                        choices=['none', 'capacity_factor'],
                        help='for pv_multires: normalize target by per-station capacity proxy before standard scaling')
    parser.add_argument('--metric_space', type=str, default='power',
                        choices=['power', 'normalized'],
                        help='for pv_multires test metrics: power multiplies normalized targets back by capacity; normalized keeps the target-normalized space')
    parser.add_argument('--capacity_proxy_quantile', type=float, default=99.5,
                        help='for pv_multires capacity_factor mode: fallback proxy capacity quantile in percent')
    parser.add_argument('--feature_cols', type=str, default='',
                        help='comma-separated feature columns for pv_multires')
    parser.add_argument('--pvfm_v6_profile', type=str, default='',
                        help='Frozen JSON architecture profile for scratch-only PVFMV6FullShot')
    parser.add_argument('--nwp_mode', type=str, default='none',
                        help='pv_multires NWP fusion mode: none, feature_concat, time_concat, or a direct model adapter')
    parser.add_argument('--dag_source_root', type=str,
                        default='models/external_dag',
                        help='root of the vendored, read-only DAG benchmark sources')
    parser.add_argument('--dag_alpha', type=float, default=0.7,
                        help='DAG temporal/covariate output mixing weight')
    parser.add_argument('--dag_beta', type=float, default=0.1,
                        help='DAG causality auxiliary-loss weight')
    parser.add_argument('--gcgnet_rank', type=int, default=4,
                        help='GCGNet low-rank graph sparsifier rank')
    parser.add_argument('--base_model_name', type=str, default='',
                        help='when using a fusion wrapper model, the underlying baseline model name, e.g. DLinear or PatchTST')
    parser.add_argument('--resample_to_1h', action='store_true', default=False,
                        help='for pv_multires: dynamically resample power to hourly before modeling')
    parser.add_argument('--hourly_resample_mode', type=str, default='mean', choices=['mean', 'exact', 'nearest'],
                        help='for pv_multires: how to build 1h targets from higher-frequency data; mean=hourly average, exact=sample exact HH:00 values only, nearest=sample nearest point to HH:00 within tolerance')
    parser.add_argument('--hourly_resample_tolerance_minutes', type=float, default=0.0,
                        help='for pv_multires hourly_resample_mode=nearest: max absolute distance to HH:00 in minutes')
    parser.add_argument('--weather_file_name', type=str, default='',
                        help='deprecated alias for future_covariate_file_name in pv_multires')
    parser.add_argument('--weather_time_col', type=str, default='datetime',
                        help='deprecated alias for future_covariate_time_col in pv_multires')
    parser.add_argument('--history_covariate_file_name', type=str, default='',
                        help='for pv_multires: historical covariate file aligned to the encoder timeline, e.g. ERA5 history')
    parser.add_argument('--history_covariate_time_col', type=str, default='datetime',
                        help='time column name for the historical covariate file')
    parser.add_argument('--history_covariate_cols', type=str, default='',
                        help='comma-separated historical covariate columns, e.g. temperature_2m,cloud_cover,shortwave_radiation')
    parser.add_argument('--future_covariate_file_name', type=str, default='',
                        help='for pv_multires: future-known covariate file aligned to the decoder timeline, e.g. NWP forecast')
    parser.add_argument('--future_covariate_time_col', type=str, default='datetime',
                        help='time column name for the future covariate file')
    parser.add_argument('--future_covariate_cols', type=str, default='',
                        help='comma-separated future-known covariate columns, typically NWP variables')
    parser.add_argument('--future_covariate_min_datetime', type=str, default='',
                        help='optional lower-bound timestamp for future covariates; rows earlier than this are dropped before merging')
    parser.add_argument('--data_min_datetime', type=str, default='',
                        help='optional lower-bound timestamp for pv_multires model samples; rows earlier than this are dropped before splitting')
    parser.add_argument('--future_covariate_align_to_hour', type=str, default='none',
                        choices=['none', 'floor', 'ceil', 'round'],
                        help='optional timestamp alignment applied to future covariates before merging, useful for half-hour timezone stations')
    parser.add_argument('--sample_nan_ratio_threshold', type=float, default=0.0,
                        help='for pv_multires: maximum allowed NaN ratio inside a candidate sample window')
    parser.add_argument('--min_past_target_valid_ratio', type=float, default=1.0,
                        help='for pv_multires: minimum valid target ratio in the encoder/past window')
    parser.add_argument('--min_future_target_valid_ratio', type=float, default=1.0,
                        help='for pv_multires: minimum valid target ratio in the prediction/future window')
    parser.add_argument('--min_history_covariate_valid_ratio', type=float, default=1.0,
                        help='for pv_multires: minimum valid covariate ratio in the encoder/past window')
    parser.add_argument('--min_future_covariate_valid_ratio', type=float, default=1.0,
                        help='for pv_multires: minimum valid covariate ratio in the prediction/future window')
    parser.add_argument('--sample_index_csv', type=str, default='',
                        help='optional canonical sample-window whitelist CSV for pv_multires alignment')
    parser.add_argument('--eval_origin_manifest', '--eval-origin-manifest',
                        dest='eval_origin_manifest', type=str, default='',
                        help='canonical evaluation-origin manifest; aliases --sample_index_csv in strict mode')
    parser.add_argument('--data_qc_version', '--data-qc-version',
                        dest='data_qc_version', type=str, default='original-aligned-qc',
                        help='version label for the data/QC protocol')
    parser.add_argument('--data_qc_root', '--data-qc-root',
                        dest='data_qc_root', type=str, default='',
                        help='versioned virtual QC release directory')
    parser.add_argument('--audit_mode', '--audit-mode',
                        dest='audit_mode', choices=['original', 'clean', 'paired'],
                        default='original',
                        help='original source, virtual-clean overlay, or paired clean-origin audit')
    parser.add_argument('--strict_future_target_nan', action='store_true', default=False,
                        help='for pv_multires: discard a sample when the prediction horizon target contains NaN')
    parser.add_argument('--seq_nan_interp_threshold', type=float, default=-1.0,
                        help='for pv_multires: if seq_len NaN ratio is within this threshold, linearly interpolate the history window before training; negative disables it')
    parser.add_argument('--seq_nan_fill_method', type=str, default='linear',
                        help='for pv_multires: interpolation method for the seq_len history window, currently only linear is supported')
    parser.add_argument('--use_masked_future_loss', action='store_true', default=False,
                        help='for pv_multires: allow NaNs in future targets and compute loss only on valid target positions')
    parser.add_argument('--min_future_valid_ratio', type=float, default=1.0,
                        help='for pv_multires: minimum valid ratio required inside pred_len target horizon when masked future loss is enabled')
    parser.add_argument('--enable_station_cache', action='store_true', default=False,
                        help='for pv_multires: enable persistent per-station preprocessing cache')
    parser.add_argument('--station_cache_dir', type=str, default='./cache/pv_multires_station_cache',
                        help='for pv_multires: directory for persistent per-station preprocessing cache')
    parser.add_argument('--refresh_station_cache', action='store_true', default=False,
                        help='for pv_multires: ignore existing station cache files and rebuild them')
    parser.add_argument('--enable_physical_sample_filter', action='store_true', default=False,
                        help='enable sample-level physical quality filtering using target and solar-radiation consistency')
    parser.add_argument('--physical_filter_history_solar_col', type=str, default='shortwave_radiation',
                        help='historical covariate column used for sample-level PV-vs-solar consistency checks')
    parser.add_argument('--physical_filter_future_solar_col', type=str, default='shortwave_radiation',
                        help='future covariate column used for sample-level daylight screening')
    parser.add_argument('--physical_filter_daylight_threshold', type=float, default=100.0,
                        help='radiation threshold that defines daylight for sample-level physical filtering')
    parser.add_argument('--physical_filter_target_zero_threshold', type=float, default=0.1,
                        help='raw target threshold below which a point is treated as near-zero during daylight filtering')
    parser.add_argument('--physical_filter_max_future_day_zero_ratio', type=float, default=1.0,
                        help='maximum allowed near-zero ratio inside daylight future points; 1.0 disables the filter')
    parser.add_argument('--physical_filter_max_history_day_zero_ratio', type=float, default=1.0,
                        help='maximum allowed near-zero ratio inside daylight history points; 1.0 disables the filter')
    parser.add_argument('--physical_filter_min_history_day_corr', type=float, default=-2.0,
                        help='minimum allowed history-window PV-vs-solar correlation over daylight points; values below -1 disable the filter')
    parser.add_argument('--physical_filter_min_future_daylight_points', type=int, default=0,
                        help='minimum number of daylight future points required before future-window physical filtering is evaluated')
    parser.add_argument('--fusionsf_ctx_source', type=str, default='history_cov',
                        help='FusionSFBaseline context source: history_cov or zeros')
    parser.add_argument('--fusionsf_dim', type=int, default=64,
                        help='hidden dimension for FusionSFBaseline')
    parser.add_argument('--fusionsf_depth', type=int, default=4,
                        help='transformer depth for FusionSFBaseline')
    parser.add_argument('--fusionsf_heads', type=int, default=4,
                        help='attention heads for FusionSFBaseline')
    parser.add_argument('--fusionsf_dropout', type=float, default=0.3,
                        help='dropout for FusionSFBaseline')
    parser.add_argument('--fusionsf_ff_mult', type=int, default=4,
                        help='feed-forward expansion multiplier for FusionSFBaseline')
    parser.add_argument('--fusionsf_dim_head', type=int, default=64,
                        help='FusionSFNoSpatial attention head width')
    parser.add_argument('--fusionsf_decoder_dim', type=int, default=128,
                        help='FusionSFNoSpatial temporal decoder width')
    parser.add_argument('--fusionsf_decoder_depth', type=int, default=4,
                        help='FusionSFNoSpatial temporal decoder depth')
    parser.add_argument('--fusionsf_decoder_heads', type=int, default=6,
                        help='FusionSFNoSpatial temporal decoder attention heads')
    parser.add_argument('--fusionsf_decoder_dim_head', type=int, default=128,
                        help='FusionSFNoSpatial temporal decoder head width')
    parser.add_argument('--fusionsf_num_mlp_heads', type=int, default=9,
                        help='FusionSFNoSpatial official multi-head output count '
                             '(official release configs use 9; a single head makes the '
                             'trailing ReLU a single point of failure)')
    fusionsf_vq_group = parser.add_mutually_exclusive_group()
    fusionsf_vq_group.add_argument('--disable_fusionsf_vq_in_ts', action='store_false',
                                  dest='fusionsf_vq_in_ts',
                                  help='disable FusionSF PV-branch VQ (official release default)')
    fusionsf_vq_group.add_argument('--enable_fusionsf_vq_in_ts', action='store_true',
                                  dest='fusionsf_vq_in_ts',
                                  help='explicitly enable FusionSF PV-branch VQ for a legacy run or ablation')
    parser.set_defaults(fusionsf_vq_in_ts=False)
    parser.add_argument('--fusionsf_nospatial_dim', type=int, default=64,
                        help='official FusionSFNoSpatial encoder width')
    parser.add_argument('--fusionsf_nospatial_depth', type=int, default=4,
                        help='official FusionSFNoSpatial encoder depth')
    parser.add_argument('--fusionsf_nospatial_heads', type=int, default=4,
                        help='official FusionSFNoSpatial encoder attention heads')
    parser.add_argument('--fusionsf_nospatial_dim_head', type=int, default=64,
                        help='official FusionSFNoSpatial encoder head width')
    parser.add_argument('--fusionsf_nospatial_mlp_ratio', type=int, default=4,
                        help='official FusionSFNoSpatial feed-forward expansion')
    parser.add_argument('--fusionsf_nospatial_dropout', type=float, default=0.3,
                        help='official FusionSFNoSpatial dropout')
    parser.add_argument('--fusionsf_nospatial_decoder_dim', type=int, default=128,
                        help='official FusionSFNoSpatial decoder width')
    parser.add_argument('--fusionsf_nospatial_decoder_depth', type=int, default=4,
                        help='official FusionSFNoSpatial decoder depth')
    parser.add_argument('--fusionsf_nospatial_decoder_heads', type=int, default=6,
                        help='official FusionSFNoSpatial decoder attention heads')
    parser.add_argument('--fusionsf_nospatial_decoder_dim_head', type=int, default=128,
                        help='official FusionSFNoSpatial decoder head width')
    parser.add_argument('--fusionsf_nospatial_num_mlp_heads', type=int, default=9,
                        help='official FusionSFNoSpatial output head count '
                             '(official release configs use 9, not the class-signature default of 1)')
    parser.add_argument('--fusionsf_source_root', type=str, default='/tmp/FusionSF-upstream',
                        help='upstream FusionSF checkout used by the masked official adapter')
    parser.add_argument('--tide_hidden_size', type=int, default=512,
                        help='TiDE hidden width; ETTh1 short-horizon release setting is 512')
    parser.add_argument('--tide_num_layers', type=int, default=2,
                        help='number of TiDE residual encoder/decoder widths')
    parser.add_argument('--tide_time_encoder_hidden', type=int, default=64,
                        help='hidden width of the original TiDE time-feature encoder')
    parser.add_argument('--tide_time_encoder_output', type=int, default=4,
                        help='output width of the original TiDE time-feature encoder')
    parser.add_argument('--tide_decoder_output_dim', type=int, default=32,
                        help='per-horizon latent decoder width')
    parser.add_argument('--tide_final_decoder_hidden', type=int, default=16,
                        help='hidden width of the original TiDE final decoder')
    parser.add_argument('--tide_cat_emb_size', type=int, default=4,
                        help='embedding width for TiDE categorical covariates')
    parser.add_argument('--tide_transform', type=int, default=1, choices=[0, 1],
                        help='retain the original optional TiDE reversible window transform')
    parser.add_argument('--tide_layer_norm', type=int, default=1, choices=[0, 1],
                        help='enable LayerNorm in original TiDE residual blocks')
    parser.add_argument('--tide_dropout_rate', type=float, default=0.5,
                        help='TiDE residual-block dropout rate')
    parser.add_argument('--use_feature_meta', type=int, default=1,
                        help='PVTC_V2: preserve original feature metadata tokens')
    parser.add_argument('--feature_meta_scale', type=float, default=0.1,
                        help='PVTC_V2: feature metadata token scale')
    parser.add_argument('--use_region_proto', type=int, default=0,
                        help='PVTC_V2: original repro launcher default is 0; keep off for target-site full-shot')
    parser.add_argument('--use_hist_group_tokens', type=int, default=1,
                        help='PVTC_V2: preserve historical/future group-token path')
    parser.add_argument('--use_relation_attention', type=int, default=0,
                        help='PVTC_V2: relation-attention ablation, disabled in original repro launcher')
    parser.add_argument('--use_pure_featmeta_baseline', type=int, default=0,
                        help='PVTC_V2: alternate feature-metadata-only ablation')
    parser.add_argument('--use_fixed_group_pool', type=int, default=0,
                        help='PVTC_V2: use fixed rather than learned group pooling')
    parser.add_argument('--region_proto_dim', type=int, default=9,
                        help='PVTC_V2 region prototype width when that optional branch is enabled')
    parser.add_argument('--v2_use_site_ctx', type=int, default=1,
                        help='PVTC_V2: retain original site-context conditioning')
    parser.add_argument('--v2_use_solar_tokens', type=int, default=1,
                        help='PVTC_V2: retain original solar-geometry tokens')
    parser.add_argument('--v2_hist_group_pool', type=int, default=1,
                        help='PVTC_V2: retain original historical group pooling')
    parser.add_argument('--v2_future_group_pool', type=int, default=1,
                        help='PVTC_V2: retain original future-NWP group pooling')

    # PVTC_Ablation: every default delegates to frozen PVTC_V2 exactly.
    parser.add_argument('--ablation_solar_dim', type=int, default=4, choices=[3, 4],
                        help='PVTC ablation C: 3 drops zenith; default 4 is frozen control')
    parser.add_argument('--ablation_use_hist_solar', type=int, default=1, choices=[0, 1],
                        help='PVTC ablation B: retain/drop history solar tokens')
    parser.add_argument('--ablation_use_future_solar', type=int, default=1, choices=[0, 1],
                        help='PVTC ablation: retain/drop future solar tokens')
    parser.add_argument('--ablation_use_site_ctx', type=int, default=1, choices=[0, 1],
                        help='PVTC ablation: retain/drop the entire static site context')
    parser.add_argument('--ablation_use_hist_group_pool', type=int, default=1, choices=[0, 1],
                        help='PVTC ablation: retain/drop historical feature-group pooling')
    parser.add_argument('--ablation_use_future_group_pool', type=int, default=1, choices=[0, 1],
                        help='PVTC ablation: retain/drop future-NWP feature-group pooling')
    parser.add_argument('--ablation_use_post_hist_attn', type=int, default=1, choices=[0, 1],
                        help='PVTC ablation A: retain/drop post-encoder PV-to-history attention')
    parser.add_argument('--ablation_site_ctx_inject_at', type=str, default='phfo',
                        help='PVTC ablation D: subset of p,h,f,o for site-context injection')
    parser.add_argument('--ablation_learnable_group_queries', type=int, default=0, choices=[0, 1],
                        help='PVTC ablation E: replace PV-derived group queries with learnable queries')
    parser.add_argument('--ablation_future_encoder', type=int, default=0, choices=[0, 1],
                        help='PVTC ablation F: add future-NWP encoder')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument(
        '--quantiles', '--quantile_levels', '--probabilistic_quantiles',
        dest='quantiles', type=str, default='',
        help='optional comma-separated quantile levels; enables probabilistic forecasting',
    )
    parser.add_argument(
        '--quantile_parameterization', type=str,
        choices=['independent', 'noncrossing'], default='independent',
        help='quantile head parameterization when --quantiles is enabled',
    )
    parser.add_argument(
        '--quantile_increment_scale', type=float, default=0.1,
        help='positive increment scale for noncrossing quantile heads',
    )

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--tv_dt', type=int, default=0, help='whether to use time variant dt for MambaSL')
    parser.add_argument('--tv_B', type=int, default=0, help='whether to use time variant B for MambaSL')
    parser.add_argument('--tv_C', type=int, default=0, help='whether to use time variant C for MambaSL')
    parser.add_argument('--use_D', type=int, default=0, help='whether to use D for MambaSL')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--useweather', type=bool, default=True, help='Cross_Unet: use weather covariates')
    parser.add_argument('--usenonlinearproject', type=bool, default=False, help='Cross_Unet: use nonlinear P-corr projection')
    parser.add_argument('--usebottle', type=bool, default=True, help='Cross_Unet: use bottleneck')
    parser.add_argument('--convmerge', type=bool, default=False, help='Cross_Unet: use convolutional segment merge')
    parser.add_argument('--swichchannel', type=bool, default=False, help='Cross_Unet: switch channel correlation direction')
    parser.add_argument('--twofilter', type=bool, default=True, help='Cross_Unet: use two-stage attention filter')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=0,
                        help='optional batch size for validation/test; 0 reuses --batch_size')
    parser.add_argument('--eval_sample_stride', '--eval-sample-stride',
                        dest='eval_sample_stride', type=int, default=6,
                        help='for pv_multires test evaluation, keep one start every N steps; standard 1h benchmark=6')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--use_accelerate', action='store_true', default=False,
                        help='use Hugging Face Accelerate for device placement, mixed precision, and distributed training')
    parser.add_argument('--ddp_find_unused_parameters', action='store_true', default=False,
                        help='allow unused parameters in Accelerate DDP; off by default for lower overhead')

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu (default: on)')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='disable gpu (force cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', action='store_true', default=False,
                        help='enable dtw metric (time consuming; default: off)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2021,
                        help="Randomization seed for Python/NumPy/Torch/DataLoader workers; 0 disables explicit seeding")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--patch_stride', type=int, default=8, help='patch stride for patch-based models')

    # GPT4TS / gpt4ts
    parser.add_argument('--gpt_layers', type=int, default=3, help='number of GPT2 layers used by gpt4ts')
    parser.add_argument('--pretrain', type=int, default=1, help='gpt4ts: load pretrained GPT2 when 1')
    parser.add_argument('--is_gpt', type=int, default=1, help='gpt4ts: enable GPT backbone when 1')
    parser.add_argument('--freeze', type=int, default=1, help='gpt4ts: freeze most pretrained GPT2 weights when 1')
    parser.add_argument('--stride', type=int, default=8, help='stride for patch embedding')
    parser.add_argument('--padding', type=int, default=8, help='right padding for patch embedding')

    # TimeLLM / TimeVLM
    parser.add_argument('--llm_model', type=str, default='GPT2',
                        choices=['LLAMA', 'GPT2', 'BERT'],
                        help='language model backbone for TimeLLM')
    parser.add_argument('--llm_dim', type=int, default=768,
                        help='hidden size of the selected LLM; GPT2/BERT use 768, LLaMA-7B uses 4096')
    parser.add_argument('--llm_layers', type=int, default=1, help='number of LLM layers to use')
    parser.add_argument('--prompt_domain', type=int, default=0, help='use domain prompt content when supported')
    parser.add_argument('--content', type=str,
                        default='Photovoltaic power forecasting with historical PV generation and weather covariates.',
                        help='prompt content used by TimeLLM/TimeVLM')

    parser.add_argument('--finetune_vlm', action='store_true', default=False,
                        help='fine-tune VLM encoder parameters in TimeVLM')
    parser.add_argument('--image_size', type=int, default=224, help='TimeVLM generated image size')
    parser.add_argument('--periodicity', type=int, default=24, help='main periodicity for TimeVLM image conversion')
    parser.add_argument('--three_channel_image', type=int, default=1, choices=[0, 1],
                        help='TimeVLM: generate three-channel images when 1')
    parser.add_argument('--learnable_image', type=int, default=1, choices=[0, 1],
                        help='TimeVLM: use learnable image conversion when 1')
    parser.add_argument('--vlm_type', type=str, default='CLIP',
                        choices=['CLIP', 'clip', 'BLIP2', 'blip2', 'ViLT', 'vilt', 'custom'],
                        help='VLM backbone type for TimeVLM')
    parser.add_argument('--patch_memory_size', type=int, default=100, help='TimeVLM patch memory bank size')
    parser.add_argument('--use_mem_gate', type=int, default=0, choices=[0, 1],
                        help='TimeVLM: use memory fusion gate when 1')
    parser.add_argument('--use_cross_attention', type=int, default=1, choices=[0, 1],
                        help='TimeVLM: use cross-modal attention when 1')
    parser.add_argument('--interpolation', type=str, default='bilinear',
                        help='TimeVLM image interpolation method')
    parser.add_argument('--norm_const', type=float, default=0.4,
                        help='TimeVLM normalization constant')
    parser.add_argument('--w_out_visual', type=int, default=0, choices=[0, 1],
                        help='TimeVLM ablation: disable visual branch when 1')
    parser.add_argument('--w_out_text', type=int, default=0, choices=[0, 1],
                        help='TimeVLM ablation: disable text branch when 1')
    parser.add_argument('--w_out_query', type=int, default=0, choices=[0, 1],
                        help='TimeVLM ablation: disable query branch when 1')
    parser.add_argument('--save_images', type=int, default=0, choices=[0, 1],
                        help='TimeVLM: save generated images when 1')
    parser.add_argument('--visualize_embeddings', type=int, default=0, choices=[0, 1],
                        help='TimeVLM: save embedding visualizations when 1')
    parser.add_argument('--memory_bank_size', type=int, default=20,
                        help='TimeVLM general memory bank size')
    parser.add_argument('--align_const', type=float, default=0.4,
                        help='TimeVLM alignment constant')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')

    args = parser.parse_args()
    _seed_everything(args.seed)
    if args.eval_sample_stride < 1:
        raise ValueError('--eval_sample_stride must be a positive integer')
    if args.eval_origin_manifest:
        if not os.path.exists(args.eval_origin_manifest):
            raise FileNotFoundError(
                f'--eval-origin-manifest does not exist: {args.eval_origin_manifest}'
            )
        if args.sample_index_csv:
            expected = os.path.abspath(args.sample_index_csv)
            supplied = os.path.abspath(args.eval_origin_manifest)
            if expected != supplied:
                raise ValueError(
                    '--sample-index-csv and --eval-origin-manifest must refer to the same file'
                )
        args.sample_index_csv = args.eval_origin_manifest
    if args.audit_mode in {'clean', 'paired'}:
        if args.data != 'pv_multires':
            raise ValueError('--audit-mode clean/paired is only supported for --data pv_multires')
        if not args.data_qc_root:
            raise ValueError('--audit-mode clean/paired requires --data-qc-root')
        if not args.resample_to_1h:
            raise ValueError('--audit-mode clean/paired requires --resample_to_1h')
    if args.audit_mode == 'paired' and not args.eval_origin_manifest:
        raise ValueError('--audit-mode paired requires --eval-origin-manifest')
    is_primary_process = os.environ.get('RANK', '0') == '0'
    if args.quantiles:
        from models.ProbabilisticOutputWrapper import parse_quantile_levels

        args.quantiles = parse_quantile_levels(args.quantiles)
        args.quantile_tag = 'q' + str(len(args.quantiles)) + '_' + '_'.join(
            f'{level:.3f}'.replace('.', '') for level in args.quantiles
        )
    else:
        args.quantiles = ()
        args.quantile_tag = ''
    if args.quantile_increment_scale <= 0.0:
        raise ValueError('--quantile_increment_scale must be positive')
    if args.data == 'pv_multires' and args.task_spec_name:
        resolved_lengths = resolve_pv_task_lengths(args.task_spec_name)
        if resolved_lengths is None:
            raise ValueError(
                f"Unknown pv_multires task_spec_name: {args.task_spec_name}. "
                f"Available specs: {', '.join(PV_TASK_SPECS.keys())}"
            )
        args.seq_len, args.label_len, args.pred_len = resolved_lengths
        if args.model in NATIVE_EQUAL_CONTEXT_MODELS:
            native_equal_context_tasks = {'1h_short_6h', '1h_dayahead_24h', '1h_week_168h'}
            if args.task_spec_name not in native_equal_context_tasks:
                raise ValueError(
                    f"{args.model} native baseline supports only equal-length 1h tasks: "
                    "1h_short_6h, 1h_dayahead_24h, or 1h_week_168h"
                )
            args.seq_len = args.pred_len
            args.label_len = args.pred_len

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        if is_primary_process:
            print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        if is_primary_process:
            print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if is_primary_process:
        print('Args in experiment:')
        print_args(args)


    if args.task_name == 'long_term_forecast':
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'zero_shot_forecast':
        from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
        Exp = Exp_Zero_Shot_Forecast
    else:
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            if is_primary_process:
                print_model_audit(exp.model, args, stage='post_init')
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            # Override setting for specific model to ensure proper checkpoint naming and logging
            if args.model == 'MambaSingleLayer' and args.task_name == 'classification':
                setting = f'{args.task_name}_CLS_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                        + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                        + f'_tvdt{int(args.tv_dt)}_tvB{int(args.tv_B)}_tvC{int(args.tv_C)}_useD{int(args.use_D)}_{args.des}_{ii}'

            if args.quantile_tag:
                setting = f'{setting}_{args.quantile_tag}'

            setting = _prepare_setting(args, setting, is_primary_process)

            if is_primary_process:
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            if is_primary_process:
                # A second audit catches lazy layers (for example, a lazy
                # input projection) after the first real training batch.
                print_model_audit(exp.model, args, stage='post_train')

            if is_primary_process:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.use_gpu:
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        if is_primary_process:
            print_model_audit(exp.model, args, stage='post_init')
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        # Override setting for specific model to ensure proper checkpoint naming and logging
        if args.model == 'MambaSingleLayer' and args.task_name == 'classification':
            setting = f'{args.task_name}_CLS_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                    + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                    + f'_tvdt{args.tv_dt}_tvB{args.tv_B}_tvC{args.tv_C}_useD{int(args.use_D)}_{args.des}_{ii}'

        if args.quantile_tag:
            setting = f'{setting}_{args.quantile_tag}'

        setting = _prepare_setting(args, setting, is_primary_process)

        if is_primary_process:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if is_primary_process:
            print_model_audit(exp.model, args, stage='post_test')
        if args.use_gpu:
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
