"""Runtime model and parameter audit helpers.

The command-line namespace is useful for reproducing a run, but it is not
always a faithful description of a wrapped model.  For example,
``NWPFusionBaseline`` changes the input width of its base model and the
quantile wrapper adds another trainable head.  This module reports both the
effective architecture and counts from the instantiated ``nn.Module``.
"""

from __future__ import annotations

import json

import torch
from torch import nn


def _unwrap_parallel(model):
    """Return the underlying module for DataParallel/DDP-like wrappers."""
    current = model
    seen = set()
    while isinstance(current, nn.Module) and hasattr(current, "module"):
        if id(current) in seen:
            break
        seen.add(id(current))
        child = getattr(current, "module")
        if not isinstance(child, nn.Module):
            break
        current = child
    return current


def _safe_numel(value):
    """Return a tensor's size, or ``None`` for an uninitialized lazy tensor."""
    try:
        return int(value.numel())
    except (RuntimeError, TypeError, ValueError):
        return None


def _count_parameters(module):
    total = 0
    trainable = 0
    uninitialized = 0
    for parameter in module.parameters():
        size = _safe_numel(parameter)
        if size is None:
            uninitialized += 1
            continue
        total += size
        if parameter.requires_grad:
            trainable += size
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
        "uninitialized": uninitialized,
    }


def _count_state_dict_tensors(module):
    """Count checkpoint tensors, including buffers, as used by old metadata."""
    total = 0
    uninitialized = 0
    try:
        state = module.state_dict()
    except (RuntimeError, TypeError, ValueError):
        return {"total": None, "uninitialized": 1}
    for value in state.values():
        if not torch.is_tensor(value):
            continue
        size = _safe_numel(value)
        if size is None:
            uninitialized += 1
        else:
            total += size
    return {"total": total, "uninitialized": uninitialized}


def _get_attr_path(root, path):
    current = root
    for name in path.split("."):
        if not hasattr(current, name):
            return None
        current = getattr(current, name)
    return current


def _first_int_attr(root, names):
    for module in root.modules():
        for name in names:
            value = getattr(module, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def _first_linear_width(root, output=True):
    for module in root.modules():
        if isinstance(module, nn.Linear):
            return int(module.out_features if output else module.in_features)
        if isinstance(module, nn.Conv1d):
            return int(module.out_channels if output else module.in_channels)
    return None


def _infer_d_ff(root):
    for module in root.modules():
        conv1 = getattr(module, "conv1", None)
        if isinstance(conv1, nn.Conv1d):
            return int(conv1.out_channels)
        value = getattr(module, "d_ff", None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _infer_d_model(root):
    # These paths cover the standard Time-Series-Library models and avoid
    # mistaking a projection head's width for the model width.
    for path in (
        "patch_embedding.value_embedding",
        "enc_value_embedding.value_embedding",
        "enc_embedding.value_embedding",
    ):
        module = _get_attr_path(root, path)
        if isinstance(module, nn.Linear):
            return int(module.out_features)
        if isinstance(module, nn.Conv1d):
            return int(module.out_channels)
    value = getattr(root, "d_model", None)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _find_base_model(root, requested_model):
    """Find the actual model whose constructor implements ``requested_model``."""
    current = root
    # QuantileOutputWrapper and NWPFusionBaseline both use ``base_model``;
    # follow only those known wrapper contracts so ordinary model internals
    # are not accidentally traversed.
    for _ in range(4):
        class_name = current.__class__.__name__
        if class_name == "QuantileOutputWrapper" and isinstance(
            getattr(current, "base_model", None), nn.Module
        ):
            current = current.base_model
            continue
        if class_name == "Model" and hasattr(current, "base_model_name") and isinstance(
            getattr(current, "base_model", None), nn.Module
        ):
            current = current.base_model
            continue
        break

    # Cross_Unet is an adapter around the imported UNet_CF implementation.
    if requested_model == "Cross_Unet" and isinstance(
        getattr(current, "core", None), nn.Module
    ):
        current = current.core
    return current


def _architecture_name(args):
    requested = str(getattr(args, "model", ""))
    if requested == "NWPFusionBaseline":
        return str(getattr(args, "base_model_name", "NWPFusionBaseline"))
    return requested


def _standard_fields(base_model, args, name):
    configured = {
        "e_layers": getattr(args, "e_layers", None),
        "d_model": getattr(args, "d_model", None),
        "d_ff": getattr(args, "d_ff", None),
        "n_heads": getattr(args, "n_heads", None),
        "dropout": getattr(args, "dropout", None),
    }
    inferred = {}

    if name in {"PatchTST", "iTransformer", "Crossformer", "TimeMixer"}:
        inferred["d_model"] = _infer_d_model(base_model)
        inferred["d_ff"] = _infer_d_ff(base_model)
        inferred["n_heads"] = _first_int_attr(base_model, ("n_heads",))
    if name in {"PatchTST", "iTransformer"}:
        layers = getattr(getattr(base_model, "encoder", None), "attn_layers", None)
        if layers is not None:
            inferred["e_layers"] = len(layers)
    elif name == "TimeMixer":
        value = getattr(base_model, "layer", None)
        if isinstance(value, int):
            inferred["e_layers"] = value
    elif name == "Crossformer":
        # Crossformer has an encoder stack plus e_layers+1 decoder blocks;
        # its public constructor receives e_layers for the encoder depth.
        inferred["e_layers"] = configured["e_layers"]
    elif name == "LightTS":
        inferred["d_model"] = getattr(base_model, "d_model", None)

    effective = {}
    mismatches = []
    for key, value in configured.items():
        actual = inferred.get(key)
        effective[key] = actual if actual is not None else value
        if actual is not None and value is not None and actual != value:
            mismatches.append(
                {"field": key, "configured": value, "inferred": actual}
            )

    # Fields which do not exist in a given architecture are deliberately N/A
    # in the paper-style row rather than inherited from parser defaults.
    if name == "DLinear":
        for key in ("e_layers", "d_model", "d_ff", "n_heads", "dropout"):
            effective[key] = "N/A"
    elif name == "LightTS":
        for key in ("e_layers", "d_ff", "n_heads"):
            effective[key] = "N/A"
    elif name == "TimeMixer":
        effective["n_heads"] = "N/A"
    return effective, mismatches


def _format_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (tuple, list)):
        return ",".join(_format_value(item) for item in value)
    return str(value)


def _format_count(value):
    if value is None:
        return "UNINITIALIZED"
    return str(int(value))


def _format_millions(value):
    if value is None:
        return "UNINITIALIZED"
    return f"{value / 1_000_000.0:.6f}"


def _extra_fields(root, base_model, args, name):
    extras = {}

    quantile_wrapper = None
    native_quantile_model = None
    nwp_wrapper = None
    for module in root.modules():
        if bool(getattr(module, "supports_native_quantiles", False)) and int(
            getattr(module, "num_quantiles", 0) or 0
        ):
            native_quantile_model = module
        if hasattr(module, "quantile_head") and hasattr(module, "quantile_levels"):
            quantile_wrapper = module
        if hasattr(module, "future_cov_dim") and hasattr(module, "nwp_mode") and hasattr(
            module, "base_model_name"
        ):
            nwp_wrapper = module

    if quantile_wrapper is not None:
        extras["quantile_levels"] = tuple(quantile_wrapper.quantile_levels)
        extras["quantile_parameterization"] = getattr(
            quantile_wrapper, "quantile_parameterization", "N/A"
        )
        extras["quantile_increment_scale"] = getattr(
            quantile_wrapper, "quantile_increment_scale", "N/A"
        )
        head_counts = _count_parameters(quantile_wrapper.quantile_head)
        extras["quantile_head_parameters"] = head_counts["total"]

    if native_quantile_model is not None:
        extras["head_mode"] = "native_quantile"
        extras["quantile_levels"] = tuple(native_quantile_model.quantile_levels)
        extras["quantile_parameterization"] = getattr(
            native_quantile_model, "quantile_parameterization", "N/A"
        )
        extras["quantile_increment_scale"] = getattr(
            native_quantile_model, "quantile_increment_scale", "N/A"
        )
        native_head = getattr(native_quantile_model, "native_quantile_head", None)
        if isinstance(native_head, nn.Module):
            extras["native_quantile_head_parameters"] = _count_parameters(native_head)["total"]
        else:
            extras["native_quantile_head_parameters"] = "embedded_in_replaced_head"
        extras["original_head_shape"] = getattr(
            native_quantile_model, "original_head_shape", "N/A"
        )
        extras["native_q9_head_shape"] = getattr(
            native_quantile_model, "native_q9_head_shape", "[B,H,Q]"
        )
        extras["active_trainable_parameters"] = _count_parameters(
            native_quantile_model
        )["trainable"]
        extras["original_head_preserved"] = any(
            name.startswith("original_point_head")
            for name in vars(native_quantile_model)
        )

    if nwp_wrapper is not None:
        extras["nwp_mode"] = getattr(nwp_wrapper, "nwp_mode", "N/A")
        extras["base_model_name"] = getattr(nwp_wrapper, "base_model_name", "N/A")
        extras["future_cov_dim"] = getattr(nwp_wrapper, "future_cov_dim", "N/A")
        extras["outer_seq_len"] = getattr(nwp_wrapper, "seq_len", "N/A")
        extras["base_seq_len"] = getattr(base_model, "seq_len", "N/A")
        extras["base_enc_in"] = getattr(nwp_wrapper, "base_enc_in", "N/A")
        extras["base_effective_enc_in"] = getattr(base_model, "enc_in", "N/A")

    if name == "DLinear":
        extras["individual"] = getattr(base_model, "individual", "N/A")
        extras["channels"] = getattr(base_model, "channels", "N/A")
    elif name == "PatchTST":
        patch = getattr(base_model, "patch_embedding", None)
        extras["patch_len"] = getattr(patch, "patch_len", "N/A")
        extras["patch_stride"] = getattr(patch, "stride", "N/A")
    elif name == "Crossformer":
        extras["seg_len"] = getattr(base_model, "seg_len", "N/A")
        extras["win_size"] = getattr(base_model, "win_size", "N/A")
        decoder = getattr(base_model, "decoder", None)
        decoder_layers = getattr(decoder, "decoders", None)
        if decoder_layers is None:
            decoder_layers = getattr(decoder, "layers", None)
        if decoder_layers is None:
            decoder_layers = getattr(decoder, "decode_layers", None)
        extras["decoder_layers"] = (
            len(decoder_layers) if decoder_layers is not None else "N/A"
        )
    elif name == "TimeMixer":
        for key in (
            "channel_independence",
            "down_sampling_layers",
            "down_sampling_window",
        ):
            extras[key] = getattr(base_model, key, getattr(args, key, "N/A"))
        extras["down_sampling_method"] = getattr(
            args, "down_sampling_method", "N/A"
        )
    elif name == "LightTS":
        extras["chunk_size"] = getattr(base_model, "chunk_size", "N/A")
        extras["num_chunks"] = getattr(base_model, "num_chunks", "N/A")
    elif name == "FusionSFNoSpatial":
        for key in (
            "dim",
            "depth",
            "heads",
            "dim_head",
            "mlp_ratio",
            "decoder_dim",
            "decoder_depth",
            "decoder_heads",
            "decoder_dim_head",
            "num_mlp_heads",
            "vq_in_ts",
        ):
            extras[key] = getattr(base_model, key, "N/A")

    return extras


def collect_model_audit(model, args, stage="post_init"):
    """Return a JSON-serializable audit record for an instantiated model."""
    root = _unwrap_parallel(model)
    name = _architecture_name(args)
    base_model = _find_base_model(root, getattr(args, "model", ""))
    fields, mismatches = _standard_fields(base_model, args, name)
    parameter_counts = _count_parameters(root)
    state_counts = _count_state_dict_tensors(root)

    components = {}
    for child_name, child in root.named_children():
        components[child_name] = _count_parameters(child)

    class_chain = []
    for module in root.modules():
        class_name = module.__class__.__name__
        if class_name not in class_chain:
            class_chain.append(class_name)

    record = {
        "stage": stage,
        "requested_model": getattr(args, "model", "N/A"),
        "architecture_name": name,
        "model_class_chain": class_chain,
        "task_name": getattr(args, "task_name", "N/A"),
        "seq_len": getattr(args, "seq_len", "N/A"),
        "label_len": getattr(args, "label_len", "N/A"),
        "pred_len": getattr(args, "pred_len", "N/A"),
        "enc_in": getattr(args, "enc_in", "N/A"),
        "dec_in": getattr(args, "dec_in", "N/A"),
        "c_out": getattr(args, "c_out", "N/A"),
        "standard_fields": fields,
        "standard_field_mismatches": mismatches,
        "parameter_count": parameter_counts,
        "state_dict_tensor_count": state_counts,
        "parameter_count_definition": (
            "sum(nn.Module.parameters().numel()); includes trainable and "
            "frozen Parameters, excludes buffers"
        ),
        "state_dict_tensor_count_definition": (
            "sum(state_dict tensor numel); includes buffers and matches the "
            "legacy checkpoint metadata convention"
        ),
        "components": components,
        "extras": _extra_fields(root, base_model, args, name),
    }
    return record


def print_model_audit(model, args, stage="post_init"):
    """Print a compact, grep-friendly runtime audit block to stdout."""
    record = collect_model_audit(model, args, stage=stage)
    fields = record["standard_fields"]
    params = record["parameter_count"]
    state = record["state_dict_tensor_count"]
    name = record["architecture_name"]

    print("\n" + "=" * 78)
    print(f"MODEL_ARCHITECTURE_AUDIT_BEGIN stage={stage}")
    print(f"requested_model={record['requested_model']}")
    print(f"effective_architecture={name}")
    print(
        "model_class_chain="
        + "->".join(dict.fromkeys(record["model_class_chain"]))
    )
    print(
        "task_shape="
        f"seq_len:{record['seq_len']} label_len:{record['label_len']} "
        f"pred_len:{record['pred_len']} enc_in:{record['enc_in']} "
        f"dec_in:{record['dec_in']} c_out:{record['c_out']}"
    )
    print(
        "effective_standard_fields="
        f"e_layers:{_format_value(fields['e_layers'])} "
        f"d_model:{_format_value(fields['d_model'])} "
        f"d_ff:{_format_value(fields['d_ff'])} "
        f"n_heads:{_format_value(fields['n_heads'])} "
        f"dropout:{_format_value(fields['dropout'])}"
    )
    print(
        "parameter_count="
        f"total:{_format_count(params['total'])} "
        f"trainable:{_format_count(params['trainable'])} "
        f"non_trainable:{_format_count(params['non_trainable'])} "
        f"uninitialized:{params['uninitialized']}"
    )
    print(
        "parameter_count_m="
        f"total:{_format_millions(params['total'])} "
        f"trainable:{_format_millions(params['trainable'])}"
    )
    print(
        "state_dict_tensor_count="
        f"total:{_format_count(state['total'])} "
        f"m:{_format_millions(state['total'])} "
        f"uninitialized:{state['uninitialized']}"
    )
    print("parameter_count_definition=" + record["parameter_count_definition"])
    print(
        "state_dict_tensor_count_definition="
        + record["state_dict_tensor_count_definition"]
    )
    print(
        "paper_table_row_parameters="
        f"| {name} | {_format_value(fields['e_layers'])} | "
        f"{_format_value(fields['d_model'])} | {_format_value(fields['d_ff'])} | "
        f"{_format_value(fields['n_heads'])} | {_format_millions(params['total'])} |"
    )
    print(
        "paper_table_row_state_dict="
        f"| {name} | {_format_value(fields['e_layers'])} | "
        f"{_format_value(fields['d_model'])} | {_format_value(fields['d_ff'])} | "
        f"{_format_value(fields['n_heads'])} | {_format_millions(state['total'])} |"
    )
    if record["standard_field_mismatches"]:
        print(
            "standard_field_mismatches="
            + json.dumps(record["standard_field_mismatches"], sort_keys=True)
        )
    for key, value in record["extras"].items():
        print(f"effective_{key}={_format_value(value)}")
    for key, value in record["components"].items():
        print(
            f"component_parameters[{key}]="
            f"{_format_count(value['total'])}"
        )
    print("MODEL_ARCHITECTURE_AUDIT_JSON=" + json.dumps(record, sort_keys=True))
    print(f"MODEL_ARCHITECTURE_AUDIT_END stage={stage}")
    print("=" * 78)
    return record
