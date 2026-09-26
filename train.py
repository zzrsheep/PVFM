"""Core PVFM Q11 training: temporal-pool batches, Adam, WSD, AMP, DDP/resume.

No pretraining data is distributed. --synthetic is solely a smoke test;
it does not read the held-out example dataset or the pretrained checkpoint.
"""

import argparse
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from pvfm.runtime import PVFM, model_inputs, safe_output
from pvfm.loss import masked_pinball_loss_components
from pvfm.optimization import _build_lr_scheduler
from pvfm.training_data import policy, load_training, prepare

ROOT = Path(__file__).resolve().parent


def synthetic_batch(step, batch_size, rank):
    # Deterministic fake data, never derived from the public test examples.
    rng = torch.Generator().manual_seed(1729 + step * 100 + rank)
    context_len, horizon_len = (24, 6) if step % 2 else (64, 16)
    batch_count = batch_size
    batch = {
        "past_target": torch.rand(batch_count, context_len, 1, generator=rng),
        "past_observed_mask": torch.ones(batch_count, context_len, 1),
        "historical_covariates": torch.randn(
            batch_count, context_len, 6, generator=rng
        ),
        "historical_covariates_mask": torch.ones(batch_count, context_len, 6),
        "future_covariates": torch.randn(batch_count, horizon_len, 6, generator=rng),
        "future_covariates_mask": torch.ones(batch_count, horizon_len, 6),
        "past_time_features": torch.zeros(batch_count, context_len, 5),
        "future_time_features": torch.zeros(batch_count, horizon_len, 5),
        "static_features": torch.tensor(
            [context_len, horizon_len, 1 if context_len == 24 else 0.25, 0.0],
            dtype=torch.float32,
        ).repeat(batch_count, 1),
        "site_features": torch.tensor([51.8, 1.2, 0.0, 0.0, 0.0]).repeat(
            batch_count, 1
        ),
        "future_target": torch.rand(batch_count, horizon_len, 1, generator=rng),
        "future_target_mask": torch.ones(batch_count, horizon_len, 1),
        "resolutions": ["1h" if context_len == 24 else "15min"] * batch_count,
    }
    return batch


def rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state[0],
            np.asarray(numpy_state[1], dtype=np.uint32),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
    )
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default=str(ROOT / "configs/pvfm_3_9m.json"))
    parser.add_argument("--data-config")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--stop-after",
        type=int,
        default=0,
        help="Stop/save at this absolute step; does not shorten the 80k LR schedule",
    )
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--resume",
        help="Checkpoint produced by THIS standalone trainer, not inference-only weights",
    )
    parser.add_argument("--output", default=str(ROOT / "outputs/train"))
    parser.add_argument("--save-every", type=int, default=10000)
    args = parser.parse_args()
    if args.prepare_only:
        if not args.data_config:
            parser.error("--prepare-only requires --data-config")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            parser.error("Prepare in one CPU process, not torchrun")
        prepare(args.data_config)
        return
    if args.synthetic and args.data_config:
        parser.error("--synthetic and --data-config are mutually exclusive")
    if not args.synthetic and not args.data_config:
        parser.error(
            "Provide external training data, or explicitly use --synthetic for a smoke test"
        )
    if args.synthetic and not args.stop_after:
        parser.error("--synthetic requires a small explicit --stop-after")
    torch.set_num_threads(args.threads)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    training_config = policy()
    seed = training_config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model_config = json.loads(Path(args.model_config).read_text())
    model = PVFM(**model_config).to(device)
    batch_size = args.batch_size or (
        2 if args.synthetic else training_config["per_gpu_batch_size"]
    )
    if args.synthetic:
        loader = sampler = None
        binding = {"synthetic_version": 1}
    else:
        loader, sampler, binding = load_training(
            args.data_config, batch_size, rank, world, args.workers
        )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        betas=tuple(training_config["betas"]),
        weight_decay=training_config["weight_decay"],
    )
    scheduler, _ = _build_lr_scheduler(optimizer, SimpleNamespace(**training_config))
    fp16 = training_config["fp16"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    contract = {
        "model_config": model_config,
        "training": training_config,
        "batch_size": batch_size,
        "world_size": world,
        "binding": binding,
        "amp": fp16,
    }
    step = 0
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=True)
        if resume.get("contract") != contract:
            raise ValueError(
                "Resume model/data/pool/profile/schedule/world-size mismatch"
            )
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        step = resume["training_step"]
        if sampler is not None:
            sampler.load_state_dict(resume["rank_states"][rank]["sampler"])
    if world > 1:
        model = DDP(
            model,
            device_ids=[local] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    if rank == 0:
        output = safe_output(args.output)
    else:
        output = Path(args.output).resolve()
    if world > 1:
        dist.barrier()
    # Create the loader iterator before restoring model/dropout RNG state.
    iterator = iter(loader) if loader is not None else None
    if resume is not None:
        restore_rng(resume["rank_states"][rank]["rng"])
    stop = args.stop_after or training_config["train_steps"]
    if not step < stop <= training_config["train_steps"]:
        raise ValueError("stop-after must exceed resumed step and not exceed 80k")
    if rank == 0:
        (output / "run_config.json").write_text(json.dumps(contract, indent=2) + "\n")

    def save():
        raw = model.module if isinstance(model, DDP) else model
        state = {
            "rng": rng_state(),
            "sampler": sampler.state_dict() if sampler is not None else None,
        }
        rank_states = [None] * world
        if world > 1:
            dist.all_gather_object(rank_states, state)
        else:
            rank_states = [state]
        if rank == 0:
            payload = {
                "format_version": 1,
                "model_config": model_config,
                "model_state_dict": raw.state_dict(),
                "training_step": step,
                "contract": contract,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rank_states": rank_states,
            }
            target = output / f"step_{step:06d}.pt"
            if target.exists():
                raise FileExistsError(target)
            temporary = output / (target.name + ".partial")
            torch.save(payload, temporary)
            temporary.replace(target)
        if world > 1:
            dist.barrier()

    while step < stop:
        batch = (
            synthetic_batch(step, batch_size, rank)
            if args.synthetic
            else next(iterator)
        )
        model.train()
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=fp16):
            _, aux = model(**model_inputs(batch, device))
            loss_sum, weight = masked_pinball_loss_components(
                aux["quantiles"],
                batch["future_target"].to(device),
                aux["quantile_levels"],
                mask=batch["future_target_mask"].to(device),
            )
            loss = loss_sum / weight
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        if fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()
        step += 1
        if sampler is not None:
            sampler.mark_consumed(batch)
        if rank == 0 and (step <= 5 or step % 100 == 0 or step == stop):
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss.detach()),
                        "lr": optimizer.param_groups[0]["lr"],
                        "C": batch["past_target"].shape[1],
                        "H": batch["future_target"].shape[1],
                    }
                ),
                flush=True,
            )
        if step % args.save_every == 0 or step == stop:
            save()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
