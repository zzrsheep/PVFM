"""Train only the satellite adapter on a frozen, already fine-tuned backbone.

Two-GPU recipe: AdamW, global batch 64, 5,000 successful steps, cosine LR,
complete seen validation every 500 steps, masked q50 MSE. Raw data is external.
"""

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from .checkpoint import DEFAULT_CHECKPOINT, load_model, save_checkpoint
from .data import WindowDataset, model_inputs, synthetic_batch, file_hash
from .metrics import masked_mse_components


class SmokeDataset(Dataset):
    def __init__(self):
        self.batch = synthetic_batch(4, seed=1729)
        self.fingerprint = "synthetic_smoke_only_v1"

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.batch.items()}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        prediction = model(**model_inputs(batch, device))
        loss, n = masked_mse_components(
            prediction,
            batch["future_target"].to(device),
            batch["future_target_mask"].to(device),
        )
        total += float(loss)
        count += int(n)
    if not count:
        raise ValueError("No validation targets")
    return total / count


def rng_state(device):
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Frozen backbone is reused; adapter is freshly initialized",
    )
    parser.add_argument(
        "--resume", type=Path, help="Resume a checkpoint created by this clean trainer"
    )
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--stop-after",
        type=int,
        default=0,
        help="Smoke interruption point; full 5000-step LR schedule is unchanged",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.synthetic == bool(args.train_manifest or args.val_manifest):
        raise ValueError("Choose synthetic OR both train/val manifests")
    if not args.synthetic and not (args.train_manifest and args.val_manifest):
        raise ValueError("Both train and validation manifests required")
    if args.stop_after < 0 or args.num_workers < 0:
        raise ValueError("Negative stop/worker count")
    if args.output.exists():
        raise FileExistsError("Choose a new output directory, including for resume")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device(
            "cuda", int(os.environ.get("LOCAL_RANK", device.index or 0))
        )
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=120)
        )
    try:
        torch.manual_seed(20260912 + rank)
        source = args.resume or args.checkpoint
        model, payload = load_model(source, device, reset_adapter=not bool(args.resume))
        recipe = payload["recipe"]
        if not args.synthetic and world != recipe["world_size"]:
            raise ValueError(
                "Formal recipe requires two ranks; use torchrun --nproc_per_node=2"
            )
        train_data = (
            SmokeDataset()
            if args.synthetic
            else WindowDataset(args.train_manifest, "train")
        )
        val_data = (
            SmokeDataset()
            if args.synthetic
            else WindowDataset(args.val_manifest, "val")
        )
        batch_size = 2 if args.synthetic else int(recipe["batch_size"])
        signature = {
            "train": train_data.fingerprint,
            "val": val_data.fingerprint,
            "world_size": world,
            "batch_size": batch_size,
            "synthetic": args.synthetic,
            "fp16": device.type == "cuda" and bool(recipe["fp16"]),
            "recipe": recipe,
            "device_type": device.type,
        }
        state = payload.get("clean_training_state") if args.resume else None
        if args.resume and (state is None or state["signature"] != signature):
            raise ValueError(
                "Resume requires matching clean trainer data/recipe/world size"
            )
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=recipe["train_steps"], eta_min=recipe["min_lr"]
        )
        scaler = torch.amp.GradScaler("cuda", enabled=signature["fp16"])
        step = batches_seen = amp_skips = 0
        best = math.inf
        if state:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            step = payload["adaptation_step"]
            batches_seen = state["batches_seen"]
            amp_skips = state["amp_skips"]
            best = state["best_validation_mse"]
        stop = args.stop_after or int(recipe["train_steps"])
        if not step < stop <= recipe["train_steps"]:
            raise ValueError("Stopping step must be after resume step and <=5000")
        parallel = (
            DistributedDataParallel(
                model, device_ids=[device.index] if device.type == "cuda" else None
            )
            if world > 1
            else model
        )
        sampler = DistributedSampler(
            train_data,
            num_replicas=world,
            rank=rank,
            shuffle=True,
            seed=recipe["seed"],
            drop_last=False,
        )
        loader = DataLoader(
            train_data,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(recipe["seed"]),
        )
        validation_loader = (
            DataLoader(
                val_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                generator=torch.Generator().manual_seed(recipe["seed"]),
            )
            if rank == 0
            else None
        )
        epoch, skip = divmod(batches_seen, len(loader))
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        for _ in range(skip):
            next(iterator)
        if state:
            torch.set_rng_state(state["rng_states"][rank]["torch"])
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(state["rng_states"][rank]["cuda"])
        if rank == 0:
            args.output.mkdir(parents=True)
        if world > 1:
            dist.barrier()
        frozen = {
            k: v.detach().cpu().clone() for k, v in model.backbone.state_dict().items()
        }
        consecutive_skips = 0
        while step < stop:
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            batches_seen += 1
            parallel.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=signature["fp16"]
            ):
                prediction = parallel(**model_inputs(batch, device))
                loss_sum, count = masked_mse_components(
                    prediction,
                    batch["future_target"].to(device),
                    batch["future_target_mask"].to(device),
                )
                global_count = count.clone()
                if world > 1:
                    dist.all_reduce(global_count)
                loss = loss_sum * world / global_count
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                trainable, recipe["grad_clip"], error_if_nonfinite=not signature["fp16"]
            )
            finite = torch.isfinite(norm).to(torch.int32)
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite):
                amp_skips += 1
                consecutive_skips += 1
                scaler.update(
                    new_scale=scaler.get_scale() * scaler.get_backoff_factor()
                )
                if consecutive_skips >= 20:
                    raise FloatingPointError(
                        "20 consecutive synchronized AMP overflows"
                    )
                continue
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            consecutive_skips = 0
            do_validate = (
                step % recipe["val_interval"] == 0 or step == recipe["train_steps"]
            )
            value = None
            improved = False
            if do_validate and rank == 0:
                value = validate(model, validation_loader, device)
                improved = value < best
                best = min(best, value)
            if rank == 0:
                row = {
                    "step": step,
                    "rank0_train_mse": float(loss_sum.detach() / count),
                    "validation_mse": value,
                    "lr": scheduler.get_last_lr()[0],
                    "amp_skips": amp_skips,
                }
                with (args.output / "train.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if do_validate or step == stop:
                assert not model.backbone.training
                assert all(
                    p.grad is None and not p.requires_grad
                    for p in model.backbone.parameters()
                )
                assert all(
                    torch.equal(v.detach().cpu(), frozen[k])
                    for k, v in model.backbone.state_dict().items()
                )
                states = [None] * world
                if world > 1:
                    dist.all_gather_object(states, rng_state(device))
                else:
                    states[0] = rng_state(device)
                if rank == 0:
                    exported = {
                        k: payload[k]
                        for k in (
                            "format",
                            "backbone_config",
                            "satellite_config",
                            "protocol",
                            "recipe",
                        )
                    }
                    exported.update(
                        model_state_dict=model.state_dict(),
                        adaptation_step=step,
                        best_validation_mse=best if math.isfinite(best) else None,
                        clean_training_state={
                            "signature": signature,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "scaler": scaler.state_dict(),
                            "batches_seen": batches_seen,
                            "amp_skips": amp_skips,
                            "rng_states": states,
                            "best_validation_mse": best,
                        },
                    )
                    save_checkpoint(args.output / "latest.pt", exported)
                    if improved:
                        save_checkpoint(args.output / "best.pt", exported)
                if world > 1:
                    dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
