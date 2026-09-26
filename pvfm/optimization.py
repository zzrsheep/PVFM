import math
import torch


def _resolve_warmup_steps(args):
    if args.warmup_steps > 0:
        return min(int(args.warmup_steps), max(int(args.train_steps), 1))
    if args.warmup_ratio > 0:
        return min(
            int(round(float(args.train_steps) * float(args.warmup_ratio))),
            max(int(args.train_steps), 1),
        )
    return 0


def _resolve_wsd_phase_steps(args, warmup_steps):
    """Resolve an exact warmup-stable-decay partition of ``train_steps``."""
    train_steps = max(int(args.train_steps), 1)
    decay_steps = int(getattr(args, "wsd_decay_steps", 0) or 0)
    stable_steps = int(getattr(args, "wsd_stable_steps", 0) or 0)
    if decay_steps <= 0:
        raise ValueError("--lradj wsd requires --wsd_decay_steps > 0.")
    if stable_steps <= 0:
        stable_steps = train_steps - int(warmup_steps) - decay_steps
    if (
        stable_steps < 0
        or int(warmup_steps) + stable_steps + decay_steps != train_steps
    ):
        raise ValueError(
            "WSD phases must exactly partition train_steps: "
            f"warmup={warmup_steps}, stable={stable_steps}, decay={decay_steps}, "
            f"train_steps={train_steps}."
        )
    return stable_steps, decay_steps


def _build_lr_scheduler(optimizer, args):
    train_steps = max(int(args.train_steps), 1)
    warmup_steps = _resolve_warmup_steps(args)
    min_lr_ratio = float(args.min_lr) / max(float(args.learning_rate), 1e-12)

    if args.lradj == "none":
        return None, warmup_steps
    if args.lradj == "cosine":
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=train_steps,
                eta_min=args.min_lr,
            ),
            warmup_steps,
        )
    if args.lradj == "warmup_constant":

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return max(float(step + 1) / float(warmup_steps), min_lr_ratio)
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), warmup_steps
    if args.lradj == "warmup_cosine":

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return max(float(step + 1) / float(warmup_steps), min_lr_ratio)
            decay_steps = max(train_steps - warmup_steps, 1)
            progress = min(
                max(float(step - warmup_steps + 1) / float(decay_steps), 0.0), 1.0
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), warmup_steps
    if args.lradj == "wsd":
        stable_steps, decay_steps = _resolve_wsd_phase_steps(args, warmup_steps)

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return max(float(step + 1) / float(warmup_steps), min_lr_ratio)
            stable_end = warmup_steps + stable_steps
            if step < stable_end:
                return 1.0
            progress = min(
                max(float(step - stable_end + 1) / float(decay_steps), 0.0), 1.0
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), warmup_steps
    raise ValueError(f"Unsupported lradj: {args.lradj}")
