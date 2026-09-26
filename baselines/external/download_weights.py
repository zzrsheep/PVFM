"""Explicit, user-initiated upstream download; never downloads on import."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    models = json.loads(Path(__file__).with_name("weights.json").read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=sorted(models), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rec = models[a.model]
    out = a.output.resolve()
    if not out.is_relative_to(ROOT):
        p.error("Output must be inside this package")
    if out.exists():
        p.error("Choose a new directory; existing weights are never overwritten")
    from huggingface_hub import snapshot_download

    out.mkdir(parents=True)
    kwargs = {"repo_id": rec["repo"], "local_dir": str(out)}
    if rec.get("revision"):
        kwargs["revision"] = rec["revision"]
    if rec.get("filename"):
        kwargs["allow_patterns"] = [
            rec["filename"],
            "LICENSE*",
            "README*",
            "config.json",
        ]
    snapshot_download(**kwargs)
    for name, digest in rec.get("expected_files", {}).items():
        with (out / name).open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"Upstream bytes differ from archived weights: {name}")
    print(
        "Downloaded. Weight hash coverage:",
        len(rec.get("expected_files", {})),
        rec["revision_note"],
    )
    print("Checkpoint:", out / rec["filename"] if rec.get("filename") else out)


if __name__ == "__main__":
    main()
