"""Verify packaged SHA256 checksums and strictly load the original PVFM-3.9M weights."""

import hashlib
import json
from pathlib import Path

from pvfm.runtime import load_model

ROOT = Path(__file__).resolve().parent


def main():
    expected = json.loads((ROOT / "SHA256SUMS.json").read_text())
    for name, digest in expected.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError("Checksum mismatch: " + name)
    profiles = json.loads((ROOT / "configs/evaluation_profiles.json").read_text())
    counts = {}
    for name, profile in profiles.items():
        path = ROOT / profile["checkpoint"]
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == profile["checkpoint_sha256"]
        )
        model, _ = load_model(path)
        counts[name] = sum(p.numel() for p in model.parameters())
        assert counts[name] == 3879252
    print(
        json.dumps(
            {
                "files_verified": len(expected),
                "parameters": counts,
                "strict_load": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
