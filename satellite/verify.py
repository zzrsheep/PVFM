"""Verify this add-on, its shared PVFM dependencies and the exported weight."""

import json
from pathlib import Path
import torch
from .checkpoint import load_model
from .data import file_hash, synthetic_batch, model_inputs


def main():
    root = Path(__file__).parent
    inventory = json.loads((root / "SHA256SUMS.json").read_text())
    for name, expected in inventory["satellite"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != expected:
            raise ValueError("Satellite checksum mismatch: " + name)
    for name, expected in inventory["shared_pvfm"].items():
        path = (root.parent / name).resolve()
        if (
            not path.is_relative_to((root.parent / "pvfm").resolve())
            or file_hash(path) != expected
        ):
            raise ValueError("Shared model dependency mismatch: " + name)
    model, payload = load_model()
    with torch.no_grad():
        prediction = model(**model_inputs(synthetic_batch(1), "cpu"))
    assert prediction.shape == (1, 4, 1) and torch.isfinite(prediction).all()
    print(
        json.dumps(
            {
                "verified_files": len(inventory["satellite"])
                + len(inventory["shared_pvfm"]),
                "strict_load": "passed",
                "synthetic_forward": "passed",
                "adaptation_step": payload["adaptation_step"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
