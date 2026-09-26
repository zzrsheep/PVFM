"""Materialize a frozen baseline protocol from shared sources and small overrides."""

import json
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent


def materialize_engine(protocol, output):
    definitions = json.loads((BASE / "protocols.json").read_text())["protocols"]
    if protocol not in definitions:
        raise ValueError(f"Unknown baseline protocol: {protocol}")
    target = Path(output) / ".engines" / protocol
    if target.is_dir():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BASE / "engine", target)
    overrides = BASE / "compat" / protocol
    if overrides.is_dir():
        shutil.copytree(overrides, target, dirs_exist_ok=True)
    return target
