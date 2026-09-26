import importlib.util
from pathlib import Path
import socket
import tempfile
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "release_baseline_launcher", ROOT / "baselines/run.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_short_ipc_path_survives_deep_release(monkeypatch, tmp_path):
    monkeypatch.delenv("PVFM_IPC_TMPDIR", raising=False)
    monkeypatch.setattr(launcher, "PACKAGE", tmp_path / ("deep_" * 25))
    ipc = launcher.ipc_directory()
    assert len(str(ipc).encode()) <= 70
    with tempfile.TemporaryDirectory(prefix="pymp-", dir=ipc) as folder:
        path = Path(folder) / "listener-12345678"
        with socket.socket(socket.AF_UNIX) as s:
            s.bind(str(path))


def test_reject_long_explicit_ipc(monkeypatch, tmp_path):
    monkeypatch.setenv("PVFM_IPC_TMPDIR", str(tmp_path / ("long_" * 25)))
    with pytest.raises(ValueError, match="short path"):
        launcher.ipc_directory()
