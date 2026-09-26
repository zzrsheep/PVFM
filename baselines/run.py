"""Run an archived full-shot configuration on packaged Australian data.

Default is a dry run. --execute starts the original training + test loop;
--audit performs dataset/origin checks and one CPU forward/backward only.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.protocol import materialize_engine

BASE = Path(__file__).resolve().parent
PACKAGE = BASE.parent


def ipc_directory():
    """Multiprocessing AF_UNIX sockets cannot use a deep experiment path."""
    explicit = os.environ.get("PVFM_IPC_TMPDIR")
    path = Path(explicit).resolve() if explicit else PACKAGE / ".ipc"
    # Python adds /pymp-XXXXXXXX/listener-XXXXXXXX; Linux permits 107 bytes.
    if len(os.fsencode(str(path))) > 70:
        if explicit:
            raise ValueError("PVFM_IPC_TMPDIR must be a short path (at most 70 bytes)")
        path = Path("/tmp")
    path.mkdir(parents=True, exist_ok=True)
    return path


def select(args):
    jobs = json.loads((BASE / "catalog.json").read_text())["jobs"]
    for key in ["id", "model", "mode", "task", "seed"]:
        value = getattr(args, key)
        if value is not None:
            jobs = [j for j in jobs if j[key] == value]
    if args.station:
        jobs = [j for j in jobs if args.station in j["station_dir"]]
    return jobs


def config_path(job):
    """Resolve a catalogued readable configuration, never an external path."""
    relative = Path(job["config"])
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.parts[0] != "configs"
        or relative.suffix != ".json"
    ):
        raise ValueError(f"Invalid baseline configuration path: {relative}")
    path = (BASE / relative).resolve()
    if not path.is_relative_to((BASE / "configs").resolve()):
        raise ValueError(f"Configuration escapes baseline configs: {relative}")
    return path


def load_config(job):
    config = json.loads(config_path(job).read_text())
    for key in ("id", "model", "mode", "task", "station_dir", "seed", "protocol"):
        if config[key] != job[key]:
            raise ValueError(f"Catalog/config mismatch for {key}: {job['config']}")
    return config


def run_directory(job, output):
    """Use the same model/task/station/seed hierarchy for new results."""
    config_path(job)
    return Path(output) / Path(job["config"]).relative_to("configs").with_suffix("")


def command_for(job, out, workers=None):
    config = load_config(job)
    opts = {
        k: [v.replace("{PACKAGE}", str(PACKAGE)) for v in values]
        for k, values in config["options"].items()
    }
    opts["--checkpoints"] = [str(out / "checkpoints")]
    opts["--station_cache_dir"] = [str(out / "station_cache")]
    # Inputs/task/head/seed/optimizer/epochs remain as recorded. Device IDs are
    # local to CUDA_VISIBLE_DEVICES; there is no four-GPU auto-launch.
    opts["--gpu"] = ["0"]
    if workers is not None:
        opts["--num_workers"] = [str(workers)]
    argv = [v for key, values in opts.items() for v in [key, *values]]
    return config, argv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ["id", "model", "mode", "task", "station"]:
        p.add_argument("--" + field)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--all-seeds", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int)
    args = p.parse_args()
    if args.execute and args.audit:
        p.error("Choose --execute or --audit")
    if args.all_seeds:
        args.seed = None
    jobs = select(args)
    if not jobs:
        p.error("No matching archived jobs")
    if not args.execute and not args.audit:
        print(json.dumps(jobs, ensure_ascii=False, indent=2))
        print(f"{len(jobs)} jobs; no process launched.")
        return
    if args.output is None:
        p.error("--output is required")
    out = args.output.resolve()
    if not out.is_relative_to(PACKAGE):
        p.error("Output must be inside the standalone package")
    out.mkdir(parents=True, exist_ok=False)
    ipc = ipc_directory()
    for job in jobs:
        run = run_directory(job, out)
        run.mkdir(parents=True)
        config, argv = command_for(job, run, 0 if args.audit else args.workers)
        engine = materialize_engine(config["protocol"], out)
        env = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(engine),
            "PVFM_RESULTS_ROOT": str(run / "results"),
            "PVFM_TEST_RESULTS_ROOT": str(run / "test_results"),
            "PVFM_RESULT_LOG": str(run / "result.log"),
            "MPLCONFIGDIR": str(out / "mplcache"),
            "TMPDIR": str(ipc),
        }
        entry = BASE / "audit_job.py" if args.audit else engine / "run.py"
        command = [sys.executable, str(entry), *argv]
        (run / "launch.json").write_text(
            json.dumps(dict(job=job, argv=command, smoke_only=args.audit), indent=2)
        )
        print(
            ("AUDIT" if args.audit else "TRAIN"),
            job["model"],
            job["mode"],
            job["task"],
            job["station_dir"],
            flush=True,
        )
        with (run / "train.log").open("w") as log:
            subprocess.run(
                command,
                cwd=engine,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if args.execute:
            subprocess.run(
                [sys.executable, str(BASE / "score.py"), "--run", str(run)],
                cwd=PACKAGE,
                env={**env, "PYTHONPATH": str(PACKAGE)},
                check=True,
            )
    print("COMPLETE", out)


if __name__ == "__main__":
    main()
