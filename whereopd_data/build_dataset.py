#!/usr/bin/env python3
"""Build the WhereOPD datasets.

    python build_dataset.py list
    python build_dataset.py build whereopd_count_mcq_3k       # Qwen3.5-4B training set
    python build_dataset.py build whereopd_count_open_3k      # Qwen3.5-9B training set


"""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENERATOR = HERE / "generate_scenes.py"
OPEN_ENDED = HERE / "make_open_ended.py"

# Generator flags that differ from generate_scenes.py's defaults.
SCENE_ARGS = {"min_objects": 12, "max_objects": 40, "min_radius": 14, "max_radius": 64}

CONFIGS: dict[str, dict] = {
    "whereopd_count_mcq_3k": {
        "out_dir": "whereopd_count_mcq_3k",
        "generate": {"n_images": 3000, "seed": 42, **SCENE_ARGS},
        "about": "3,000 multiple-choice counting scenes (A-D), 1024x1024, 12-40 objects, radius 14-64. "
                 "Qwen3.5-4B training set.",
    },
    "whereopd_count_open_3k": {
        "out_dir": "whereopd_count_open_3k",
        "derive_from": "whereopd_count_mcq_3k",
        "about": "The same 3,000 scenes asked as an open-ended question; ground truth is the count. "
                 "Qwen3.5-9B training set.",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def command_for(name: str, root: Path) -> list[str]:
    cfg = CONFIGS[name]
    out_dir = root / cfg["out_dir"]
    if "derive_from" in cfg:
        src = root / CONFIGS[cfg["derive_from"]]["out_dir"] / "train.parquet"
        return [sys.executable, str(OPEN_ENDED), "--in", str(src), "--out", str(out_dir / "train.parquet")]
    argv = [sys.executable, str(GENERATOR), "--out_dir", str(out_dir)]
    for k, v in cfg["generate"].items():
        argv += [f"--{k}", str(v)]
    return argv


def build(name: str, root: Path, force: bool) -> None:
    cfg = CONFIGS[name]
    out_dir = root / cfg["out_dir"]
    if (out_dir / "train.parquet").exists() and not force:
        print(f"{out_dir}/train.parquet exists; pass --force to rebuild")
        return
    if out_dir.exists():
        shutil.rmtree(out_dir)  # a rebuild starts from an empty folder: no stale files
    if "derive_from" in cfg and not (root / CONFIGS[cfg["derive_from"]]["out_dir"] / "train.parquet").exists():
        build(cfg["derive_from"], root, force=False)
    argv = command_for(name, root)
    print("$ " + " ".join(argv), flush=True)
    t0 = time.time()
    if subprocess.call(argv) != 0:
        sys.exit(f"{name}: build failed")
    script = OPEN_ENDED if "derive_from" in cfg else GENERATOR
    manifest = {
        "config": name,
        "argv": argv[1:],
        "script": {"file": script.name, "sha256": sha256(script)},
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "build_seconds": round(time.time() - t0, 1),
        "environment": {"python": platform.python_version(), "node": platform.node()},
    }
    (out_dir / "generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out_dir}/generation_manifest.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="print the recorded datasets")
    b = sub.add_parser("build", help="build one dataset")
    b.add_argument("config", choices=sorted(CONFIGS))
    b.add_argument("--root", default=str(HERE), help="directory the dataset folders are written into")
    b.add_argument("--force", action="store_true", help="rebuild even if train.parquet exists")
    args = ap.parse_args()
    if args.cmd == "list":
        for name, cfg in CONFIGS.items():
            print(f"{name}\n  {cfg['about']}\n  $ {' '.join(command_for(name, Path('.'))[1:])}\n")
        return 0
    build(args.config, Path(args.root), args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
