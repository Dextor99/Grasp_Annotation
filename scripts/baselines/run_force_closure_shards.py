"""Launch force-closure shards as independent, resumable Python processes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from baselines.graspnet_annotation.config import DenseAnnotationConfig


def run(geometry_run: Path, sdf_prefix: Path, shard_dir: Path, *, shard_size: int = 100, workers: int = 1) -> int:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    collision = np.load(geometry_run / "grasp_labels.npz")["collision"]
    count = int(np.count_nonzero(~np.asarray(collision, dtype=bool)))
    shard_dir.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("run_force_closure_shard.py")
    commands = []
    for start in range(0, count, shard_size):
        end = min(start + shard_size, count)
        output = shard_dir / f"fc_shard_{start:06d}.npz"
        if output.is_file() and output.with_suffix(".json").is_file():
            continue
        commands.append([
            sys.executable, str(worker), "--geometry-run", str(geometry_run),
            "--sdf-prefix", str(sdf_prefix), "--start", str(start), "--end", str(end),
            "--output", str(output),
        ])

    # Each command is still a separate Python process.  The thread pool only
    # bounds how many independent workers are alive at once and keeps resume
    # semantics unchanged.
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        futures = [executor.submit(subprocess.run, command, check=True) for command in commands]
        for future in as_completed(futures):
            future.result()
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-run", type=Path, required=True)
    parser.add_argument("--sdf-prefix", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=DenseAnnotationConfig.full().force_closure_shard_size)
    parser.add_argument("--workers", type=int, default=1,
                        help="number of independent fresh-process workers (start with 2; increase only after RAM check)")
    args = parser.parse_args()
    print({"geometry_valid_count": run(args.geometry_run, args.sdf_prefix, args.shard_dir,
                                       shard_size=args.shard_size, workers=args.workers)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
