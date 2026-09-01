"""Launch deterministic GN-Full force-closure evaluation for a fixed subset."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def partition_ids(candidate_ids: np.ndarray, shard_size: int) -> list[np.ndarray]:
    """Partition IDs in manifest order without sorting or changing selection."""

    if int(shard_size) <= 0:
        raise ValueError("shard_size must be positive")
    values = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
    if len(values) == 0:
        raise ValueError("candidate_ids must not be empty")
    return [values[start : start + int(shard_size)] for start in range(0, len(values), int(shard_size))]


def run(
    geometry_run: Path,
    sdf_prefix: Path,
    candidate_ids: Path,
    shard_dir: Path,
    *,
    shard_size: int = 100,
    workers: int = 1,
) -> int:
    if int(workers) <= 0:
        raise ValueError("workers must be positive")
    ids = np.asarray(np.load(candidate_ids), dtype=np.int64).reshape(-1)
    chunks = partition_ids(ids, shard_size)
    shard_dir.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("run_force_closure_subset_shard.py")
    commands = []
    offset = 0
    for ordinal, chunk in enumerate(chunks):
        output = shard_dir / f"subset_fc_shard_{ordinal:06d}.npz"
        end = offset + len(chunk)
        if output.is_file() and output.with_suffix(".json").is_file():
            offset = end
            continue
        commands.append(
            [
                sys.executable,
                str(worker),
                "--geometry-run",
                str(geometry_run),
                "--sdf-prefix",
                str(sdf_prefix),
                "--candidate-ids",
                str(candidate_ids),
                "--start",
                str(offset),
                "--end",
                str(end),
                "--output",
                str(output),
            ]
        )
        offset = end
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        futures = [executor.submit(subprocess.run, command, check=True) for command in commands]
        for future in as_completed(futures):
            future.result()
    return int(len(ids))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-run", type=Path, required=True)
    parser.add_argument("--sdf-prefix", type=Path, required=True)
    parser.add_argument("--candidate-ids", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    count = run(
        args.geometry_run,
        args.sdf_prefix,
        args.candidate_ids,
        args.shard_dir,
        shard_size=args.shard_size,
        workers=args.workers,
    )
    print({"subset_count": count})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
