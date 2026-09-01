"""Invoke the compatible SDFGen executable without touching the source mesh."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


# The formal GN-Full protocol uses a 100^3 signed-distance grid.  Smaller
# grids are still useful for quick sanity checks, but must be requested
# explicitly so a debug run is never mistaken for a formal annotation.
DEFAULT_GRID_DIM = 100


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf-exe", type=Path, required=True)
    parser.add_argument("--obj", type=Path, required=True)
    parser.add_argument("--grid-dim", type=int, default=DEFAULT_GRID_DIM)
    parser.add_argument("--padding", type=int, default=5)
    args = parser.parse_args()
    if not args.sdf_exe.is_file():
        raise FileNotFoundError(args.sdf_exe)
    if args.obj.suffix.lower() != ".obj" or not args.obj.is_file():
        raise FileNotFoundError(f"triangle OBJ not found: {args.obj}")
    if args.grid_dim < 2 or args.padding < 1:
        raise ValueError("grid-dim must be >= 2 and padding must be >= 1")
    subprocess.run(
        [str(args.sdf_exe), str(args.obj), str(args.grid_dim), str(args.padding)],
        check=True,
    )
    output = args.obj.with_suffix(".sdf")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"SDFGen completed without producing {output}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
