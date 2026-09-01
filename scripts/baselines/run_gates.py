"""Run the independent GraspNet baseline Gate 2--4 checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.graspnet_annotation.gates import report_dict, run_gates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path("baselines/graspnet_annotation/assets/debug_cube"),
        help="directory containing debug_cube.obj and debug_cube.sdf",
    )
    parser.add_argument("--report-json", type=Path, help="optional path for a JSON report")
    args = parser.parse_args()
    report = report_dict(run_gates(args.asset_dir))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(text + "\n", encoding="utf-8")
    if not all(report[name] for name in ("gate2_topology", "gate3_pose_convention", "gate4_collision_geometry", "sdf_load")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
