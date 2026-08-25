"""Candidate-level quality records for learning scale-normalized grasp depths."""

from __future__ import annotations

import csv
from pathlib import Path


class DepthProfiler:
    """Collect per-candidate depth outcomes without changing filtering behavior."""

    fieldnames = (
        "variant_id",
        "depth_id",
        "depth_value",
        "collision_free",
        "opening_valid",
        "intersection_valid",
        "contact_points",
        "final_valid",
    )
    fieldnames_line = ",".join(fieldnames)

    def __init__(self):
        self.records: list[dict] = []

    def add(
        self,
        variant_id,
        depth_id,
        depth_value,
        collision_free,
        opening_valid,
        intersection_valid,
        contact_points,
    ):
        self.records.append(
            {
                "variant_id": variant_id,
                "depth_id": depth_id,
                "depth_value": depth_value,
                "collision_free": int(bool(collision_free)),
                "opening_valid": int(bool(opening_valid)),
                "intersection_valid": int(bool(intersection_valid)),
                "contact_points": int(contact_points),
                "final_valid": int(bool(intersection_valid)),
            }
        )

    def save(self, path="depth_profile.csv"):
        """Write records to CSV, including a header when no candidates exist."""
        output_path = Path(path)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
