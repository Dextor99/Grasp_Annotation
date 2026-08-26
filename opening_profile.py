"""Candidate-level opening statistics for auditing monotonicity assumptions."""

from __future__ import annotations

import csv
from pathlib import Path


class OpeningProfiler:
    """Collect opening outcomes without changing the grasp filtering pipeline."""

    fieldnames = (
        "candidate_id",
        "depth_id",
        "depth_value",
        "angle_deg",
        "opening",
        "collision_free",
        "opening_selected",
        "final_valid",
    )
    fieldnames_line = ",".join(fieldnames)

    def __init__(self):
        self.records: list[dict] = []

    def add(
        self,
        candidate_id,
        depth_id,
        depth_value,
        angle_deg,
        opening,
        collision_free,
        opening_selected,
        final_valid,
    ):
        self.records.append({
            "candidate_id": candidate_id,
            "depth_id": depth_id,
            "depth_value": depth_value,
            "angle_deg": angle_deg,
            "opening": opening,
            "collision_free": int(bool(collision_free)),
            "opening_selected": int(bool(opening_selected)),
            "final_valid": int(bool(final_valid)),
        })

    def save(self, path="opening_profile.csv"):
        """Write all opening records, including a header for an empty run."""
        output_path = Path(path)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows({key: record.get(key) for key in self.fieldnames} for record in self.records)
