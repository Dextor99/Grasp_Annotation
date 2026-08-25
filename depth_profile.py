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
        "depth_ratio",
        "collision_free",
        "opening_valid",
        "intersection_valid",
        "contact_points",
        "contact_ratio",
        "score_total",
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
        depth_ratio,
        collision_free,
        opening_valid,
        intersection_valid,
        contact_points,
        surface_point_count,
        candidate_id,
    ):
        point_count = max(int(surface_point_count), 0)
        contact_count = max(int(contact_points), 0)
        self.records.append({
                "_candidate_id": candidate_id,
                "variant_id": variant_id,
                "depth_id": depth_id,
                "depth_value": depth_value,
                "depth_ratio": depth_ratio,
                "collision_free": int(bool(collision_free)),
                "opening_valid": int(bool(opening_valid)),
                "intersection_valid": int(bool(intersection_valid)),
                "contact_points": contact_count,
                "contact_ratio": (contact_count / point_count) if point_count else 0.0,
                "score_total": None,
                "final_valid": int(bool(intersection_valid)),
            })

    def update_scores(self, candidates, score_key="final_score"):
        """Attach scores computed by the existing scoring stage to profile rows."""
        scores = {
            candidate.get("id"): candidate.get(score_key)
            for candidate in candidates
            if candidate.get("id") is not None and candidate.get(score_key) is not None
        }
        for record in self.records:
            candidate_id = record.get("_candidate_id")
            if candidate_id in scores:
                record["score_total"] = scores[candidate_id]

    def save(self, path="depth_profile.csv"):
        """Write records to CSV, including a header when no candidates exist."""
        output_path = Path(path)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows({key: record.get(key) for key in self.fieldnames} for record in self.records)
