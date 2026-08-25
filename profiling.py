"""Opt-in, low-overhead stage timing for the grasp generation pipeline."""

from __future__ import annotations

import os
import time
import csv
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Iterator


_ACTIVE: ContextVar["ProfileRecorder | None"] = ContextVar("active_profile_recorder", default=None)


@dataclass(frozen=True)
class ProfileRecord:
    name: str
    seconds: float


class ProfileRecorder:
    """Collect named stage durations and print a compact report when enabled."""

    def __init__(self, enabled: bool | None = None):
        if enabled is None:
            enabled = os.getenv("GRASP_PROFILE", "").strip().lower() in {"1", "true", "yes", "on"}
        self.enabled = bool(enabled)
        self.records: list[ProfileRecord] = []
        self.counts: dict[str, int | float] = {}
        self.group_counts: dict[str, dict[str, dict[str, int]]] = {}
        self.matrix_counts: dict[str, dict[tuple[str, str], dict[str, int]]] = {}
        self._token = None

    def __enter__(self) -> "ProfileRecorder":
        if self.enabled:
            self._token = _ACTIVE.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.enabled and self._token is not None:
            _ACTIVE.reset(self._token)
            self._token = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self.records.append(ProfileRecord(name, time.perf_counter() - started))

    def print_report(self) -> None:
        if not self.enabled or (not self.records and not self.counts and not self.group_counts and not self.matrix_counts):
            return
        total = sum(record.seconds for record in self.records)
        print("\n=== Grasp pipeline profiling ===")
        if self.records:
            print(f"{'stage':40} {'seconds':>10} {'share':>8}")
            print("-" * 62)
            for record in self.records:
                share = (record.seconds / total * 100.0) if total else 0.0
                print(f"{record.name:40} {record.seconds:10.4f} {share:7.1f}%")
            print(f"{'sum of measured stages':40} {total:10.4f}")
        if self.counts:
            print("\n=== Candidate counts ===")
            for name, value in self.counts.items():
                print(f"{name:40} {value}")
        for group_name, groups in self.group_counts.items():
            print(f"\n=== {group_name} funnel ===")
            print(f"{'id':>12} {'candidate':>12} {'collision_free':>16} {'final':>12}")
            def sort_key(group_id):
                try:
                    return (0, float(group_id))
                except (TypeError, ValueError):
                    return (1, str(group_id))

            for group_id in sorted(groups, key=sort_key):
                phases = groups[group_id]
                print(
                    f"{str(group_id):>12} {phases.get('candidate', 0):12d} "
                    f"{phases.get('collision_free', 0):16d} {phases.get('final', 0):12d}"
                )
        self.write_matrix_csv()

    def count(self, name: str, value) -> None:
        """Record a candidate/cardinality metric when profiling is enabled."""
        if self.enabled:
            self.counts[name] = value

    def group_count(self, group_name: str, group_id, phase: str, amount: int = 1) -> None:
        """Record a candidate funnel count grouped by depth or variant."""
        if not self.enabled:
            return
        group = self.group_counts.setdefault(group_name, {})
        phases = group.setdefault(str(group_id), {})
        phases[phase] = phases.get(phase, 0) + amount

    def matrix_count(self, matrix_name: str, row_id, column_id, phase: str, amount: int = 1) -> None:
        """Record a two-dimensional candidate funnel, such as variant by depth."""
        if not self.enabled:
            return
        matrix = self.matrix_counts.setdefault(matrix_name, {})
        key = (str(row_id), str(column_id))
        phases = matrix.setdefault(key, {})
        phases[phase] = phases.get(phase, 0) + amount

    def write_matrix_csv(self, path="variant_depth_matrix.csv") -> None:
        """Write the variant/depth matrix as a stable CSV when profiling is active."""
        if not self.enabled or not self.matrix_counts:
            return
        path = os.fspath(path)
        for matrix_name, matrix in self.matrix_counts.items():
            output_path = path if matrix_name == "variant_depth" else f"{matrix_name}.csv"
            with open(output_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["variant_id", "depth", "candidate", "collision_free", "final"])
                for (row_id, column_id) in sorted(matrix, key=lambda item: (float(item[0]), float(item[1]))):
                    phases = matrix[(row_id, column_id)]
                    writer.writerow([
                        row_id,
                        column_id,
                        phases.get("candidate", 0),
                        phases.get("collision_free", 0),
                        phases.get("final", 0),
                    ])

    def measure(self, name: str, function, *args, **kwargs):
        """Call *function* and record its duration without changing its result."""
        with self.stage(name):
            return function(*args, **kwargs)


def active_profiler() -> ProfileRecorder:
    """Return the active recorder or a disabled recorder for instrumentation sites."""
    recorder = _ACTIVE.get()
    return recorder if recorder is not None else _DISABLED


_DISABLED = ProfileRecorder(enabled=False)


def profiled(name: str):
    """Decorate a function so its total duration is recorded when profiling is active."""
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            return active_profiler().measure(name, function, *args, **kwargs)
        return wrapper
    return decorator
