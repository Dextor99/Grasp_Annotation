"""Opt-in, low-overhead stage timing for the grasp generation pipeline."""

from __future__ import annotations

import os
import time
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
        if not self.enabled or (not self.records and not self.counts):
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

    def count(self, name: str, value) -> None:
        """Record a candidate/cardinality metric when profiling is enabled."""
        if self.enabled:
            self.counts[name] = value

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
