"""Light-weight progress tracking utilities for parameter scans."""

from __future__ import annotations

import time


def format_duration(seconds: float) -> str:
    """Return a human-readable duration string."""
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    if seconds < 3600.0:
        return f"{seconds / 60.0:.1f}m"
    return f"{seconds / 3600.0:.1f}h"


class ProgressTracker:
    """Minimal progress tracker with ETA and throughput."""

    def __init__(
        self,
        total: int,
        label: str = "Progress",
        interval: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self.total = total
        self.label = label
        self.interval = interval
        self.enabled = enabled
        self.start = time.time()
        self.last_print = self.start
        self.done = 0
        self.failed = 0

    def update(self, n: int = 1, failed: int = 0) -> None:
        self.done += n
        self.failed += failed
        if not self.enabled:
            return
        now = time.time()
        if now - self.last_print >= self.interval:
            self.print_progress(now)

    def print_progress(self, now: float | None = None) -> None:
        if not self.enabled:
            return
        now = now or time.time()
        elapsed = now - self.start
        per_point = elapsed / max(self.done, 1)
        remaining = (self.total - self.done) * per_point
        pct = 100.0 * self.done / self.total
        pts_per_sec = self.done / elapsed if elapsed > 0 else 0.0
        print(
            f"  {self.label}: {self.done}/{self.total} ({pct:.1f}%) | "
            f"elapsed {format_duration(elapsed)} | "
            f"ETA {format_duration(remaining)} | "
            f"{pts_per_sec:.2f} pts/s | "
            f"failed {self.failed}",
            flush=True,
        )
        self.last_print = now

    def finish(self) -> None:
        if not self.enabled:
            return
        self.print_progress()
