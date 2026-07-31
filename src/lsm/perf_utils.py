"""
Stage 6: wall-clock + peak-RSS measurement for the scale rehearsal.

`resource.getrusage` (POSIX-only) is not usable on this project's Windows dev
box, so peak RSS is sampled cross-platform via a polling thread reading
`psutil.Process().memory_info().rss` -- an approximation (true peak between
samples can be missed), not an exact accounting, and the report should say so.
Zero project-internal imports, so this is safe to import from anywhere
(scripts, scale_eval.py, tests) with no import-order risk.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import psutil


@dataclass
class PerfResult:
    label: str
    n_rows: int | None = None
    wall_seconds: float = 0.0
    start_rss_bytes: int = 0
    peak_rss_bytes: int = 0

    @property
    def peak_rss_mb(self) -> float:
        return self.peak_rss_bytes / (1024 * 1024)

    @property
    def rows_per_sec(self) -> float | None:
        if self.n_rows is None or self.wall_seconds <= 0:
            return None
        return self.n_rows / self.wall_seconds


@contextmanager
def measure(label: str, n_rows: int | None = None, poll_interval_s: float = 0.05) -> Iterator[PerfResult]:
    """Yields a `PerfResult` that is mutated in place -- read its fields AFTER
    the `with` block exits, not inside it (wall_seconds/peak_rss_bytes are 0
    until __exit__ has run).
    """
    process = psutil.Process()
    start_rss = process.memory_info().rss
    result = PerfResult(label=label, n_rows=n_rows, start_rss_bytes=start_rss, peak_rss_bytes=start_rss)

    stop_event = threading.Event()
    peak = [start_rss]

    def _poll() -> None:
        while not stop_event.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            stop_event.wait(poll_interval_s)

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.wall_seconds = time.perf_counter() - start
        stop_event.set()
        thread.join()
        peak[0] = max(peak[0], process.memory_info().rss)
        result.peak_rss_bytes = peak[0]


def format_perf_table(results: list[PerfResult]) -> str:
    """Markdown table: step | wall_seconds | rows/sec | peak RSS (MB)."""
    lines = [
        "| step | rows | wall (s) | rows/sec | peak RSS (MB) |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        rows_str = f"{r.n_rows:,}" if r.n_rows is not None else "-"
        rate_str = f"{r.rows_per_sec:,.0f}" if r.rows_per_sec is not None else "-"
        lines.append(f"| {r.label} | {rows_str} | {r.wall_seconds:.2f} | {rate_str} | {r.peak_rss_mb:,.1f} |")
    return "\n".join(lines)
