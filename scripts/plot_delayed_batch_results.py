#!/usr/bin/env python3
"""Backfill timing and delayed-cluster size plots for completed gamma batch jobs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).absolute().parents[1]


def plot_batch(output_dir: Path, *, force: bool = False, python: str = sys.executable) -> tuple[int, int]:
    if not output_dir.is_dir():
        raise ValueError(f"Batch output directory does not exist: {output_dir}")
    plotted = 0
    skipped = 0
    for job_dir in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        events = job_dir / "clusters" / "events.csv"
        clusters = job_dir / "clusters" / "clusters.csv"
        if not (job_dir / "analysis.done").is_file() or not events.is_file() or not clusters.is_file():
            continue
        plots = (
            (
                job_dir / "delayed_time_histogram.png",
                [python, str(ROOT / "scripts" / "plot_delayed_times.py"), str(events), "--no-fit"],
            ),
            (
                job_dir / "delayed_cluster_hits.png",
                [python, str(ROOT / "scripts" / "plot_delayed_cluster_hits.py"), str(events),
                 "--clusters-csv", str(clusters)],
            ),
        )
        for destination, command in plots:
            if destination.is_file() and not force:
                skipped += 1
                continue
            subprocess.run([*command, "--output", str(destination)], check=True)
            plotted += 1
    return plotted, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Parent of per-NPZ batch output directories")
    parser.add_argument("--force", action="store_true", help="Regenerate existing plot files")
    args = parser.parse_args()
    plotted, skipped = plot_batch(args.output_dir, force=args.force)
    print(f"Created {plotted} plots; kept {skipped} existing plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
