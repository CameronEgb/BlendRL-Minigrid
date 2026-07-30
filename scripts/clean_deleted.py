#!/usr/bin/env python3
"""
clean_deleted.py

Reads deleted experiment output paths from a tracking file (default: in/deleted_experiments.txt)
and moves matching files/directories in results/ on the cluster to results/legacy_results/.

This prevents rsync from pulling deleted experiments back from the cluster into local results.

Usage:
  # Run clean_deleted on cluster or locally:
  python3 scripts/clean_deleted.py

  # Preview moves with dry-run:
  python3 scripts/clean_deleted.py --dry-run
"""

import argparse
import os
import shutil
from pathlib import Path


DEFAULT_TRACKING_FILE = "in/deleted_experiments.txt"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_LEGACY_DIR = "results/legacy_results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Move deleted experiment outputs from results/ to results/legacy_results/ based on tracking file."
    )
    parser.add_argument(
        "-t",
        "--tracking-file",
        default=DEFAULT_TRACKING_FILE,
        help=f"Path to deleted experiments tracking file (default: {DEFAULT_TRACKING_FILE}).",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"Path to main results directory (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--legacy-dir",
        default=DEFAULT_LEGACY_DIR,
        help=f"Path to legacy results destination directory (default: {DEFAULT_LEGACY_DIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without moving any files.",
    )
    return parser.parse_args()


def merge_move(src_path: Path, dst_path: Path, dry_run: bool = False):
    """Recursively moves src_path to dst_path, merging directories if dst_path exists."""
    if not src_path.exists():
        return False
    if src_path.resolve() == dst_path.resolve():
        return False

    if dry_run:
        item_type = "DIR " if src_path.is_dir() else "FILE"
        print(f"  [DRY RUN MOVE] [{item_type}] {src_path} -> {dst_path}")
        return True

    if not dst_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dst_path))
        print(f"MOVED: {src_path} -> {dst_path}")
        return True
    elif src_path.is_dir() and dst_path.is_dir():
        moved = False
        for item in list(src_path.iterdir()):
            if merge_move(item, dst_path / item.name, dry_run=False):
                moved = True
        try:
            src_path.rmdir()
            print(f"REMOVED EMPTY DIR: {src_path}")
        except OSError:
            pass
        return moved
    else:
        # Both are files: overwrite if src is newer or different
        if dst_path.exists():
            dst_path.unlink()
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dst_path))
        print(f"MOVED (OVERWROTE): {src_path} -> {dst_path}")
        return True


def clean_deleted(
    tracking_file: str = DEFAULT_TRACKING_FILE,
    results_dir: str = DEFAULT_RESULTS_DIR,
    legacy_dir: str = DEFAULT_LEGACY_DIR,
    dry_run: bool = False,
):
    tracking_path = Path(tracking_file)
    results_path = Path(results_dir)
    legacy_path = Path(legacy_dir)

    if not tracking_path.exists():
        print(f"Tracking file '{tracking_file}' not found. No deleted experiments to clean.")
        return

    if not results_path.exists():
        print(f"Results directory '{results_dir}' not found.")
        return

    entries = []
    with open(tracking_path, "r") as f:
        for line in f:
            line_str = line.strip()
            if line_str and not line_str.startswith("#"):
                entries.append(line_str)

    if not entries:
        print(f"No deleted experiment paths found in '{tracking_file}'.")
        return

    print(f"=== Cleaning Deleted Experiments from '{results_dir}' -> '{legacy_dir}' ===")
    print(f"Loaded {len(entries)} tracked entries from '{tracking_file}'.\n")

    items_to_move = []
    for rel_str in entries:
        src_target = results_path / rel_str
        dst_target = legacy_path / rel_str

        if src_target.exists():
            items_to_move.append((src_target, dst_target))

    if not items_to_move:
        print("No matching deleted files or directories found in results/ to move.")
        return

    print(f"Found {len(items_to_move)} matching item(s) to move:")
    moved_count = 0
    for src, dst in items_to_move:
        if merge_move(src, dst, dry_run=dry_run):
            moved_count += 1

    if dry_run:
        print(f"\n[DRY RUN] Would move {len(items_to_move)} item(s). No files were moved.")
    else:
        print(f"\nSuccessfully moved {moved_count} item(s) to '{legacy_dir}'.")


def main():
    args = parse_args()
    clean_deleted(
        tracking_file=args.tracking_file,
        results_dir=args.results_dir,
        legacy_dir=args.legacy_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
