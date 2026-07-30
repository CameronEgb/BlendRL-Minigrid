#!/usr/bin/env python3
"""
Script to erase experiment results from the results directory tree and track deleted items.

Usage:
  # Erase a specific experiment across all result directories:
  python scripts/erase_experiment.py -e early_prediction_test -g early_prediction

  # Erase all experiments in a group except specified ones to keep:
  python scripts/erase_experiment.py -g early_prediction -k weighted_pr

  # Dry run to preview what would be deleted:
  python scripts/erase_experiment.py -g early_prediction -k weighted_pr --dry-run
"""

import argparse
import os
import shutil
from pathlib import Path


DEFAULT_RESULT_CATEGORIES = [
    "logs",
    "datasets",
    "checkpoints",
    "plots",
    "experiments",
    "tensorboard",
    "optuna",
    "slurm_ids",
]

DEFAULT_TRACKING_FILE = "in/deleted_experiments.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Erase experiment results from the results directory and track deleted items."
    )
    parser.add_argument(
        "-e",
        "--experiment",
        nargs="+",
        help="Experiment ID(s) to erase.",
    )
    parser.add_argument(
        "-g",
        "--group",
        help="Group name (e.g. early_prediction, cartpole, mimic, tuning).",
    )
    parser.add_argument(
        "-k",
        "--keep",
        nargs="+",
        help="Experiment ID(s) or pattern(s) to keep when cleaning a group.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Path to results directory (default: results).",
    )
    parser.add_argument(
        "--tracking-file",
        default=DEFAULT_TRACKING_FILE,
        help=f"File to store tracked deletions (default: {DEFAULT_TRACKING_FILE}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without deleting any files.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force deletion without interactive confirmation prompt.",
    )
    return parser.parse_args()


def expand_keep_patterns(keep_list: list) -> list:
    """Expand keep patterns with common aliases and variations (e.g. weighted_pr -> weighted, weighted-pr)."""
    expanded = set()
    for kp in keep_list:
        kp_clean = kp.strip().lower()
        expanded.add(kp_clean)
        expanded.add(kp_clean.replace("_", "-"))
        expanded.add(kp_clean.replace("-", "_"))
        if "_pr" in kp_clean:
            expanded.add(kp_clean.replace("_pr", ""))
        if "-pr" in kp_clean:
            expanded.add(kp_clean.replace("-pr", ""))
    return list(expanded)


def should_keep_file_or_dir(item_name: str, keep_patterns: list) -> bool:
    """Check if an item matches any keep pattern or essential file rule."""
    if item_name.startswith(".git"):
        return True
    if not keep_patterns:
        return False
    item_lower = item_name.lower()
    expanded_patterns = expand_keep_patterns(keep_patterns)
    for kp in expanded_patterns:
        if kp == item_lower or kp in item_lower:
            return True
    return False


def record_deleted_items(items_to_remove: list, results_dir: Path, tracking_file: Path):
    """Record relative paths of deleted items into the tracking file."""
    tracking_file.parent.mkdir(parents=True, exist_ok=True)
    existing_entries = set()
    if tracking_file.exists():
        with open(tracking_file, "r") as f:
            for line in f:
                line_str = line.strip()
                if line_str and not line_str.startswith("#"):
                    existing_entries.add(line_str)

    new_entries = []
    for item in items_to_remove:
        try:
            rel_path = str(item.relative_to(results_dir))
        except ValueError:
            rel_path = str(item)
        if rel_path not in existing_entries:
            existing_entries.add(rel_path)
            new_entries.append(rel_path)

    if new_entries:
        with open(tracking_file, "a") as f:
            for rel_path in new_entries:
                f.write(f"{rel_path}\n")
        print(f"Recorded {len(new_entries)} deleted item(s) to '{tracking_file}'.")


def erase_experiments(
    results_dir: str = "results",
    experiments: list = None,
    group: str = None,
    keep: list = None,
    tracking_file: str = DEFAULT_TRACKING_FILE,
    dry_run: bool = False,
):
    results_path = Path(results_dir)
    tracking_path = Path(tracking_file)
    if not results_path.exists():
        print(f"Error: Results directory '{results_dir}' does not exist.")
        return

    experiments = experiments or []
    keep = keep or []

    items_to_remove = []

    # Case 1: Cleaning a specific group while keeping certain experiments/patterns
    if group and keep:
        print(f"=== Purging Group '{group}' (Keeping: {', '.join(keep)}) ===")
        for cat in DEFAULT_RESULT_CATEGORIES:
            group_dir = results_path / cat / group
            if not group_dir.exists() or not group_dir.is_dir():
                continue

            for item in group_dir.iterdir():
                if should_keep_file_or_dir(item.name, keep):
                    continue
                items_to_remove.append(item)

    # Case 2: Erasing explicit experiment ID(s)
    elif experiments:
        print(f"=== Erasing Experiment(s): {', '.join(experiments)} ===")
        for exp_id in experiments:
            for cat in DEFAULT_RESULT_CATEGORIES:
                # Check results/<cat>/<group>/<exp_id> if group provided
                if group:
                    target = results_path / cat / group / exp_id
                    if target.exists():
                        items_to_remove.append(target)
                else:
                    # Check across all groups: results/<cat>/*/<exp_id>
                    cat_dir = results_path / cat
                    if cat_dir.exists():
                        for group_dir in cat_dir.iterdir():
                            if group_dir.is_dir():
                                target = group_dir / exp_id
                                if target.exists():
                                    items_to_remove.append(target)
                    # Also check un-grouped: results/<cat>/<exp_id>
                    ungrouped_target = results_path / cat / exp_id
                    if ungrouped_target.exists():
                        items_to_remove.append(ungrouped_target)

    # Case 3: Erasing an entire group with no keep rules
    elif group and not keep:
        print(f"=== Erasing ENTIRE Group '{group}' ===")
        for cat in DEFAULT_RESULT_CATEGORIES:
            group_dir = results_path / cat / group
            if group_dir.exists() and group_dir.is_dir():
                for item in group_dir.iterdir():
                    if item.name.startswith(".git"):
                        continue
                    items_to_remove.append(item)

    else:
        print("Error: Must specify either --experiment, or --group with optional --keep.")
        return

    if not items_to_remove:
        print("No matching files or directories found to delete.")
        return

    print(f"\nFound {len(items_to_remove)} items to remove:")
    for item in items_to_remove:
        item_type = "DIR " if item.is_dir() else "FILE"
        print(f"  [{item_type}] {item}")

    if dry_run:
        print("\n[DRY RUN] No files were deleted.")
        return

    print("\nDeleting target items...")
    deleted_count = 0
    successfully_removed = []
    for item in items_to_remove:
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            print(f"  DELETED: {item}")
            deleted_count += 1
            successfully_removed.append(item)
        except Exception as e:
            print(f"  ERROR deleting {item}: {e}")

    record_deleted_items(successfully_removed, results_path, tracking_path)
    print(f"\nSuccessfully erased {deleted_count} items.")


def main():
    args = parse_args()
    erase_experiments(
        results_dir=args.results_dir,
        experiments=args.experiment,
        group=args.group,
        keep=args.keep,
        tracking_file=args.tracking_file,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
