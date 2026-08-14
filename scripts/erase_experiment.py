#!/usr/bin/env python3
"""
Script to erase experiment results from the workspace and track deleted items.

Usage:
  # Erase a specific experiment across all result directories:
  python scripts/erase_experiment.py -e mimic_test

  # Erase multiple experiments:
  python scripts/erase_experiment.py -e exp1 exp2 exp3

  # Erase all experiments in a group except specified ones to keep:
  python scripts/erase_experiment.py -g early_prediction -k weighted_pr

  # Also delete the YAML config in in/config/experiment/:
  python scripts/erase_experiment.py -e mimic_test --delete-config

  # Dry run to preview what would be deleted without deleting anything:
  python scripts/erase_experiment.py -e mimic_test --dry-run
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
        help="Group name (e.g. early_prediction, cartpole, mimic, tuning, quick_tests).",
    )
    parser.add_argument(
        "-k",
        "--keep",
        nargs="+",
        help="Experiment ID(s) or pattern(s) to keep when cleaning a group.",
    )
    parser.add_argument(
        "--delete-config",
        "-c",
        action="store_true",
        help="Also delete the YAML config file from in/config/experiment/.",
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
    """Expand keep patterns with common aliases and variations."""
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
    delete_config: bool = False,
    tracking_file: str = DEFAULT_TRACKING_FILE,
    dry_run: bool = False,
):
    results_path = Path(results_dir)
    tracking_path = Path(tracking_file)

    experiments = experiments or []
    keep = keep or []
    items_to_remove = set()

    # Case 1: Cleaning a specific group while keeping certain experiments/patterns
    if group and keep:
        print(f"=== Purging Group '{group}' (Keeping: {', '.join(keep)}) ===")
        for cat in DEFAULT_RESULT_CATEGORIES:
            group_dir = results_path / cat / group
            if group_dir.exists() and group_dir.is_dir():
                for item in group_dir.iterdir():
                    if not should_keep_file_or_dir(item.name, keep):
                        items_to_remove.add(item)
        # Check slurm logs
        slurm_group_dir = results_path / "logs" / "slurm" / group
        if slurm_group_dir.exists() and slurm_group_dir.is_dir():
            for item in slurm_group_dir.iterdir():
                if not should_keep_file_or_dir(item.name, keep):
                    items_to_remove.add(item)
        # Check in/datasets
        in_ds_group = Path("in/datasets") / group
        if in_ds_group.exists() and in_ds_group.is_dir():
            for item in in_ds_group.iterdir():
                if not should_keep_file_or_dir(item.name, keep):
                    items_to_remove.add(item)

    # Case 2: Erasing explicit experiment ID(s)
    elif experiments:
        print(f"=== Erasing Experiment(s): {', '.join(experiments)} ===")
        for exp_id in experiments:
            clean_id = exp_id[:-5] if exp_id.endswith(".yaml") else exp_id
            if "/" in clean_id:
                clean_id = clean_id.split("/")[-1]

            # 1. Standard result categories
            for cat in DEFAULT_RESULT_CATEGORIES:
                if group:
                    target = results_path / cat / group / clean_id
                    if target.exists():
                        items_to_remove.add(target)
                else:
                    cat_dir = results_path / cat
                    if cat_dir.exists():
                        for group_dir in cat_dir.iterdir():
                            if group_dir.is_dir():
                                target = group_dir / clean_id
                                if target.exists():
                                    items_to_remove.add(target)
                    ungrouped_target = results_path / cat / clean_id
                    if ungrouped_target.exists():
                        items_to_remove.add(ungrouped_target)

            # 2. Slurm logs: results/logs/slurm/[group]/[exp_id]
            slurm_base = results_path / "logs" / "slurm"
            if slurm_base.exists():
                if group:
                    st = slurm_base / group / clean_id
                    if st.exists():
                        items_to_remove.add(st)
                else:
                    for g in slurm_base.iterdir():
                        if g.is_dir():
                            st = g / clean_id
                            if st.exists():
                                items_to_remove.add(st)

            # 3. Generated datasets: in/datasets/[group]/[exp_id]
            in_ds_base = Path("in/datasets")
            if in_ds_base.exists():
                if group:
                    dt = in_ds_base / group / clean_id
                    if dt.exists():
                        items_to_remove.add(dt)
                else:
                    for g in in_ds_base.iterdir():
                        if g.is_dir():
                            dt = g / clean_id
                            if dt.exists():
                                items_to_remove.add(dt)

            # 4. Optuna studies and legacy databases
            optuna_dir = results_path / "optuna"
            if optuna_dir.exists():
                for db in optuna_dir.glob(f"*{clean_id}*.db"):
                    if db.name != "optuna.db":
                        items_to_remove.add(db)
                shared_db = optuna_dir / "optuna.db"
                if shared_db.exists() and not dry_run:
                    try:
                        import optuna
                        storage_url = f"sqlite:///{shared_db}"
                        for summary in optuna.get_all_study_summaries(storage=storage_url):
                            if summary.study_name == clean_id or summary.study_name.startswith(f"{clean_id}_"):
                                optuna.delete_study(study_name=summary.study_name, storage=storage_url)
                                print(f"  DELETED Optuna study '{summary.study_name}' from {shared_db}")
                    except Exception as e:
                        pass

            # 5. Optional: YAML config file in in/config/experiment/
            if delete_config:
                matches = list(Path("in/config/experiment").glob(f"**/{clean_id}.yaml"))
                for cfg_file in matches:
                    items_to_remove.add(cfg_file)

    # Case 3: Erasing an entire group with no keep rules
    elif group and not keep:
        print(f"=== Erasing ENTIRE Group '{group}' ===")
        for cat in DEFAULT_RESULT_CATEGORIES:
            group_dir = results_path / cat / group
            if group_dir.exists() and group_dir.is_dir():
                for item in group_dir.iterdir():
                    if not item.name.startswith(".git"):
                        items_to_remove.add(item)
        slurm_group_dir = results_path / "logs" / "slurm" / group
        if slurm_group_dir.exists() and slurm_group_dir.is_dir():
            for item in slurm_group_dir.iterdir():
                items_to_remove.add(item)
        in_ds_group = Path("in/datasets") / group
        if in_ds_group.exists() and in_ds_group.is_dir():
            for item in in_ds_group.iterdir():
                items_to_remove.add(item)

    else:
        print("Error: Must specify either --experiment (-e), or --group (-g) with optional --keep (-k).")
        return

    items_list = sorted(list(items_to_remove))

    if not items_list:
        print("No matching files or directories found to delete.")
        return

    print(f"\nFound {len(items_list)} item(s) to remove:")
    for item in items_list:
        item_type = "DIR " if item.is_dir() else "FILE"
        print(f"  [{item_type}] {item}")

    if dry_run:
        print("\n[DRY RUN] No files were deleted.")
        return

    print("\nDeleting target items...")
    deleted_count = 0
    successfully_removed = []
    for item in items_list:
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
    print(f"\nSuccessfully erased {deleted_count} item(s).")


def main():
    args = parse_args()
    erase_experiments(
        results_dir=args.results_dir,
        experiments=args.experiment,
        group=args.group,
        keep=args.keep,
        delete_config=args.delete_config,
        tracking_file=args.tracking_file,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
