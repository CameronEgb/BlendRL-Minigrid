import os
import shutil
import glob
from pathlib import Path

# Standard experiment groups defined in NeSyRL
STANDARD_GROUPS = [
    "pyrenees",
    "mimic",
    "early_prediction",
    "tuning",
    "quick_tests",
    "ungrouped",
    "cew",
    "sepsis",
    "verification",
    "final_demo",
    "blendrl_tuning",
    "multi_module_tests"
]

SUBDIRS = ["logs", "checkpoints", "plots", "tensorboard", "experiments"]

def fix_results_structure(results_dir="results"):
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: {results_dir} directory not found.")
        return

    archive_dir = results_path / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Cleaning and Structuring '{results_dir}' ===")

    for sub in SUBDIRS:
        sub_path = results_path / sub
        if not sub_path.exists():
            sub_path.mkdir(parents=True, exist_ok=True)
            continue

        print(f"\nProcessing '{sub_path}'...")
        
        # Ensure all standard group folders exist
        for group in STANDARD_GROUPS:
            (sub_path / group).mkdir(parents=True, exist_ok=True)

        # Move loose/legacy item folders that don't belong to a standard group into archive
        for item in list(sub_path.iterdir()):
            if item.is_dir():
                if item.name not in STANDARD_GROUPS and item.name != "slurm":
                    print(f"  -> Archiving un-grouped folder: {item}")
                    target = archive_dir / f"{sub}_{item.name}"
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.move(str(item), str(target))
            elif item.is_file():
                # Move loose root files in logs/checkpoints/plots/etc to archive
                if item.name not in [".DS_Store"]:
                    print(f"  -> Archiving loose file: {item}")
                    target = archive_dir / f"{sub}_{item.name}"
                    shutil.move(str(item), str(target))

    print("\n=== Results Structure Cleanup Complete! ===")

if __name__ == "__main__":
    fix_results_structure()
