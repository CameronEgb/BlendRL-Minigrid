#!/usr/bin/env python3
import glob
import os
import re
import shutil
from pathlib import Path

def merge_move(src_path: Path, dst_path: Path):
    """Recursively moves src_path to dst_path, merging directories if dst_path exists."""
    if not src_path.exists():
        return
    if src_path.resolve() == dst_path.resolve():
        return

    if not dst_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dst_path))
        print(f"MOVED: {src_path} -> {dst_path}")
    elif src_path.is_dir() and dst_path.is_dir():
        for item in list(src_path.iterdir()):
            merge_move(item, dst_path / item.name)
        try:
            src_path.rmdir()
            print(f"REMOVED EMPTY DIR: {src_path}")
        except OSError:
            pass  # Directory not empty
    else:
        # Both are files
        if src_path.stat().st_mtime > dst_path.stat().st_mtime:
            dst_path.unlink()
            shutil.move(str(src_path), str(dst_path))
            print(f"OVERWROTE (newer src): {src_path} -> {dst_path}")
        else:
            src_path.unlink()
            print(f"DISCARDED (older src): {src_path}")

def parse_experiment_configs(conf_dir="conf/experiment"):
    exp_to_group = {}
    for c in glob.glob(f"{conf_dir}/*.yaml"):
        try:
            with open(c) as f:
                content = f.read()
            exp_id_match = re.search(r'^\s*experiment_id:\s*[\"\']?([^\"\'\n#]+)[\"\']?', content, re.MULTILINE)
            group_match = re.search(r'^\s*group:\s*[\"\']?([^\"\'\n#]+)[\"\']?', content, re.MULTILINE)
            if exp_id_match and group_match:
                eid = exp_id_match.group(1).strip()
                grp = group_match.group(1).strip()
                exp_to_group[eid] = grp
        except Exception as e:
            print(f"Warning reading config {c}: {e}")
            
    # Known agent/experiment overrides
    exp_to_group.update({
        'cp_final_blendrl': 'cartpole',
        'cp_final_ppo': 'cartpole',
        'cp_final_online_ppo': 'cartpole',
        'cp_final_online_blendrl': 'cartpole',
        'cp_final_offline_iql': 'cartpole',
        'cp_final_offline_blendrl_iql': 'cartpole',
        'blendrl_final_cp': 'cartpole',
        'ppo_final_cp': 'cartpole',
        'cp_iql_tune': 'tuning',
        'iql_tune': 'tuning',
        'cp_ppo_tuning': 'tuning',
        'cp_final_tune': 'tuning',
        'tune_blendrl_v2': 'tuning',
        'cp_tune_ppo': 'tuning',
        'cp_blendrl_fast_500': 'quick_tests',
        'cp_blendrl_ultra_fast': 'quick_tests',
    })
    return exp_to_group

def main(results_dir="results"):
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: {results_dir} directory not found.")
        return

    exp_to_group = parse_experiment_configs()

    legacy_group_map = {
        'sepsis': 'mimic',
        'blendrl_tuning': 'tuning',
        'final_results': 'cartpole',
        'final_demo': 'cartpole',
        'final_cartpole': 'cartpole',
        'cew': 'cew',
        'verification': 'cew',
        'multi_module_tests': 'cartpole',
        'quick_test': 'quick_tests',
        'combined': 'mimic'
    }

    # Root folders in results/datasets where the group directory was named after experiment_id
    exp_as_group_map = {
        'cp_final_tune': ('tuning', 'tune_cp_final'),
        'tune_human_cew': ('tuning', 'tune_human_cew'),
        'tune_human_cew_poc': ('tuning', 'tune_human_cew_poc'),
        'final_cartpole': ('cartpole', 'final_cartpole')
    }

    root_types = ['logs', 'datasets', 'checkpoints', 'plots', 'experiments', 'tensorboard']

    print("=== Reorganizing Results Tree to Match Experiment Configs ===")

    for t in root_types:
        base = results_path / t
        if not base.exists():
            continue

        print(f"\nProcessing '{base}'...")

        for group_dir in list(base.iterdir()):
            if not group_dir.is_dir() or group_dir.name == "archive":
                continue

            old_group = group_dir.name

            # 1. Handle slurm directory inside logs/slurm/
            if old_group == 'slurm':
                for s_dir in list(group_dir.iterdir()):
                    if not s_dir.is_dir():
                        continue
                    s_name = s_dir.name
                    new_g = legacy_group_map.get(s_name, s_name)
                    if new_g != s_name:
                        dest = base / 'slurm' / new_g
                        merge_move(s_dir, dest)
                continue

            # 2. Handle cases where top-level group folder was actually an experiment ID (e.g. cp_final_tune)
            if old_group in exp_as_group_map:
                target_group, exp_id = exp_as_group_map[old_group]
                dest = base / target_group / exp_id
                merge_move(group_dir, dest)
                continue

            # 3. Process experiment subdirectories
            for exp_dir in list(group_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue

                exp_id = exp_dir.name
                
                # Determine new target group
                if exp_id in exp_to_group:
                    target_group = exp_to_group[exp_id]
                elif old_group in legacy_group_map:
                    target_group = legacy_group_map[old_group]
                else:
                    target_group = old_group

                dest = base / target_group / exp_id
                if exp_dir.resolve() != dest.resolve():
                    merge_move(exp_dir, dest)

            # Try removing old group directory if empty
            try:
                if not any(group_dir.iterdir()):
                    group_dir.rmdir()
                    print(f"REMOVED EMPTY GROUP DIR: {group_dir}")
            except OSError:
                pass

    print("\n=== Reorganization Complete ===")

if __name__ == "__main__":
    main()
