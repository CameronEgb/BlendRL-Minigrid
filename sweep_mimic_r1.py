#!/usr/bin/env python3
"""
MIMIC HP Sweep — Round 1
========================
Runs 14 configurations (6 DNN + 8 BlendRL) with the blend entropy fix,
then analyzes val/loss + blend_entropy from metrics.csv.

Estimated time: ~35 minutes (6×45s DNN + 8×3.5min BlendRL).
"""
import subprocess
import sys
import os
import csv
import json
import time
from pathlib import Path

PROJECT_ROOT = "/Users/cameronegbert/Documents/NCSU/Research/NeSyRL"
EXPERIMENT = "mimic/mimic_test"
EXPERIMENT_ID = "mimic_hp_sweep"  # Separate from mimic_test to avoid pollution
DATASET_PATH = "in/datasets/mimic/mimic_lazy_0_interventions_balanced"
LOG_ROOT = Path(PROJECT_ROOT) / "results" / "logs" / "mimic" / EXPERIMENT_ID

# Detect python executable
PYTHON = None
for candidate in [
    os.path.join(PROJECT_ROOT, ".venv/bin/python"),
    os.path.join(PROJECT_ROOT, "venv/bin/python"),
    sys.executable,
]:
    if os.path.exists(candidate):
        PYTHON = candidate
        break

# ──────────────────────────────────────────────
#  Sweep Configurations
# ──────────────────────────────────────────────

DNN_CONFIGS = [
    # (name, overrides_dict)
    ("dnn_e10_base",      {"epochs_per_interval": 10, "lr": 3e-4, "cql_alpha": 1.0, "weight_decay": 1e-4, "batch_size": 256}),
    ("dnn_e10_cql2",      {"epochs_per_interval": 10, "lr": 3e-4, "cql_alpha": 2.0, "weight_decay": 1e-4, "batch_size": 256}),
    ("dnn_e10_cql5",      {"epochs_per_interval": 10, "lr": 3e-4, "cql_alpha": 5.0, "weight_decay": 1e-4, "batch_size": 256}),
    ("dnn_e10_lr1e4",     {"epochs_per_interval": 10, "lr": 1e-4, "cql_alpha": 2.0, "weight_decay": 1e-4, "batch_size": 256}),
    ("dnn_e20_cql2",      {"epochs_per_interval": 20, "lr": 3e-4, "cql_alpha": 2.0, "weight_decay": 1e-4, "batch_size": 256}),
    ("dnn_e20_cql2_wd",   {"epochs_per_interval": 20, "lr": 3e-4, "cql_alpha": 2.0, "weight_decay": 1e-3, "batch_size": 256}),
]

BLEND_CONFIGS = [
    # (name, overrides_dict)  — all inherit cql/blendrl_human_neural base
    ("blend_e10_bec01",       {"epochs_per_interval": 10, "lr": 3e-4, "cql_alpha": 1.0, "blend_ent_coef": 0.01, "ent_coef": 0.05, "weight_decay": 1e-4}),
    ("blend_e10_bec05",       {"epochs_per_interval": 10, "lr": 3e-4, "cql_alpha": 1.0, "blend_ent_coef": 0.05, "ent_coef": 0.05, "weight_decay": 1e-4}),
    ("blend_e10_bec10",       {"epochs_per_interval": 10, "lr": 3e-4, "cql_alpha": 1.0, "blend_ent_coef": 0.1,  "ent_coef": 0.05, "weight_decay": 1e-4}),
    ("blend_e10_bec20",       {"epochs_per_interval": 10, "lr": 3e-4, "cql_alpha": 1.0, "blend_ent_coef": 0.2,  "ent_coef": 0.05, "weight_decay": 1e-4}),
    ("blend_e10_bec10_cql5",  {"epochs_per_interval": 10, "lr": 3e-4, "cql_alpha": 5.0, "blend_ent_coef": 0.1,  "ent_coef": 0.05, "weight_decay": 1e-4}),
    ("blend_e10_bec10_lr1e4", {"epochs_per_interval": 10, "lr": 1e-4, "cql_alpha": 2.0, "blend_ent_coef": 0.1,  "ent_coef": 0.05, "weight_decay": 1e-4}),
    ("blend_e20_bec10",       {"epochs_per_interval": 20, "lr": 3e-4, "cql_alpha": 2.0, "blend_ent_coef": 0.1,  "ent_coef": 0.05, "weight_decay": 1e-4}),
    ("blend_e10_bec10_gumbel",{"epochs_per_interval": 10, "lr": 3e-4, "cql_alpha": 1.0, "blend_ent_coef": 0.1,  "ent_coef": 0.05, "weight_decay": 1e-4, "blend_function": "gumbel"}),
]


def build_cmd(agent_config, agent_name, overrides):
    """Build the Hydra CLI command for src/train.py."""
    cmd = [
        PYTHON, "src/train.py",
        f"+experiment={EXPERIMENT}",
        f"++experiment_id={EXPERIMENT_ID}",
        "++local=true",
        "mode=offline",
        f"agent={agent_config}",
        f"++agent.name={agent_name}",
        f"++mode.dataset_path={DATASET_PATH}",
        "++agent.eval_interval_epochs=1",
    ]
    for key, val in overrides.items():
        cmd.append(f"++agent.{key}={val}")
    return cmd


def run_single(agent_config, name, overrides):
    """Run a single training experiment and return elapsed time."""
    cmd = build_cmd(agent_config, name, overrides)
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        os.path.join(PROJECT_ROOT, "src") + ":" +
        os.path.join(PROJECT_ROOT, "src", "fyd_repo", "src") + ":" +
        env.get("PYTHONPATH", "")
    )
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    print(f"\n{'='*60}")
    print(f"  RUNNING: {name}")
    print(f"  Config: {agent_config}")
    print(f"  Overrides: {overrides}")
    print(f"{'='*60}")
    
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, 
                                capture_output=True, text=True, timeout=600)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"  FAILED ({elapsed:.1f}s)")
            print(f"  stderr (last 500 chars): {result.stderr[-500:]}")
            return elapsed, False
        print(f"  DONE ({elapsed:.1f}s)")
        return elapsed, True
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  TIMEOUT ({elapsed:.1f}s)")
        return elapsed, False


def find_latest_version(agent_name):
    """Find the latest version directory for an agent."""
    agent_dir = LOG_ROOT / agent_name
    if not agent_dir.exists():
        return None
    versions = sorted(agent_dir.glob("version_*"), 
                      key=lambda p: int(p.name.split("_")[1]))
    return versions[-1] if versions else None


def parse_metrics(version_dir):
    """Parse the metrics.csv file and extract key final metrics."""
    metrics_file = version_dir / "metrics.csv"
    if not metrics_file.exists():
        return {}
    
    results = {}
    with open(metrics_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        return {}
    
    # Get final validation metrics (from the last row that has val/loss)
    val_rows = [r for r in rows if r.get("val/loss", "")]
    if val_rows:
        last_val = val_rows[-1]
        results["val/loss"] = float(last_val["val/loss"])
        if "val/bellman_loss" in last_val and last_val["val/bellman_loss"]:
            results["val/bellman_loss"] = float(last_val["val/bellman_loss"])
        if "val/cql_loss" in last_val and last_val["val/cql_loss"]:
            results["val/cql_loss"] = float(last_val["val/cql_loss"])
        if "val/q_mean" in last_val and last_val["val/q_mean"]:
            results["val/q_mean"] = float(last_val["val/q_mean"])
    
    # Get first and last val/loss for overfitting detection
    if len(val_rows) >= 2:
        first_val = float(val_rows[0]["val/loss"])
        last_val_loss = float(val_rows[-1]["val/loss"])
        min_val_loss = min(float(r["val/loss"]) for r in val_rows)
        results["val/loss_first"] = first_val
        results["val/loss_min"] = min_val_loss
        results["val/loss_delta"] = last_val_loss - min_val_loss  # >0 means overfitting
    
    # Get final blend_entropy (from the last training row that has it)
    blend_rows = [r for r in rows if r.get("losses/blend_entropy", "")]
    if blend_rows:
        first_be = float(blend_rows[0]["losses/blend_entropy"])
        last_be = float(blend_rows[-1]["losses/blend_entropy"])
        results["blend_entropy_first"] = first_be
        results["blend_entropy_final"] = last_be
    
    # Get final training losses
    train_rows = [r for r in rows if r.get("losses/total_loss", "")]
    if train_rows:
        results["train/total_loss_final"] = float(train_rows[-1]["losses/total_loss"])
        results["train/bellman_loss_final"] = float(train_rows[-1].get("losses/bellman_loss", 0))
    
    return results


def analyze_results(run_results):
    """Print a sorted summary table of all results."""
    print("\n" + "="*120)
    print("  SWEEP RESULTS SUMMARY (sorted by val/loss)")
    print("="*120)
    
    # Separate DNN and BlendRL results
    dnn_results = {k: v for k, v in run_results.items() if k.startswith("dnn_")}
    blend_results = {k: v for k, v in run_results.items() if k.startswith("blend_")}
    
    for label, results in [("CQL DNN (Pure Neural)", dnn_results), ("BlendRL (Hybrid)", blend_results)]:
        print(f"\n---")
        print(f"  {label}")
        print(f"---")
        
        # Sort by val/loss
        sorted_items = sorted(results.items(), 
                              key=lambda x: x[1].get("metrics", {}).get("val/loss", 999))
        
        header = f"{'Name':<30} {'val/loss':>10} {'val/bell':>10} {'val/cql':>10} {'q_mean':>10} {'overfit_d':>10} {'time':>8}"
        if "blend" in label.lower():
            header += f" {'be_final':>10} {'be_first':>10}"
        print(header)
        print("-" * len(header))
        
        for name, info in sorted_items:
            m = info.get("metrics", {})
            vl = f"{m.get('val/loss', 'N/A'):>10.4f}" if isinstance(m.get('val/loss'), (int, float)) else f"{'N/A':>10}"
            vb = f"{m.get('val/bellman_loss', 'N/A'):>10.4f}" if isinstance(m.get('val/bellman_loss'), (int, float)) else f"{'N/A':>10}"
            vc = f"{m.get('val/cql_loss', 'N/A'):>10.4f}" if isinstance(m.get('val/cql_loss'), (int, float)) else f"{'N/A':>10}"
            qm = f"{m.get('val/q_mean', 'N/A'):>10.4f}" if isinstance(m.get('val/q_mean'), (int, float)) else f"{'N/A':>10}"
            delta = m.get("val/loss_delta", None)
            overfit = f"{delta:>10.4f}" if isinstance(delta, (int, float)) else f"{'N/A':>10}"
            elapsed = f"{info.get('time', 0):>7.1f}s"
            
            row = f"{name:<30} {vl} {vb} {vc} {qm} {overfit} {elapsed}"
            if "blend" in label.lower():
                be_f = f"{m.get('blend_entropy_final', 'N/A'):>10.4f}" if isinstance(m.get('blend_entropy_final'), (int, float)) else f"{'N/A':>10}"
                be_i = f"{m.get('blend_entropy_first', 'N/A'):>10.4f}" if isinstance(m.get('blend_entropy_first'), (int, float)) else f"{'N/A':>10}"
                row += f" {be_f} {be_i}"
            print(row)
    
    print(f"\n{'='*120}")


def main():
    total_start = time.time()
    run_results = {}
    
    # Phase 1: Run DNN configs
    print("\n" + "#"*60)
    print("  PHASE 1: CQL DNN (Pure Neural) Configs")
    print("#"*60)
    
    for name, overrides in DNN_CONFIGS:
        elapsed, success = run_single("cql/dnn", name, overrides)
        metrics = {}
        if success:
            version_dir = find_latest_version(name)
            if version_dir:
                metrics = parse_metrics(version_dir)
        run_results[name] = {"time": elapsed, "success": success, "metrics": metrics, "overrides": overrides}
    
    # Phase 2: Run BlendRL configs
    print("\n" + "#"*60)
    print("  PHASE 2: BlendRL (Hybrid) Configs")
    print("#"*60)
    
    for name, overrides in BLEND_CONFIGS:
        elapsed, success = run_single("cql/blendrl_human_neural", name, overrides)
        metrics = {}
        if success:
            version_dir = find_latest_version(name)
            if version_dir:
                metrics = parse_metrics(version_dir)
        run_results[name] = {"time": elapsed, "success": success, "metrics": metrics, "overrides": overrides}
    
    # Phase 3: Analyze
    analyze_results(run_results)
    
    total_elapsed = time.time() - total_start
    print(f"\nTotal sweep time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    
    # Save results to JSON for later analysis
    results_file = LOG_ROOT / "sweep_summary.json"
    os.makedirs(LOG_ROOT, exist_ok=True)
    
    # Make JSON-serializable
    serializable = {}
    for k, v in run_results.items():
        serializable[k] = {
            "time": v["time"],
            "success": v["success"],
            "metrics": v["metrics"],
            "overrides": {str(ok): str(ov) for ok, ov in v["overrides"].items()}
        }
    with open(results_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
