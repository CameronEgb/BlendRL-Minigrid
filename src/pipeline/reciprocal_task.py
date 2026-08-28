"""Reciprocal Refinement: Iterative co-training of EP predictors and CQL policies.

Orchestrates the iterative loop where:
  - EP models provide reward shaping signals for CQL training
  - CQL policies provide V(s) features for EP training
  - Each round improves both models

Usage:
  Called from run_pipeline.py when task == 'reciprocal_refinement'
  or via: python -m src.pipeline.reciprocal_task --config <experiment.yaml>

Architecture:
  Round 0 (Bootstrap):
    1. Train CQL₀ with TQN reward -> checkpoint₀
    2. Train EP₀ (no V features) -> EP checkpoint₀
    
  Round k >= 1:
    1. Export V_{k-1}(s) from CQL_{k-1}
    2. Train EP_k with V_{k-1}(s) features -> EP checkpoint_k
    3. Shape rewards: r = r_TQN + λ*(γ*Φ_k(s') - Φ_k(s)) where Φ_k = -P_EP_k(shock)
    4. Train CQL_k with shaped rewards -> checkpoint_k
    5. Evaluate: counterfactual mortality, AUPRC
    6. Check convergence
"""
import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime



from src.pipeline.runtime import get_python_executable, get_subprocess_env
from src.pipeline.task_registry import register_task

@register_task("reciprocal_refinement")
def run_reciprocal_refinement_task(cfg, context):
    run_reciprocal_refinement(cfg, context.get("local_val", True))
    sys.exit(0)

def run_reciprocal_refinement(cfg, local_val):
    """Execute the reciprocal refinement iterative co-training loop.
    
    Args:
        cfg: Hydra experiment config with 'reciprocal' section
        local_val: Whether to run locally (True) or via Slurm (False)
    """
    rc = cfg.get("reciprocal", {})
    max_rounds = rc.get("max_rounds", 3)
    lambda_coef = rc.get("lambda_coef", 1.0)
    lambda_decay = rc.get("lambda_decay", 1.0)  # Multiply λ by this each round
    convergence_threshold = rc.get("convergence_threshold", 0.005)
    ep_architecture = rc.get("ep_architecture", None)  # None = use all
    gamma = cfg.env.get("gamma", 0.99)
    
    group = cfg.get("group", "mimic")
    exp_id = cfg.get("experiment_id", cfg.get("name", "reciprocal_refinement"))
    
    site_cfg = cfg.get("site", None)
    python_exe = get_python_executable(site_cfg)
    
    # Standard paths
    base_ckpt_dir = Path("results/checkpoints") / group / exp_id
    base_ep_ckpt_dir = Path("results/checkpoints/early_prediction")
    base_results_dir = Path("results/plots") / group / exp_id
    convergence_log = base_results_dir / "convergence_log.json"
    
    base_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Convergence tracking
    round_metrics = []
    
    print(f"\n{'='*70}")
    print(f"  RECIPROCAL REFINEMENT: Iterative EP ↔ CQL Co-Training")
    print(f"  Experiment: {group}/{exp_id}")
    print(f"  Max Rounds: {max_rounds}, λ₀={lambda_coef}, decay={lambda_decay}")
    print(f"  Convergence threshold: {convergence_threshold}")
    print(f"{'='*70}\n")
    
    for round_k in range(max_rounds + 1):  # Round 0 is bootstrap
        round_lambda = lambda_coef * (lambda_decay ** max(0, round_k - 1)) if round_k > 0 else 0.0
        round_exp_id = f"{exp_id}_round{round_k}"
        round_ckpt_dir = base_ckpt_dir / f"round{round_k}"
        round_ep_ckpt_dir = base_ep_ckpt_dir / f"{exp_id}_round{round_k}"
        round_results_dir = base_results_dir / f"round{round_k}"
        round_results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"  ROUND {round_k} {'(Bootstrap)' if round_k == 0 else f'(λ={round_lambda:.4f})'}")
        print(f"{'='*60}")
        
        # ==================================================================
        # Step 1: Train CQL (with EP-shaped rewards if round > 0)
        # ==================================================================
        print(f"\n--- Round {round_k}: Step 1 — Train CQL ---")
        
        env_vars = get_subprocess_env(site_cfg)
        env_vars["BLENDRL_ENV_NAME"] = "mimic"
        
        if round_k == 0:
            # Bootstrap: standard TQN reward
            env_vars["MIMIC_REWARD_TYPE"] = "tqn"
        else:
            # Use EP-shaped reward
            env_vars["MIMIC_REWARD_TYPE"] = "ep_shaped"
            env_vars["EP_SHAPE_CKPT_DIR"] = str(round_ep_ckpt_dir.parent / f"{exp_id}_round{round_k - 1}")
            env_vars["EP_SHAPE_LAMBDA"] = str(round_lambda)
            env_vars["EP_SHAPE_GAMMA"] = str(gamma)
            
            # Point to previous round's CQL for V(s) computation
            prev_ckpt_dir = base_ckpt_dir / f"round{round_k - 1}"
            env_vars["EP_SHAPE_CQL_CKPT"] = str(prev_ckpt_dir)
        
        # Build CQL training command
        cql_methods = cfg.get("offline_methods", "cql/dnn")
        if isinstance(cql_methods, (list, tuple)):
            cql_methods = ",".join(cql_methods)
        
        cql_cmd = [
            python_exe, "-u", "src/train.py",
            f"experiment={cfg.get('experiment_name', '')}",
            f"experiment_id={round_exp_id}",
            "mode=offline",
            f"agent.name={cql_methods.split(',')[0]}",
        ]
        
        # Add any extra hydra overrides from the config
        cql_overrides = rc.get("cql_overrides", [])
        if isinstance(cql_overrides, (list, tuple)):
            cql_cmd.extend(cql_overrides)
        
        print(f"  CMD: {' '.join(cql_cmd)}")
        
        if local_val:
            result = subprocess.run(cql_cmd, env=env_vars)
            if result.returncode != 0:
                print(f"  ERROR: CQL training failed in round {round_k} (exit code {result.returncode})")
                break
        else:
            print(f"  [Slurm mode not yet implemented for reciprocal refinement]")
            return
        
        # Copy best checkpoint to round-specific directory
        _copy_best_checkpoint(cfg, round_exp_id, round_ckpt_dir)
        
        # ==================================================================
        # Step 2: Train EP (with V(s) from this round's CQL)
        # ==================================================================
        print(f"\n--- Round {round_k}: Step 2 — Train EP Models ---")
        
        ep_cmd = [
            python_exe, "-u", "src/early_prediction/model.py",
            "--exp-id", f"{exp_id}_round{round_k}",
            "--checkpoint", str(round_ckpt_dir),
            "--save-checkpoints",
        ]
        
        # Add EP-specific args from config
        ep_cfg = rc.get("early_prediction", {})
        if ep_cfg.get("n_splits"):
            ep_cmd.extend(["--n-splits", str(ep_cfg["n_splits"])])
        if ep_cfg.get("epochs"):
            ep_cmd.extend(["--epochs", str(ep_cfg["epochs"])])
        if ep_cfg.get("tau_train"):
            ep_cmd.extend(["--tau-train", str(ep_cfg["tau_train"])])
        
        print(f"  CMD: {' '.join(ep_cmd)}")
        
        if local_val:
            result = subprocess.run(ep_cmd, env=env_vars)
            if result.returncode != 0:
                print(f"  WARNING: EP training had issues in round {round_k} (exit code {result.returncode})")
        
        # ==================================================================
        # Step 3: Evaluate this round
        # ==================================================================
        print(f"\n--- Round {round_k}: Step 3 — Evaluate ---")
        
        if local_val:
            from src.early_prediction.eval_logic import (
                compute_ep_eval_data,
                plot_agreement_vs_shock,
                plot_agreement_vs_shock_deciles,
                plot_ep_shock_over_tau,
                write_counterfactual_table
            )
            try:
                eval_results = compute_ep_eval_data(
                    checkpoint_root=str(round_ckpt_dir),
                    dataset_path=None,
                    ep_ckpt_root=str(round_ep_ckpt_dir)
                )
                
                plot_agreement_vs_shock(eval_results["rl_agreements"], eval_results["y"], round_results_dir)
                plot_agreement_vs_shock_deciles(eval_results["rl_agreements"], eval_results["y"], round_results_dir)
                plot_ep_shock_over_tau(eval_results["ep_shock_results"], round_results_dir)
                write_counterfactual_table(
                    eval_results["cf_data"], 
                    round_results_dir / "counterfactual_summary.csv", 
                    round_results_dir / "counterfactual_summary.txt"
                )
            except Exception as e:
                print(f"  WARNING: Evaluation had issues in round {round_k}: {e}")
        
        # ==================================================================
        # Step 4: Collect metrics and check convergence
        # ==================================================================
        metrics = _collect_round_metrics(round_results_dir, round_k)
        round_metrics.append(metrics)
        
        # Save convergence log atomically
        tmp_log = str(convergence_log) + ".tmp"
        with open(tmp_log, "w") as f:
            json.dump(round_metrics, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_log, convergence_log)
        
        print(f"\n  Round {round_k} metrics: {json.dumps(metrics, indent=2)}")
        
        # Check convergence (need at least 2 rounds to compare)
        if round_k >= 1 and len(round_metrics) >= 2:
            prev_metrics = round_metrics[-2]
            curr_metrics = round_metrics[-1]
            
            delta_mort = abs(
                curr_metrics.get("counterfactual_mortality", 0) -
                prev_metrics.get("counterfactual_mortality", 0)
            )
            
            if delta_mort < convergence_threshold:
                print(f"\n  *** CONVERGED at round {round_k}: "
                      f"Δ(mortality) = {delta_mort:.6f} < {convergence_threshold} ***")
                break
            else:
                print(f"  Δ(mortality) = {delta_mort:.6f} (threshold: {convergence_threshold})")
    
    # ==================================================================
    # Final: Generate convergence plots
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  Generating convergence summary plots...")
    print(f"{'='*60}")
    
    _plot_convergence(round_metrics, base_results_dir)
    
    print(f"\nReciprocal Refinement complete! Results in: {base_results_dir}")
    print(f"Convergence log: {convergence_log}")


def _copy_best_checkpoint(cfg, round_exp_id, dest_dir):
    """Copy the best model checkpoint from training logs to the round checkpoint dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    group = cfg.get("group", "mimic")
    
    # Search for checkpoints in standard locations
    search_dirs = [
        Path("results/checkpoints") / group / round_exp_id,
        Path("results/logs") / group / round_exp_id,
    ]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            ckpts = sorted(search_dir.rglob("best_model*.ckpt"), 
                          key=lambda p: p.stat().st_mtime, reverse=True)
            if not ckpts:
                ckpts = sorted(search_dir.rglob("*.ckpt"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
            if ckpts:
                dest_path = dest_dir / ckpts[0].name
                if not dest_path.exists():
                    shutil.copy2(ckpts[0], dest_path)
                print(f"  Checkpoint saved: {dest_path}")
                return
    
    print(f"  WARNING: No checkpoint found for {round_exp_id}")


def _collect_round_metrics(results_dir, round_k):
    """Collect evaluation metrics from a round's results directory."""
    metrics = {"round": round_k, "timestamp": datetime.now().isoformat()}
    
    # Try to read counterfactual summary
    cf_csv = results_dir / "counterfactual_summary.csv"
    if cf_csv.exists():
        try:
            import csv
            with open(cf_csv) as f:
                reader_csv = csv.DictReader(f)
                for row in reader_csv:
                    method = row.get("method", "")
                    if method and method != "clinician":
                        metrics["counterfactual_mortality"] = float(row.get("pred_mortality_mean", 0))
                        metrics["agreement"] = float(row.get("agreement_mean", 0))
                        metrics["f1"] = float(row.get("f1_mean", 0))
                        break
        except Exception as e:
            print(f"  WARNING: Could not parse counterfactual metrics: {e}")
    
    # Try to read EP metrics
    for json_file in results_dir.glob("metrics_*.json"):
        try:
            with open(json_file) as f:
                ep_data = json.load(f)
                if ep_data.get("auc"):
                    key = json_file.stem.replace("metrics_", "")
                    metrics[f"ep_{key}_auc_mean"] = float(sum(ep_data["auc"]) / len(ep_data["auc"]))
                    if ep_data.get("auprc"):
                        metrics[f"ep_{key}_auprc_mean"] = float(sum(ep_data["auprc"]) / len(ep_data["auprc"]))
        except Exception:
            pass
    
    return metrics


def _plot_convergence(round_metrics, output_dir):
    """Generate convergence plots showing metric improvement across rounds."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  WARNING: matplotlib not available, skipping convergence plots.")
        return
    
    if len(round_metrics) < 2:
        print("  Not enough rounds for convergence plot.")
        return
    
    rounds = [m["round"] for m in round_metrics]
    
    # Collect available metric series
    metric_series = {}
    for key in ["counterfactual_mortality", "agreement", "f1"]:
        vals = [m.get(key) for m in round_metrics]
        if any(v is not None for v in vals):
            metric_series[key] = [v if v is not None else float('nan') for v in vals]
    
    # Also collect EP metrics
    ep_keys = set()
    for m in round_metrics:
        for k in m:
            if k.startswith("ep_") and k.endswith("_auprc_mean"):
                ep_keys.add(k)
    for key in sorted(ep_keys):
        vals = [m.get(key) for m in round_metrics]
        if any(v is not None for v in vals):
            metric_series[key] = [v if v is not None else float('nan') for v in vals]
    
    if not metric_series:
        print("  No metrics available for convergence plot.")
        return
    
    n_plots = len(metric_series)
    cols = min(n_plots, 3)
    rows = (n_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    
    for idx, (key, vals) in enumerate(metric_series.items()):
        ax = axes[idx // cols][idx % cols]
        ax.plot(rounds, vals, 'o-', linewidth=2, markersize=8, color='tab:blue')
        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel(key.replace("_", " ").title(), fontsize=12)
        ax.set_title(key.replace("_", " ").title(), fontsize=13, fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks(rounds)
    
    # Hide unused subplots
    for idx in range(n_plots, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    
    plt.suptitle("Reciprocal Refinement — Convergence Across Rounds",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plot_path = output_dir / "convergence.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved convergence plot: {plot_path}")
