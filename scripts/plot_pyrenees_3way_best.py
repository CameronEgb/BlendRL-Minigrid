#!/usr/bin/env python3
"""
scripts/plot_pyrenees_3way_best.py — Plot best Optuna trial runs across all 3 methods and 11 problems.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METHODS = [
    ("cql_blendrl_human_neural", "CQL (Neural Baseline)", "#e74c3c", "--"),
    ("cql_blendrl_human_neural_logic", "BlendRL (Human Logic + Neural)", "#2980b9", "-"),
    ("cql_blendrl_human_dueling_resnet_logic", "BlendRL (Dueling ResNet + Logic)", "#27ae60", "-."),
]

PROBLEMS = [
    "problem",
    "ex132_w",
    "ex132a_w",
    "ex152a_w",
    "ex212_w",
    "ex242_w",
    "ex252_w",
    "ex252a_w",
    "exc137_w",
    "exp426d_w",
    "exp426e_w",
]

EXP_ID = "tune_pyrenees_3way"
LOG_DIR = PROJECT_ROOT / "results" / "logs" / "pyrenees" / EXP_ID
OUT_DIR = PROJECT_ROOT / "results" / "plots" / "pyrenees" / EXP_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_best_trial_data(method_prefix, problem_id):
    """Locate the best trial (lowest final val loss) for a method and problem."""
    agent_dir_name = f"{method_prefix}_{problem_id}"
    agent_dir = LOG_DIR / agent_dir_name
    
    if not agent_dir.exists():
        return None
        
    versions = sorted(agent_dir.glob("version_*"), key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else 0)
    if not versions:
        return None
        
    best_df = None
    best_loss = float("inf")
    best_ver = None
    
    for v in versions:
        metrics_file = v / "metrics.csv"
        if not metrics_file.exists():
            continue
        try:
            df = pd.read_csv(metrics_file)
            if "val/loss" in df.columns:
                v_series = df["val/loss"].dropna()
                if len(v_series) > 0:
                    min_val = v_series.min()
                    if min_val < best_loss:
                        best_loss = min_val
                        best_df = df
                        best_ver = v.name
        except Exception:
            pass
            
    if best_df is None and versions:
        # Fallback to latest
        latest_file = versions[-1] / "metrics.csv"
        if latest_file.exists():
            try:
                best_df = pd.read_csv(latest_file)
                best_ver = versions[-1].name
            except Exception:
                pass
                
    return {
        "df": best_df,
        "best_ver": best_ver,
        "best_loss": best_loss if best_loss != float("inf") else None,
    }


def main():
    print("=" * 80)
    print("      EXTRACTING & PLOTTING BEST OPTUNA TRIALS ACROSS 33 MODELS")
    print("=" * 80)

    summary_records = []

    for prob in PROBLEMS:
        for m_prefix, m_label, _, _ in METHODS:
            res = find_best_trial_data(m_prefix, prob)
            if res and res["df"] is not None:
                df = res["df"]
                final_val = None
                final_bellman = None
                final_cql = None
                
                if "val/loss" in df.columns:
                    s = df["val/loss"].dropna()
                    if len(s) > 0: final_val = float(s.iloc[-1])
                if "losses/bellman_loss" in df.columns:
                    s = df["losses/bellman_loss"].dropna()
                    if len(s) > 0: final_bellman = float(s.iloc[-1])
                if "losses/cql_loss" in df.columns:
                    s = df["losses/cql_loss"].dropna()
                    if len(s) > 0: final_cql = float(s.iloc[-1])
                    
                steps = int(df["step"].dropna().max()) if "step" in df.columns else None
                
                summary_records.append({
                    "Problem": prob,
                    "Method": m_label,
                    "Method_Key": m_prefix,
                    "Best_Trial": res["best_ver"],
                    "Final_Val_Loss": final_val,
                    "Final_Bellman_Loss": final_bellman,
                    "Final_CQL_Loss": final_cql,
                    "Steps": steps,
                })

    df_summary = pd.DataFrame(summary_records)
    summary_csv = OUT_DIR / "3way_best_runs_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV -> {summary_csv}")

    # ── Figure: Multi-Panel Publication-Quality Comparison ────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor="#fafafa")
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    # Panel 1: Bar Chart of Final Validation Loss across all 11 problems
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#f5f5f5")
    
    x = np.arange(len(PROBLEMS))
    width = 0.26
    
    for i, (m_prefix, m_label, color, _) in enumerate(METHODS):
        vals = []
        for prob in PROBLEMS:
            row = df_summary[(df_summary["Problem"] == prob) & (df_summary["Method_Key"] == m_prefix)]
            if len(row) > 0 and row["Final_Val_Loss"].values[0] is not None:
                vals.append(row["Final_Val_Loss"].values[0])
            else:
                vals.append(np.nan)
        bars = ax1.bar(x + (i - 1) * width, vals, width, label=m_label, color=color, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    clean_prob_names = [p.replace("_w", "(w)") for p in PROBLEMS]
    ax1.set_xticks(x)
    ax1.set_xticklabels(clean_prob_names, fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Validation Loss (Lower is Better)", fontsize=11, fontweight="bold")
    ax1.set_title("Performance Comparison Across All 11 Pyrenees Problem Models (Best Optuna Trials)", fontsize=12.5, fontweight="bold")
    ax1.legend(fontsize=10.5, framealpha=0.95, loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax1.set_ylim(0, max(df_summary["Final_Val_Loss"].dropna().max() * 1.15, 2.5))

    # Panel 2: Learning Curves for Problem-Level Policy
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#f5f5f5")
    for m_prefix, m_label, color, ls in METHODS:
        res = find_best_trial_data(m_prefix, "problem")
        if res and res["df"] is not None:
            df = res["df"]
            if "step" in df.columns and "val/loss" in df.columns:
                sub = df[["step", "val/loss"]].dropna()
                if len(sub) > 0:
                    ax2.plot(sub["step"], sub["val/loss"], label=m_label, color=color, linestyle=ls, linewidth=2)
    ax2.set_xlabel("Training Steps", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Validation Loss", fontsize=10.5, fontweight="bold")
    ax2.set_title("Problem-Level Policy Convergence (problem.csv)", fontsize=11.5, fontweight="bold")
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Panel 3: Learning Curves for Exercise Step Model (ex132(w))
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#f5f5f5")
    for m_prefix, m_label, color, ls in METHODS:
        res = find_best_trial_data(m_prefix, "ex132_w")
        if res and res["df"] is not None:
            df = res["df"]
            if "step" in df.columns and "val/loss" in df.columns:
                sub = df[["step", "val/loss"]].dropna()
                if len(sub) > 0:
                    ax3.plot(sub["step"], sub["val/loss"], label=m_label, color=color, linestyle=ls, linewidth=2)
    ax3.set_xlabel("Training Steps", fontsize=10.5, fontweight="bold")
    ax3.set_ylabel("Validation Loss", fontsize=10.5, fontweight="bold")
    ax3.set_title("Step-Level Policy Convergence (ex132(w).csv)", fontsize=11.5, fontweight="bold")
    ax3.legend(fontsize=9, framealpha=0.9)
    ax3.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Pyrenees ITS: 3-Way Offline Model Evaluation (Neural vs. Human Logic vs. Dueling ResNet)", fontsize=14, fontweight="bold", y=0.98)
    
    out_fig = OUT_DIR / "3way_comparison_best_runs.png"
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved publication comparison figure -> {out_fig}")
    print("\nPlotting complete!")


if __name__ == "__main__":
    main()
