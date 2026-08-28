#!/usr/bin/env python3
"""
plot/action_distribution.py — Pyrenees ITS Pedagogical Action Distribution Plotter.

Evaluates trained RL policies against the historical ITS tutor baseline, segmenting
decisions across 3 student competency tiers (Low, Medium, High).
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Ensure project root and src are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
src_path = os.path.join(PROJECT_ROOT, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot.base import BasePlotter, clean_label, get_canonical_method_name, get_method_aliases
from src.method_registry import get_style as get_method_style
from src.pyrenees_evaluator import PyreneesEvaluator
from plot.pyrenees_reporter import PyreneesReporter


class ActionDistributionPlotter(BasePlotter):
    def __init__(self):
        super().__init__("action_distribution")
        self.tier_names = [("Low Tier", 0), ("Med Tier", 1), ("High Tier", 2)]

    def run(self, exp_id: str, cli_overrides: Optional[dict] = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        clean_exp = Path(exp_id).stem
        self._run_pyrenees_eval(exp_id, cfg, group, clean_exp, output_dir)

    def _discover_checkpoints(self, exp_id: str, group: str, clean_exp: str):
        """Discovers all Pyrenees checkpoints (single models, multi-dataset runs, per-problem models)."""
        ckpt_root = Path("results/checkpoints") / group / clean_exp
        if not ckpt_root.exists():
            ckpt_root = Path("results/checkpoints") / clean_exp
        if not ckpt_root.exists():
            return {}

        exp_cfg = self.get_experiment_config(exp_id)
        active_aliases = set()
        has_active_filter = False
        for key in ["online_methods", "offline_methods"]:
            val = exp_cfg.get(key, [])
            if val:
                has_active_filter = True
                if isinstance(val, (list, tuple)):
                    methods = list(val)
                else:
                    methods = [item.strip() for item in str(val).split(",") if item.strip()]
                for m in methods:
                    active_aliases.update(get_method_aliases(m))

        known_problems = ["problem", "ex132(w)", "ex132a(w)", "ex152a(w)", "ex212(w)", "ex242(w)", "ex252(w)", "ex252a(w)", "exc137(w)", "exp426d(w)", "exp426e(w)"]
        discovered = {}
        storage_url = exp_cfg.get("hydra", {}).get("sweeper", {}).get("storage", None)

        for entry in sorted(ckpt_root.rglob("best_model*.ckpt")):
            rel_parts = entry.relative_to(ckpt_root).parts
            parent_dir_name = rel_parts[0]
            detected_dataset = None
            detected_method = parent_dir_name

            if parent_dir_name in known_problems and len(rel_parts) > 2:
                detected_dataset = parent_dir_name
                detected_method = rel_parts[1]
            else:
                for prob in known_problems:
                    prob_clean = prob.replace("(", "_").replace(")", "_").rstrip("_")
                    if parent_dir_name.endswith(f"_{prob}"):
                        detected_dataset = prob
                        detected_method = parent_dir_name[:-len(f"_{prob}")]
                        break
                    elif parent_dir_name.endswith(f"_{prob_clean}"):
                        detected_dataset = prob
                        detected_method = parent_dir_name[:-len(f"_{prob_clean}")].rstrip("_")
                        break

            if has_active_filter:
                aliases_detected = get_method_aliases(detected_method)
                is_active = bool(aliases_detected.intersection(active_aliases)) or any(parent_dir_name.startswith(a) for a in active_aliases)
                if not is_active:
                    continue

            best_ckpt = entry
            if storage_url:
                from src.pipeline.optuna_utils import get_best_trial_id
                study_name = f"{clean_exp}_{parent_dir_name}"
                best_id = get_best_trial_id(storage_url, study_name)
                candidate = (ckpt_root / parent_dir_name / best_id / "best_model.ckpt")
                if candidate.exists():
                    best_ckpt = candidate

            canon_method = get_canonical_method_name(detected_method)
            key = f"{canon_method}_{detected_dataset}" if detected_dataset else canon_method
            if key not in discovered or entry.name == "best_model.ckpt":
                discovered[key] = {
                    "path": best_ckpt,
                    "method": canon_method,
                    "dataset": detected_dataset,
                    "dir_name": parent_dir_name,
                }
        return discovered

    def _run_pyrenees_eval(self, exp_id: str, cfg: dict, group: str, clean_exp: str, output_dir: Path):
        print(f"\n==========================================================================================")
        print(f"=== Action Distribution Evaluation by Student Competency Tier (Pyrenees ITS) ===")
        print(f"==========================================================================================")

        discovered_ckpts = self._discover_checkpoints(exp_id, group, clean_exp)
        if not discovered_ckpts:
            print(f"Notice [action_distribution]: No policy checkpoints found for '{clean_exp}'")
            return
        
        print(f"Discovered {len(discovered_ckpts)} model checkpoints to evaluate.")

        evaluator = PyreneesEvaluator()
        problem_rows, step_rows, tidy_tier_rows = evaluator.evaluate(discovered_ckpts)

        reporter = PyreneesReporter(output_dir, group, clean_exp)
        reporter.report(problem_rows, step_rows, tidy_tier_rows)

        self._plot_problem_level(problem_rows, output_dir, clean_exp)
        self._plot_step_level(step_rows, output_dir, clean_exp)
        self._plot_combined(problem_rows, step_rows, output_dir, clean_exp)
        self._plot_tutor_agreement(problem_rows, step_rows, output_dir, clean_exp)
        
        print("==========================================================================================\n")

    def _plot_problem_level(self, problem_rows, output_dir, clean_exp):
        if not problem_rows:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
        actions = ["PS", "WE", "FWE"]
        action_colors = ["#2b5c8f", "#d95f02", "#7570b3"]
        methods = [r["Method"] for r in problem_rows]
        num_methods = len(problem_rows)
        x_idx = np.arange(num_methods)
        bar_w = 0.25

        for ax_idx, (tier_prefix, _) in enumerate(self.tier_names):
            ax_curr = axes[ax_idx]
            for a_idx, act in enumerate(actions):
                col_name = f"{tier_prefix} {act} %"
                vals = [r.get(col_name, 0.0) for r in problem_rows]
                offset = (a_idx - 1) * bar_w
                ax_curr.bar(x_idx + offset, vals, width=bar_w * 0.9,
                            label=act if ax_idx == 0 else "",
                            color=action_colors[a_idx], alpha=0.85, edgecolor="#333333", linewidth=0.8)

            ax_curr.set_title(f"{tier_prefix} (Problem Level)", fontsize=11, fontweight="bold")
            ax_curr.set_xticks(x_idx)
            ax_curr.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
            ax_curr.set_ylim(0, 105)
            ax_curr.grid(True, axis="y", linestyle="--", alpha=0.4)
            if ax_idx == 0:
                ax_curr.set_ylabel("Action Proportion (%)", fontsize=11, fontweight="bold")

        fig.legend(["Problem Solving (PS)", "Worked Example (WE)", "Faded Worked Example (FWE)"],
                   loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=10, framealpha=0.9)
        fig.suptitle(f"Problem-Level Action Distribution Across Competency Tiers ({clean_exp})", fontsize=13, fontweight="bold", y=1.07)
        fig.tight_layout()
        p_plot_path = output_dir / "action_distribution_by_tier_problem.png"
        plt.savefig(p_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {p_plot_path}")

    def _plot_step_level(self, step_rows, output_dir, clean_exp):
        if not step_rows:
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
        step_actions = ["PS/Elicit", "WE/Tell"]
        step_colors = ["#2b5c8f", "#d95f02"]
        s_methods = [r["Method"] for r in step_rows]
        num_s_methods = len(step_rows)
        x_s_idx = np.arange(num_s_methods)
        bar_w_s = 0.35

        for ax_idx, (tier_prefix, _) in enumerate(self.tier_names):
            ax_curr = axes[ax_idx]
            for a_idx, act in enumerate(step_actions):
                col_name = f"{tier_prefix} {act} %"
                vals = [r.get(col_name, 0.0) for r in step_rows]
                offset = (a_idx - 0.5) * bar_w_s
                ax_curr.bar(x_s_idx + offset, vals, width=bar_w_s * 0.9,
                            label=act if ax_idx == 0 else "",
                            color=step_colors[a_idx], alpha=0.85, edgecolor="#333333", linewidth=0.8)

            ax_curr.set_title(f"{tier_prefix} (Step Level)", fontsize=11, fontweight="bold")
            ax_curr.set_xticks(x_s_idx)
            ax_curr.set_xticklabels(s_methods, rotation=20, ha="right", fontsize=9)
            ax_curr.set_ylim(0, 105)
            ax_curr.grid(True, axis="y", linestyle="--", alpha=0.4)
            if ax_idx == 0:
                ax_curr.set_ylabel("Action Proportion (%)", fontsize=11, fontweight="bold")

        fig.legend(["Problem Solving / Elicit", "Worked Example / Tell"],
                   loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=10, framealpha=0.9)
        fig.suptitle(f"Step-Level Action Distribution Across Competency Tiers ({clean_exp})", fontsize=13, fontweight="bold", y=1.07)
        fig.tight_layout()
        s_plot_path = output_dir / "action_distribution_by_tier_step.png"
        plt.savefig(s_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {s_plot_path}")

    def _plot_combined(self, problem_rows, step_rows, output_dir, clean_exp):
        if not problem_rows or not step_rows:
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharey=True)
        # Row 0: Problem Level
        for ax_idx, (tier_prefix, _) in enumerate(self.tier_names):
            ax_curr = axes[0, ax_idx]
            for a_idx, act in enumerate(["PS", "WE", "FWE"]):
                col_name = f"{tier_prefix} {act} %"
                vals = [r.get(col_name, 0.0) for r in problem_rows]
                offset = (a_idx - 1) * 0.25
                ax_curr.bar(np.arange(len(problem_rows)) + offset, vals, width=0.22,
                            color=["#2b5c8f", "#d95f02", "#7570b3"][a_idx], alpha=0.85, edgecolor="#333333", linewidth=0.8)
            ax_curr.set_title(f"Problem Level: {tier_prefix}", fontsize=11, fontweight="bold")
            ax_curr.set_xticks(np.arange(len(problem_rows)))
            ax_curr.set_xticklabels([r["Method"] for r in problem_rows], rotation=15, ha="right", fontsize=8.5)
            ax_curr.set_ylim(0, 105)
            ax_curr.grid(True, axis="y", linestyle="--", alpha=0.4)
            if ax_idx == 0:
                ax_curr.set_ylabel("Problem Action %", fontsize=11, fontweight="bold")

        # Row 1: Step Level
        for ax_idx, (tier_prefix, _) in enumerate(self.tier_names):
            ax_curr = axes[1, ax_idx]
            for a_idx, act in enumerate(["PS/Elicit", "WE/Tell"]):
                col_name = f"{tier_prefix} {act} %"
                vals = [r.get(col_name, 0.0) for r in step_rows]
                offset = (a_idx - 0.5) * 0.35
                ax_curr.bar(np.arange(len(step_rows)) + offset, vals, width=0.31,
                            color=["#2b5c8f", "#d95f02"][a_idx], alpha=0.85, edgecolor="#333333", linewidth=0.8)
            ax_curr.set_title(f"Step Level: {tier_prefix}", fontsize=11, fontweight="bold")
            ax_curr.set_xticks(np.arange(len(step_rows)))
            ax_curr.set_xticklabels([r["Method"] for r in step_rows], rotation=15, ha="right", fontsize=8.5)
            ax_curr.set_ylim(0, 105)
            ax_curr.grid(True, axis="y", linestyle="--", alpha=0.4)
            if ax_idx == 0:
                ax_curr.set_ylabel("Step Action %", fontsize=11, fontweight="bold")

        fig.suptitle(f"Pyrenees ITS: Problem & Step Level Action Distributions by Student Competency Tier ({clean_exp})",
                     fontsize=13, fontweight="bold", y=0.995)
        fig.tight_layout()
        comb_plot_path = output_dir / "action_distribution_by_tier_combined.png"
        plt.savefig(comb_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {comb_plot_path}")

    def _plot_tutor_agreement(self, problem_rows, step_rows, output_dir, clean_exp):
        all_eval_rows = problem_rows + [r for r in step_rows if r["Method"] != "Historical Tutor (Baseline)"]
        if not all_eval_rows:
            return
            
        fig, ax = plt.subplots(figsize=(max(8, len(all_eval_rows) * 1.8), 5.5))
        m_labels = [f"{r['Method']} ({r['Level']})" for r in all_eval_rows]
        m_agreements = [r["Tutor Agreement %"] for r in all_eval_rows]

        bar_colors = []
        for r in all_eval_rows:
            m_n = r["Method"]
            if "Historical" in m_n or "Baseline" in m_n:
                bar_colors.append("#7f7f7f")
            else:
                style = get_method_style(m_n.split(" [")[0])
                bar_colors.append(style.get("color") or "tab:blue")

        bars = ax.bar(m_labels, m_agreements, color=bar_colors, width=0.55, edgecolor="#333333", linewidth=1.0, alpha=0.85)
        ax.set_ylabel("Tutor Agreement (%)", fontsize=12, fontweight="bold")
        ax.set_title(f"Pyrenees ITS Tutor Agreement ({clean_exp})", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        plt.xticks(rotation=20, ha="right", fontsize=9.5, fontweight="bold")

        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9.5, fontweight="bold")

        fig.tight_layout()
        agr_plot_path = output_dir / "tutor_agreement.png"
        plt.savefig(agr_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {agr_plot_path}")

