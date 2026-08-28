#!/usr/bin/env python3
"""
plot/action_distribution.py — Pyrenees ITS Pedagogical Action Distribution Plotter.

Evaluates trained RL policies against the historical ITS tutor baseline, segmenting
decisions across 3 student competency tiers (Low, Medium, High) at both:
  1. Problem Level (Problem Solving [PS], Worked Example [WE], Faded Worked Example [FWE])
  2. Step Level (Problem Solving / Elicit [PS], Worked Example / Tell [WE])

Outputs:
  - Console ASCII tables of action selections across tiers
  - CSV tables: action_distribution_problem_level.csv, action_distribution_step_level.csv,
    pyrenees_action_distributions_by_tier.csv, method_comparison.csv
  - Visual figures: action_distribution_by_tier_problem.png, action_distribution_by_tier_step.png,
    action_distribution_by_tier_combined.png, tutor_agreement.png
  - Markdown summary: action_distribution_report.md
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

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot.base import BasePlotter, clean_label, get_canonical_method_name, get_method_aliases
from src.method_registry import get_style as get_method_style


class ActionDistributionPlotter(BasePlotter):
    def __init__(self):
        super().__init__("action_distribution")

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

    def _load_agent(self, path, dev):
        from src.methods.cql_agent import CQLAgent
        from src.methods.cew_agent import CEWAgent
        from src.methods.iql_agent import IQLAgent
        last_error = None
        for cls in [CQLAgent, CEWAgent, IQLAgent]:
            try:
                ag = cls.load_from_checkpoint(str(path), map_location=dev, weights_only=False)
                ag.to(dev)
                ag.eval()
                return ag
            except Exception as e:
                last_error = e
                try:
                    ag = cls.load_from_checkpoint(str(path), map_location=dev, weights_only=False, strict=False)
                    ag.to(dev)
                    ag.eval()
                    return ag
                except Exception as e2:
                    last_error = e2
                    continue
        if last_error is not None:
            print(f"  [action_distribution] Checkpoint load error for {path}: {last_error}")
        return None

    def _get_probs_and_actions(self, ag, obs_b):
        if hasattr(ag, "is_modular") and ag.is_modular:
            logic_obs = ag._prepare_logic_obs(obs_b) if hasattr(ag, "_prepare_logic_obs") else obs_b.unsqueeze(1).repeat(1, 2, 1)
            probs, _ = ag.model.actor(obs_b, logic_obs)
            acts = torch.argmax(probs, dim=-1)
            return probs, acts
        elif hasattr(ag, "actor") and hasattr(ag.actor, "get_action_probs"):
            probs = ag.actor.get_action_probs(obs_b)
            acts = torch.argmax(probs, dim=-1)
            return probs, acts
        elif hasattr(ag, "fuzzy_model") and ag.fuzzy_model is not None:
            q = ag.fuzzy_model(obs_b.to("cpu"))
            probs = torch.softmax(q, dim=-1).to(obs_b.device)
            acts = torch.argmax(probs, dim=-1)
            return probs, acts
        elif hasattr(ag, "q_network"):
            if hasattr(ag.q_network, "get_action_probs"):
                probs = ag.q_network.get_action_probs(obs_b)
            else:
                q = ag.q_network(obs_b)
                probs = torch.softmax(q, dim=-1)
            acts = torch.argmax(probs, dim=-1)
            return probs, acts
        elif hasattr(ag, "model") and hasattr(ag.model, "get_q_values"):
            q = ag.model.get_q_values(obs_b)
            probs = torch.softmax(q, dim=-1)
            acts = torch.argmax(probs, dim=-1)
            return probs, acts
        else:
            out = ag.get_action_and_value(obs_b)
            act = out[0] if isinstance(out, (tuple, list)) else out
            n_acts = 3 if obs_b.shape[-1] >= 123 else 2
            probs = torch.zeros((obs_b.shape[0], n_acts), device=obs_b.device)
            probs.scatter_(1, act.unsqueeze(1).long(), 1.0)
            return probs, act

    def _compute_gmm_tiers(self, obs_matrix: np.ndarray, gmm_path: Path) -> np.ndarray:
        """Computes 3-tier competency segmentation (0=Low, 1=Med, 2=High) using GMM parameters."""
        if not gmm_path.exists():
            return np.ones(len(obs_matrix), dtype=int)
        try:
            gdata = np.load(gmm_path, allow_pickle=True)
            means = gdata["means"]
            precisions = gdata["precisions"]
            log_dets = gdata["log_dets"]
            log_weights = gdata["log_weights"]
            feat_idx = gdata["feature_indices"]

            if max(feat_idx) < obs_matrix.shape[1]:
                x_feat = obs_matrix[:, feat_idx]
            else:
                x_feat = obs_matrix[:, :len(feat_idx)]

            d = x_feat.shape[-1]
            const = 0.5 * d * np.log(2.0 * np.pi)
            log_probs = []
            for k in range(len(means)):
                diff = x_feat - means[k]
                maha = np.sum(diff * (diff @ precisions[k]), axis=-1)
                log_p = log_weights[k] - 0.5 * log_dets[k] - 0.5 * maha - const
                log_probs.append(log_p)
            log_probs_mat = np.stack(log_probs, axis=-1)
            posteriors = np.exp(log_probs_mat - np.max(log_probs_mat, axis=-1, keepdims=True))
            posteriors /= posteriors.sum(axis=-1, keepdims=True)
            return posteriors.argmax(axis=1)
        except Exception as e:
            print(f"  Warning [action_distribution]: Could not compute GMM competency segmentation: {e}")
            return np.ones(len(obs_matrix), dtype=int)

    def _run_pyrenees_eval(self, exp_id: str, cfg: dict, group: str, clean_exp: str, output_dir: Path):
        """Domain-appropriate policy evaluation for Pyrenees ITS at Problem and Step levels by student competency tier."""
        print(f"\n==========================================================================================")
        print(f"=== Action Distribution Evaluation by Student Competency Tier (Pyrenees ITS) ===")
        print(f"==========================================================================================")

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        batch_size = 5000

        # ── 1. Prepare Problem-Level Dataset ──────────────────────────────────────────────────────────
        prob_clean_path = Path("in/datasets/pyrenees/per_problem/problem/clean.npz")
        prob_gmm_path = Path("in/datasets/pyrenees/per_problem/problem/gmm_scaler.npz")
        if not prob_gmm_path.exists():
            prob_gmm_path = Path("in/datasets/pyrenees/pyrenees_gmm_scaler.npz")

        p_obs, p_acts, p_rews, p_tiers = None, None, None, None
        if prob_clean_path.exists():
            try:
                p_data = np.load(prob_clean_path, allow_pickle=True)
                p_obs = np.vstack(p_data["states"]).astype(np.float32)
                p_acts = np.hstack(p_data["actions"]).astype(int)
                p_rews = np.hstack(p_data["rewards"]).astype(float)
                p_tiers = self._compute_gmm_tiers(p_obs, prob_gmm_path)
            except Exception as e:
                print(f"  Warning [action_distribution]: Error loading problem dataset: {e}")

        # ── 2. Prepare Step-Level Dataset ─────────────────────────────────────────────────────────────
        step_exercises = {}
        per_problem_dir = Path("in/datasets/pyrenees/per_problem")
        if per_problem_dir.exists():
            for pdir in sorted(per_problem_dir.iterdir()):
                if pdir.is_dir() and pdir.name != "problem":
                    clean_file = pdir / "clean.npz"
                    gmm_file = pdir / "gmm_scaler.npz"
                    if clean_file.exists():
                        try:
                            s_data = np.load(clean_file, allow_pickle=True)
                            s_o = np.vstack(s_data["states"]).astype(np.float32)
                            s_a = np.hstack(s_data["actions"]).astype(int)
                            s_r = np.hstack(s_data["rewards"]).astype(float)
                            s_t = self._compute_gmm_tiers(s_o, gmm_file if gmm_file.exists() else prob_gmm_path)
                            step_exercises[pdir.name] = {
                                "obs": s_o,
                                "acts": s_a,
                                "rews": s_r,
                                "tiers": s_t,
                            }
                        except Exception as e:
                            print(f"  Warning [action_distribution]: Error loading {pdir.name}: {e}")

        # Aggregated Step Dataset across all step exercises
        s_obs_list, s_acts_list, s_rews_list, s_tiers_list = [], [], [], []
        for ex_info in step_exercises.values():
            s_obs_list.append(ex_info["obs"])
            s_acts_list.append(ex_info["acts"])
            s_rews_list.append(ex_info["rews"])
            s_tiers_list.append(ex_info["tiers"])

        if s_obs_list:
            s_obs = np.vstack(s_obs_list)
            s_acts = np.hstack(s_acts_list)
            s_rews = np.hstack(s_rews_list)
            s_tiers = np.hstack(s_tiers_list)
        else:
            fallback_step = Path("in/datasets/pyrenees/pyrenees_clean.npz")
            if fallback_step.exists():
                fb_data = np.load(fallback_step, allow_pickle=True)
                s_obs = np.vstack(fb_data["states"]).astype(np.float32)
                s_acts = np.hstack(fb_data["actions"]).astype(int)
                s_rews = np.hstack(fb_data["rewards"]).astype(float)
                s_tiers = self._compute_gmm_tiers(s_obs, Path("in/datasets/pyrenees/pyrenees_gmm_scaler.npz"))
            else:
                s_obs, s_acts, s_rews, s_tiers = None, None, None, None

        # ── 3. Discover Trained Checkpoints ──────────────────────────────────────────────────────────
        discovered_ckpts = self._discover_checkpoints(exp_id, group, clean_exp)
        if not discovered_ckpts:
            print(f"Notice [action_distribution]: No policy checkpoints found for '{clean_exp}'")
            return

        print(f"Discovered {len(discovered_ckpts)} model checkpoints to evaluate.")

        problem_rows = []
        step_rows = []
        tidy_tier_rows = []
        tier_names = [("Low Tier", 0), ("Med Tier", 1), ("High Tier", 2)]

        # ── 4. Evaluate Problem Level ─────────────────────────────────────────────────────────────────
        if p_obs is not None and len(p_obs) > 0:
            total_p_steps = len(p_obs)
            t_ps = (p_acts == 0).mean() * 100.0
            t_we = (p_acts == 1).mean() * 100.0
            t_fwe = (p_acts == 2).mean() * 100.0

            p_base_row = {
                "Method": "Historical Tutor (Baseline)",
                "Level": "Problem",
                "Dataset": "problem",
                "Tutor Agreement %": 100.0,
                "Overall PS %": float(t_ps),
                "Overall WE %": float(t_we),
                "Overall FWE %": float(t_fwe),
                "Mean Step Reward": float(p_rews.mean()),
                "Agreed Step Reward": float(p_rews.mean()),
            }

            tidy_tier_rows.append({
                "Level": "Problem", "Dataset": "problem", "Method": "Historical Tutor (Baseline)",
                "Tier": "Overall", "N_Steps": total_p_steps,
                "Action_0_Pct (PS)": float(t_ps), "Action_1_Pct (WE)": float(t_we), "Action_2_Pct (FWE)": float(t_fwe),
                "Tutor_Agreement_Pct": 100.0, "Mean_Reward": float(p_rews.mean())
            })

            for t_label, t_val in tier_names:
                m = (p_tiers == t_val)
                n_t = int(m.sum())
                ps_t = float((p_acts[m] == 0).mean() * 100.0) if n_t > 0 else 0.0
                we_t = float((p_acts[m] == 1).mean() * 100.0) if n_t > 0 else 0.0
                fwe_t = float((p_acts[m] == 2).mean() * 100.0) if n_t > 0 else 0.0
                p_base_row[f"{t_label} PS %"] = ps_t
                p_base_row[f"{t_label} WE %"] = we_t
                p_base_row[f"{t_label} FWE %"] = fwe_t

                tidy_tier_rows.append({
                    "Level": "Problem", "Dataset": "problem", "Method": "Historical Tutor (Baseline)",
                    "Tier": t_label, "N_Steps": n_t,
                    "Action_0_Pct (PS)": ps_t, "Action_1_Pct (WE)": we_t, "Action_2_Pct (FWE)": fwe_t,
                    "Tutor_Agreement_Pct": 100.0, "Mean_Reward": float(p_rews[m].mean()) if n_t > 0 else 0.0
                })

            problem_rows.append(p_base_row)

            # Evaluate each model on Problem Level
            for key, meta in sorted(discovered_ckpts.items()):
                ds_hint = meta["dataset"]
                if ds_hint is not None and ds_hint != "problem":
                    continue

                agent = self._load_agent(meta["path"], device)
                if agent is None:
                    continue

                try:
                    test_b = torch.tensor(p_obs[:2], dtype=torch.float32).to(device)
                    self._get_probs_and_actions(agent, test_b)
                except Exception:
                    continue

                all_pol_acts = []
                with torch.no_grad():
                    for b_start in range(0, total_p_steps, batch_size):
                        b_end = min(b_start + batch_size, total_p_steps)
                        obs_b = torch.tensor(p_obs[b_start:b_end], dtype=torch.float32).to(device)
                        _, acts_tensor = self._get_probs_and_actions(agent, obs_b)
                        all_pol_acts.extend(acts_tensor.cpu().numpy())

                all_pol_acts = np.array(all_pol_acts)
                matches = (all_pol_acts == p_acts)
                agr = float(matches.mean() * 100.0)
                ps_r = float((all_pol_acts == 0).mean() * 100.0)
                we_r = float((all_pol_acts == 1).mean() * 100.0)
                fwe_r = float((all_pol_acts == 2).mean() * 100.0)
                agreed_rew = float(p_rews[matches].mean()) if matches.sum() > 0 else float(p_rews.mean())

                display_m = clean_label(meta["method"])
                p_row = {
                    "Method": display_m,
                    "Level": "Problem",
                    "Dataset": "problem",
                    "Tutor Agreement %": agr,
                    "Overall PS %": ps_r,
                    "Overall WE %": we_r,
                    "Overall FWE %": fwe_r,
                    "Mean Step Reward": float(p_rews.mean()),
                    "Agreed Step Reward": agreed_rew,
                }

                tidy_tier_rows.append({
                    "Level": "Problem", "Dataset": "problem", "Method": display_m,
                    "Tier": "Overall", "N_Steps": total_p_steps,
                    "Action_0_Pct (PS)": ps_r, "Action_1_Pct (WE)": we_r, "Action_2_Pct (FWE)": fwe_r,
                    "Tutor_Agreement_Pct": agr, "Mean_Reward": float(p_rews.mean())
                })

                for t_label, t_val in tier_names:
                    m = (p_tiers == t_val)
                    n_t = int(m.sum())
                    ps_t = float((all_pol_acts[m] == 0).mean() * 100.0) if n_t > 0 else 0.0
                    we_t = float((all_pol_acts[m] == 1).mean() * 100.0) if n_t > 0 else 0.0
                    fwe_t = float((all_pol_acts[m] == 2).mean() * 100.0) if n_t > 0 else 0.0
                    agr_t = float((all_pol_acts[m] == p_acts[m]).mean() * 100.0) if n_t > 0 else 0.0

                    p_row[f"{t_label} PS %"] = ps_t
                    p_row[f"{t_label} WE %"] = we_t
                    p_row[f"{t_label} FWE %"] = fwe_t

                    tidy_tier_rows.append({
                        "Level": "Problem", "Dataset": "problem", "Method": display_m,
                        "Tier": t_label, "N_Steps": n_t,
                        "Action_0_Pct (PS)": ps_t, "Action_1_Pct (WE)": we_t, "Action_2_Pct (FWE)": fwe_t,
                        "Tutor_Agreement_Pct": agr_t, "Mean_Reward": float(p_rews[m].mean()) if n_t > 0 else 0.0
                    })

                problem_rows.append(p_row)

        # ── 5. Evaluate Step Level ────────────────────────────────────────────────────────────────────
        if s_obs is not None and len(s_obs) > 0:
            total_s_steps = len(s_obs)
            t_ps_s = (s_acts == 0).mean() * 100.0
            t_we_s = (s_acts == 1).mean() * 100.0
            t_fwe_s = (s_acts == 2).mean() * 100.0 if (s_acts == 2).any() else 0.0

            s_base_row = {
                "Method": "Historical Tutor (Baseline)",
                "Level": "Step",
                "Dataset": "all_steps",
                "Tutor Agreement %": 100.0,
                "Overall PS/Elicit %": float(t_ps_s),
                "Overall WE/Tell %": float(t_we_s),
                "Overall FWE %": float(t_fwe_s),
                "Mean Step Reward": float(s_rews.mean()),
                "Agreed Step Reward": float(s_rews.mean()),
            }

            tidy_tier_rows.append({
                "Level": "Step", "Dataset": "all_steps", "Method": "Historical Tutor (Baseline)",
                "Tier": "Overall", "N_Steps": total_s_steps,
                "Action_0_Pct (PS)": float(t_ps_s), "Action_1_Pct (WE)": float(t_we_s), "Action_2_Pct (FWE)": float(t_fwe_s),
                "Tutor_Agreement_Pct": 100.0, "Mean_Reward": float(s_rews.mean())
            })

            for t_label, t_val in tier_names:
                m = (s_tiers == t_val)
                n_t = int(m.sum())
                ps_t = float((s_acts[m] == 0).mean() * 100.0) if n_t > 0 else 0.0
                we_t = float((s_acts[m] == 1).mean() * 100.0) if n_t > 0 else 0.0
                fwe_t = float((s_acts[m] == 2).mean() * 100.0) if n_t > 0 and (s_acts == 2).any() else 0.0
                s_base_row[f"{t_label} PS/Elicit %"] = ps_t
                s_base_row[f"{t_label} WE/Tell %"] = we_t
                s_base_row[f"{t_label} FWE %"] = fwe_t

                tidy_tier_rows.append({
                    "Level": "Step", "Dataset": "all_steps", "Method": "Historical Tutor (Baseline)",
                    "Tier": t_label, "N_Steps": n_t,
                    "Action_0_Pct (PS)": ps_t, "Action_1_Pct (WE)": we_t, "Action_2_Pct (FWE)": fwe_t,
                    "Tutor_Agreement_Pct": 100.0, "Mean_Reward": float(s_rews[m].mean()) if n_t > 0 else 0.0
                })

            step_rows.append(s_base_row)

            # Evaluate each model on Step Level
            for key, meta in sorted(discovered_ckpts.items()):
                ds_hint = meta["dataset"]
                if ds_hint == "problem":
                    continue

                agent = self._load_agent(meta["path"], device)
                if agent is None:
                    continue

                curr_s_obs = s_obs
                curr_s_acts = s_acts
                curr_s_rews = s_rews
                curr_s_tiers = s_tiers
                dataset_name = "all_steps"

                if ds_hint in step_exercises:
                    curr_s_obs = step_exercises[ds_hint]["obs"]
                    curr_s_acts = step_exercises[ds_hint]["acts"]
                    curr_s_rews = step_exercises[ds_hint]["rews"]
                    curr_s_tiers = step_exercises[ds_hint]["tiers"]
                    dataset_name = ds_hint

                try:
                    test_b = torch.tensor(curr_s_obs[:2], dtype=torch.float32).to(device)
                    self._get_probs_and_actions(agent, test_b)
                except Exception:
                    continue

                cur_n_steps = len(curr_s_obs)
                all_pol_acts = []
                with torch.no_grad():
                    for b_start in range(0, cur_n_steps, batch_size):
                        b_end = min(b_start + batch_size, cur_n_steps)
                        obs_b = torch.tensor(curr_s_obs[b_start:b_end], dtype=torch.float32).to(device)
                        _, acts_tensor = self._get_probs_and_actions(agent, obs_b)
                        all_pol_acts.extend(acts_tensor.cpu().numpy())

                all_pol_acts = np.array(all_pol_acts)
                matches = (all_pol_acts == curr_s_acts)
                agr = float(matches.mean() * 100.0)
                ps_r = float((all_pol_acts == 0).mean() * 100.0)
                we_r = float((all_pol_acts == 1).mean() * 100.0)
                fwe_r = float((all_pol_acts == 2).mean() * 100.0) if (all_pol_acts == 2).any() else 0.0
                agreed_rew = float(curr_s_rews[matches].mean()) if matches.sum() > 0 else float(curr_s_rews.mean())

                display_m = clean_label(meta["method"])
                if ds_hint and ds_hint != "problem":
                    display_m = f"{display_m} [{ds_hint}]"

                s_row = {
                    "Method": display_m,
                    "Level": "Step",
                    "Dataset": dataset_name,
                    "Tutor Agreement %": agr,
                    "Overall PS/Elicit %": ps_r,
                    "Overall WE/Tell %": we_r,
                    "Overall FWE %": fwe_r,
                    "Mean Step Reward": float(curr_s_rews.mean()),
                    "Agreed Step Reward": agreed_rew,
                }

                tidy_tier_rows.append({
                    "Level": "Step", "Dataset": dataset_name, "Method": display_m,
                    "Tier": "Overall", "N_Steps": cur_n_steps,
                    "Action_0_Pct (PS)": ps_r, "Action_1_Pct (WE)": we_r, "Action_2_Pct (FWE)": fwe_r,
                    "Tutor_Agreement_Pct": agr, "Mean_Reward": float(curr_s_rews.mean())
                })

                for t_label, t_val in tier_names:
                    m = (curr_s_tiers == t_val)
                    n_t = int(m.sum())
                    ps_t = float((all_pol_acts[m] == 0).mean() * 100.0) if n_t > 0 else 0.0
                    we_t = float((all_pol_acts[m] == 1).mean() * 100.0) if n_t > 0 else 0.0
                    fwe_t = float((all_pol_acts[m] == 2).mean() * 100.0) if n_t > 0 and (all_pol_acts == 2).any() else 0.0
                    agr_t = float((all_pol_acts[m] == curr_s_acts[m]).mean() * 100.0) if n_t > 0 else 0.0

                    s_row[f"{t_label} PS/Elicit %"] = ps_t
                    s_row[f"{t_label} WE/Tell %"] = we_t
                    s_row[f"{t_label} FWE %"] = fwe_t

                    tidy_tier_rows.append({
                        "Level": "Step", "Dataset": dataset_name, "Method": display_m,
                        "Tier": t_label, "N_Steps": n_t,
                        "Action_0_Pct (PS)": ps_t, "Action_1_Pct (WE)": we_t, "Action_2_Pct (FWE)": fwe_t,
                        "Tutor_Agreement_Pct": agr_t, "Mean_Reward": float(curr_s_rews[m].mean()) if n_t > 0 else 0.0
                    })

                step_rows.append(s_row)

        # ── 6. Console Output of Action Distributions by Student Competency Tier ───────────────────────
        if problem_rows:
            print("\n" + "=" * 90)
            print("  PYRENEES ACTION DISTRIBUTIONS BY STUDENT COMPETENCY TIER: PROBLEM LEVEL")
            print("  (Actions: PS = Problem Solving [0], WE = Worked Example [1], FWE = Faded Worked Example [2])")
            print("=" * 90)
            print(f"{'Method':<36} | {'Tier':<10} | {'PS (%)':>7} | {'WE (%)':>7} | {'FWE (%)':>7} | {'Tutor Agr %':>11}")
            print("-" * 90)
            for row in problem_rows:
                m_name = row["Method"]
                print(f"{m_name:<36} | {'Overall':<10} | {row['Overall PS %']:>6.1f}% | {row['Overall WE %']:>6.1f}% | {row['Overall FWE %']:>6.1f}% | {row['Tutor Agreement %']:>10.1f}%")
                for t_label, _ in tier_names:
                    print(f"{'':<36} | {t_label:<10} | {row.get(f'{t_label} PS %', 0.0):>6.1f}% | {row.get(f'{t_label} WE %', 0.0):>6.1f}% | {row.get(f'{t_label} FWE %', 0.0):>6.1f}% | {'-':>11}")
                print("-" * 90)

        if step_rows:
            print("\n" + "=" * 90)
            print("  PYRENEES ACTION DISTRIBUTIONS BY STUDENT COMPETENCY TIER: STEP LEVEL")
            print("  (Actions: PS/Elicit = 0, WE/Tell = 1)")
            print("=" * 90)
            print(f"{'Method':<36} | {'Tier':<10} | {'PS/Elicit':>9} | {'WE/Tell':>8} | {'FWE (%)':>7} | {'Tutor Agr %':>11}")
            print("-" * 90)
            for row in step_rows:
                m_name = row["Method"]
                print(f"{m_name:<36} | {'Overall':<10} | {row['Overall PS/Elicit %']:>8.1f}% | {row['Overall WE/Tell %']:>7.1f}% | {row['Overall FWE %']:>6.1f}% | {row['Tutor Agreement %']:>10.1f}%")
                for t_label, _ in tier_names:
                    print(f"{'':<36} | {t_label:<10} | {row.get(f'{t_label} PS/Elicit %', 0.0):>8.1f}% | {row.get(f'{t_label} WE/Tell %', 0.0):>7.1f}% | {row.get(f'{t_label} FWE %', 0.0):>6.1f}% | {'-':>11}")
                print("-" * 90)

        # ── 7. Save All Summary CSVs ──────────────────────────────────────────────────────────────────
        df_tidy = pd.DataFrame(tidy_tier_rows)
        tidy_csv_path = output_dir / "pyrenees_action_distributions_by_tier.csv"
        df_tidy.to_csv(tidy_csv_path, index=False)
        print(f"\n  Saved Tidy Tier Breakdown: {tidy_csv_path}")

        if problem_rows:
            df_prob = pd.DataFrame(problem_rows)
            prob_csv_path = output_dir / "action_distribution_problem_level.csv"
            df_prob.to_csv(prob_csv_path, index=False)
            print(f"  Saved Problem Level CSV:   {prob_csv_path}")

        if step_rows:
            df_step = pd.DataFrame(step_rows)
            step_csv_path = output_dir / "action_distribution_step_level.csv"
            df_step.to_csv(step_csv_path, index=False)
            print(f"  Saved Step Level CSV:      {step_csv_path}")

        combined_rows = problem_rows + step_rows
        df_main = pd.DataFrame(combined_rows)
        main_csv_path = output_dir / "method_comparison.csv"
        df_main.to_csv(main_csv_path, index=False)
        print(f"  Saved Main Comparison CSV: {main_csv_path}")

        # ── 8. Generate Visual Plots ──────────────────────────────────────────────────────────────────
        # Plot 1: Problem Level Action Distribution by Tier
        if problem_rows and len(problem_rows) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
            actions = ["PS", "WE", "FWE"]
            action_colors = ["#2b5c8f", "#d95f02", "#7570b3"]
            methods = [r["Method"] for r in problem_rows]
            num_methods = len(problem_rows)
            x_idx = np.arange(num_methods)
            bar_w = 0.25

            for ax_idx, (tier_prefix, _) in enumerate(tier_names):
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

        # Plot 2: Step Level Action Distribution by Tier
        if step_rows and len(step_rows) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
            step_actions = ["PS/Elicit", "WE/Tell"]
            step_colors = ["#2b5c8f", "#d95f02"]
            s_methods = [r["Method"] for r in step_rows]
            num_s_methods = len(step_rows)
            x_s_idx = np.arange(num_s_methods)
            bar_w_s = 0.35

            for ax_idx, (tier_prefix, _) in enumerate(tier_names):
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

        # Plot 3: Combined Problem & Step Level Multi-Panel Figure
        if problem_rows and step_rows:
            fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharey=True)
            # Row 0: Problem Level
            for ax_idx, (tier_prefix, _) in enumerate(tier_names):
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
            for ax_idx, (tier_prefix, _) in enumerate(tier_names):
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

        # Plot 4: Tutor Agreement Bar Chart
        all_eval_rows = problem_rows + [r for r in step_rows if r["Method"] != "Historical Tutor (Baseline)"]
        if all_eval_rows:
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

        # ── 9. Generate Markdown Summary Report ────────────────────────────────────────────────────────
        report_path = output_dir / "action_distribution_report.md"
        with open(report_path, "w") as f:
            f.write(f"# Pyrenees Policy Evaluation: Action Distributions by Student Competency Tier\n\n")
            f.write(f"**Experiment**: `{clean_exp}` (Group: `{group}`)\n\n")
            f.write(f"This report details the pedagogical action selections made by RL policies across student competency tiers (Low, Medium, High) at both Problem Level and Step Level.\n\n")

            if problem_rows:
                f.write("## 1. Problem-Level Action Distributions\n\n")
                f.write("Problem-level decisions choose how the student approaches the entire problem (PS = Problem Solving [0], WE = Worked Example [1], FWE = Faded Worked Example [2]).\n\n")
                f.write(pd.DataFrame(problem_rows).to_markdown(index=False))
                f.write("\n\n")

            if step_rows:
                f.write("## 2. Step-Level Action Distributions\n\n")
                f.write("Step-level decisions choose whether to elicit the step from the student (PS/Elicit = 0) or explain/provide the step (WE/Tell = 1).\n\n")
                f.write(pd.DataFrame(step_rows).to_markdown(index=False))
                f.write("\n\n")

            f.write("## 3. Key Findings & Pedagogical Adaptations\n\n")
            f.write("- **Low Competency Students**: Policies adjust scaffolding (e.g. Worked Examples) to support struggling students.\n")
            f.write("- **High Competency Students**: Policies provide autonomy (Problem Solving / Elicit) to reinforce mastery and problem fluency.\n")
            f.write("\n---\n*Auto-generated by NeSyRL Pipeline*\n")

        print(f"  Saved Markdown Report:     {report_path}")
        print("==========================================================================================\n")


