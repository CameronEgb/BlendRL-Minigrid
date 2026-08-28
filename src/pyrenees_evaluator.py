import torch
import numpy as np
import pandas as pd
from pathlib import Path

from plot.base import clean_label

class PyreneesEvaluator:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.batch_size = 5000
        self.tier_names = [("Low Tier", 0), ("Med Tier", 1), ("High Tier", 2)]
        
    def _load_agent(self, path):
        from src.methods.cql_agent import CQLAgent
        from src.methods.cew_agent import CEWAgent
        from src.methods.iql_agent import IQLAgent
        last_error = None
        for cls in [CQLAgent, CEWAgent, IQLAgent]:
            try:
                ag = cls.load_from_checkpoint(str(path), map_location=self.device, weights_only=False)
                ag.to(self.device)
                ag.eval()
                return ag
            except Exception as e:
                last_error = e
                try:
                    ag = cls.load_from_checkpoint(str(path), map_location=self.device, weights_only=False, strict=False)
                    ag.to(self.device)
                    ag.eval()
                    return ag
                except Exception as e2:
                    last_error = e2
                    continue
        if last_error is not None:
            print(f"  [PyreneesEvaluator] Checkpoint load error for {path}: {last_error}")
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
            print(f"  Warning [PyreneesEvaluator]: Could not compute GMM competency segmentation: {e}")
            return np.ones(len(obs_matrix), dtype=int)

    def evaluate(self, discovered_ckpts: dict):
        p_obs, p_acts, p_rews, p_tiers = self._load_problem_data()
        s_obs, s_acts, s_rews, s_tiers, step_exercises = self._load_step_data()
        
        problem_rows = []
        step_rows = []
        tidy_tier_rows = []
        
        if p_obs is not None and len(p_obs) > 0:
            self._evaluate_problem_level(
                p_obs, p_acts, p_rews, p_tiers, 
                discovered_ckpts, problem_rows, tidy_tier_rows
            )
            
        if s_obs is not None and len(s_obs) > 0:
            self._evaluate_step_level(
                s_obs, s_acts, s_rews, s_tiers, step_exercises, 
                discovered_ckpts, step_rows, tidy_tier_rows
            )
            
        return problem_rows, step_rows, tidy_tier_rows

    def _load_problem_data(self):
        prob_clean_path = Path("in/datasets/pyrenees/per_problem/problem/clean.npz")
        prob_gmm_path = Path("in/datasets/pyrenees/per_problem/problem/gmm_scaler.npz")
        if not prob_gmm_path.exists():
            prob_gmm_path = Path("in/datasets/pyrenees/pyrenees_gmm_scaler.npz")

        if prob_clean_path.exists():
            try:
                p_data = np.load(prob_clean_path, allow_pickle=True)
                p_obs = np.vstack(p_data["states"]).astype(np.float32)
                p_acts = np.hstack(p_data["actions"]).astype(int)
                p_rews = np.hstack(p_data["rewards"]).astype(float)
                p_tiers = self._compute_gmm_tiers(p_obs, prob_gmm_path)
                return p_obs, p_acts, p_rews, p_tiers
            except Exception as e:
                print(f"  Warning [PyreneesEvaluator]: Error loading problem dataset: {e}")
        return None, None, None, None

    def _load_step_data(self):
        step_exercises = {}
        per_problem_dir = Path("in/datasets/pyrenees/per_problem")
        prob_gmm_path = Path("in/datasets/pyrenees/pyrenees_gmm_scaler.npz")
        
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
                                "obs": s_o, "acts": s_a, "rews": s_r, "tiers": s_t,
                            }
                        except Exception as e:
                            print(f"  Warning [PyreneesEvaluator]: Error loading {pdir.name}: {e}")

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
                s_tiers = self._compute_gmm_tiers(s_obs, prob_gmm_path)
            else:
                return None, None, None, None, {}
                
        return s_obs, s_acts, s_rews, s_tiers, step_exercises

    def _evaluate_problem_level(self, p_obs, p_acts, p_rews, p_tiers, discovered_ckpts, problem_rows, tidy_tier_rows):
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

        for t_label, t_val in self.tier_names:
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

        for key, meta in sorted(discovered_ckpts.items()):
            ds_hint = meta["dataset"]
            if ds_hint is not None and ds_hint != "problem":
                continue

            agent = self._load_agent(meta["path"])
            if agent is None:
                continue

            try:
                test_b = torch.tensor(p_obs[:2], dtype=torch.float32).to(self.device)
                self._get_probs_and_actions(agent, test_b)
            except Exception:
                continue

            all_pol_acts = []
            with torch.no_grad():
                for b_start in range(0, total_p_steps, self.batch_size):
                    b_end = min(b_start + self.batch_size, total_p_steps)
                    obs_b = torch.tensor(p_obs[b_start:b_end], dtype=torch.float32).to(self.device)
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

            for t_label, t_val in self.tier_names:
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

    def _evaluate_step_level(self, s_obs, s_acts, s_rews, s_tiers, step_exercises, discovered_ckpts, step_rows, tidy_tier_rows):
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

        for t_label, t_val in self.tier_names:
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

        for key, meta in sorted(discovered_ckpts.items()):
            ds_hint = meta["dataset"]
            if ds_hint == "problem":
                continue

            agent = self._load_agent(meta["path"])
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
                test_b = torch.tensor(curr_s_obs[:2], dtype=torch.float32).to(self.device)
                self._get_probs_and_actions(agent, test_b)
            except Exception:
                continue

            cur_n_steps = len(curr_s_obs)
            all_pol_acts = []
            with torch.no_grad():
                for b_start in range(0, cur_n_steps, self.batch_size):
                    b_end = min(b_start + self.batch_size, cur_n_steps)
                    obs_b = torch.tensor(curr_s_obs[b_start:b_end], dtype=torch.float32).to(self.device)
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

            for t_label, t_val in self.tier_names:
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
