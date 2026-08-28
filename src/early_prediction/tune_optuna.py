import os
import sys
import argparse
import hydra
from omegaconf import DictConfig, OmegaConf
import math
import numpy as np
import torch
import torch.nn as nn
import optuna
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc

# Add project root and src to PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.early_prediction.model import (
    SepsisTransformer, SepsisLSTM, compute_volatility_features,
    normalize_features, train_transformer_model, evaluate_transformer_model,
    train_lstm_model, evaluate_lstm_model, find_default_mimic_npz
)

from src.pipeline.datasets import resolve_mimic_npz_path

def compute_metric(y_true, probs, metric_name="auprc"):
    metric_name = metric_name.lower()
    if metric_name in ("acc", "accuracy"):
        preds = (probs >= 0.5).astype(int)
        return float(np.mean(preds == y_true))
    elif metric_name == "f1":
        preds = (probs >= 0.5).astype(int)
        return float(f1_score(y_true, preds, zero_division=0))
    elif metric_name in ("roc_auc", "auc"):
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, probs))
    else:  # auprc
        precisions, recalls, _ = precision_recall_curve(y_true, probs)
        return float(auc(recalls, precisions))

def objective(trial, seq_data_dict, y_cohort, c_indices, args):
    model_target = args.model_target.lower()
    
    if model_target == "lstm_no_v":
        model_type = "lstm"
        use_v_feat = False
    elif model_target == "lstm_with_v":
        model_type = "lstm"
        use_v_feat = True
    elif model_target == "transformer_no_v":
        model_type = "transformer"
        use_v_feat = False
    elif model_target == "transformer_with_v":
        model_type = "transformer"
        use_v_feat = True
    else:
        model_type = trial.suggest_categorical("model_type", ["transformer", "lstm"])
        use_v_feat = trial.suggest_categorical("use_v_feat", [True, False])

    seq_data = seq_data_dict["with_v"] if (use_v_feat and "with_v" in seq_data_dict) else seq_data_dict["no_v"]
            
    input_dim = seq_data[0].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.4)
    epochs = trial.suggest_int("epochs", 15, 25)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    use_focal_loss = trial.suggest_categorical("use_focal_loss", [True, False])
    use_tcn_conv = trial.suggest_categorical("use_tcn_conv", [True, False])
    
    split_scores = []
    n_eval_splits = getattr(args, "n_eval_splits", 5) # 5 stratified cross-validation splits per trial for robust evaluation
    
    if model_type == "transformer":
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        nhead = trial.suggest_categorical("nhead", [2, 4, 8])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        pos_type = trial.suggest_categorical("pos_type", ["learned", "sinusoidal"])
        use_cls_token = trial.suggest_categorical("use_cls_token", [True, False])
        norm_first = trial.suggest_categorical("norm_first", [True, False])
        
        for m_idx in range(n_eval_splits):
            seed_val = 100 + m_idx
            train_idx, test_idx = train_test_split(
                np.arange(len(c_indices)), test_size=0.2, random_state=seed_val, stratify=y_cohort
            )
            X_tr, X_te = [seq_data[k] for k in train_idx], [seq_data[k] for k in test_idx]
            y_tr, y_te = y_cohort[train_idx], y_cohort[test_idx]
            
            X_tr, X_te = normalize_features(X_tr, X_te)
            model, _ = train_transformer_model(
                X_tr, y_tr, input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers,
                dropout=dropout, weight_decay=weight_decay, norm_first=norm_first,
                pos_type=pos_type, use_cls_token=use_cls_token, use_tcn_conv=use_tcn_conv,
                use_focal_loss=use_focal_loss, epochs=epochs, batch_size=batch_size,
                lr=lr, device=device, seed=seed_val
            )
            probs = evaluate_transformer_model(model, X_te, input_dim, device=device)
            score_val = compute_metric(y_te, probs, metric_name=args.metric)
            split_scores.append(score_val)
            
    else: # lstm
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        bidirectional = trial.suggest_categorical("bidirectional", [True, False])
        
        for m_idx in range(n_eval_splits):
            seed_val = 100 + m_idx
            train_idx, test_idx = train_test_split(
                np.arange(len(c_indices)), test_size=0.2, random_state=seed_val, stratify=y_cohort
            )
            X_tr, X_te = [seq_data[k] for k in train_idx], [seq_data[k] for k in test_idx]
            y_tr, y_te = y_cohort[train_idx], y_cohort[test_idx]
            
            X_tr, X_te = normalize_features(X_tr, X_te)
            model, _ = train_lstm_model(
                X_tr, y_tr, input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                use_focal_loss=use_focal_loss, use_tcn_conv=use_tcn_conv,
                bidirectional=bidirectional, device=device, seed=seed_val
            )
            probs = evaluate_lstm_model(model, X_te, input_dim, device=device)
            score_val = compute_metric(y_te, probs, metric_name=args.metric)
            split_scores.append(score_val)
            
    mean_score = float(np.mean(split_scores))
    return mean_score

class Dict2Obj:
    def __init__(self, d, defaults=None):
        if defaults:
            for k, v in defaults.items():
                setattr(self, k, v)
        for k, v in d.items():
            setattr(self, k, v)

@hydra.main(version_base=None, config_path="../../in/config", config_name="config")
def main(cfg: DictConfig):
    # Auto-infer experiment_id from Hydra task override if not explicitly specified
    if cfg.get("experiment_id", "default_exp") == "default_exp":
        try:
            from hydra.core.hydra_config import HydraConfig
            if HydraConfig.initialized():
                for override in HydraConfig.get().overrides.task:
                    if override.startswith("+experiment=") or override.startswith("experiment="):
                        exp_stem = Path(override.split("=")[-1]).stem
                        cfg.experiment_id = exp_stem
                        break
        except Exception:
            pass

    ep_cfg = cfg.get("early_prediction", {})
    if isinstance(ep_cfg, DictConfig):
        ep_cfg = OmegaConf.to_container(ep_cfg, resolve=True)
        
    defaults = {
        "n_trials": 30,
        "model_target": "all",
        "dataset_path": str(resolve_mimic_npz_path()),
        "checkpoint": "results/checkpoints/mimic/tune_mimic_cql",
        "window_hours": 12,
        "use_volatility": True,
        "metric": "auprc",
        "n_eval_splits": 5,
        "out_dir": "results/plots/early_prediction/tune_early_pred"
    }
    
    # Check for CLI overrides mapped through the task pipeline
    import sys
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--model-target"):
            if "=" in arg:
                defaults["model_target"] = arg.split("=")[1]
            else:
                defaults["model_target"] = sys.argv[i+1]
                
    args = Dict2Obj(ep_cfg, defaults)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading dataset from: {args.dataset_path}")
    data = np.load(args.dataset_path, allow_pickle=True)
    X = data['X']
    y = data['y'].squeeze()
    mask = data['mask']
    mask_2d = mask.squeeze(-1) if mask.ndim == 3 else mask
    patient_lengths = (mask_2d != -1).sum(axis=-1)
    
    # Pre-compute CQL state values V(s) if available
    v_vals_all = None
    cql_ckpt_path = None
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        cand = list(ckpt_path.glob("**/*.ckpt"))
        if cand:
            cql_ckpt_path = str(cand[-1])
            
    if cql_ckpt_path and os.path.exists(cql_ckpt_path):
        try:
            from src.methods.cql_agent import CQLAgent
            torch.serialization.add_safe_globals([
                getattr(sys.modules.get('omegaconf.dictconfig', None), 'DictConfig', None)
            ])
            print(f"Loading CQL agent from: {cql_ckpt_path} for V(s)...")
            cql_agent = CQLAgent.load_from_checkpoint(cql_ckpt_path, map_location=device, weights_only=False)
            cql_agent.eval()
            
            v_vals_all = np.zeros((len(X), 240, 1), dtype=np.float32)
            with torch.no_grad():
                for i in range(0, len(X), 128):
                    batch_x = torch.as_tensor(X[i:i+128, :, :46], dtype=torch.float32, device=device)
                    B_curr = batch_x.size(0)
                    flat_q = cql_agent.q_network(batch_x.view(-1, 46))
                    v_vals = torch.max(flat_q.view(B_curr, 240, -1), dim=-1)[0].unsqueeze(-1).cpu().numpy()
                    v_vals_all[i:i+128] = v_vals
        except Exception as e:
            print(f"Notice: CQL loading skipped: {e}")
            v_vals_all = np.zeros((len(X), 240, 1), dtype=np.float32)
            
    if v_vals_all is None:
        v_vals_all = np.zeros((len(X), 240, 1), dtype=np.float32)

    # Pre-extract cohort sequences once across all trials
    tau = 9
    steps_early = 2 * tau
    w_steps = 2 * args.window_hours
    min_stay_steps = 2 * 36 + w_steps
    c_indices = np.where(patient_lengths >= min_stay_steps)[0]
    t_cutoffs = patient_lengths[c_indices] - steps_early
    y_cohort = y[c_indices]

    print(f"Pre-extracting features for {len(c_indices)} cohort patients (tau={tau}h, window={args.window_hours}h)...")
    seq_data_base = []
    seq_data_with_v = []
    for i, original_idx in enumerate(c_indices):
        tc = t_cutoffs[i]
        st = max(0, tc - w_steps)
        raw_seq = X[original_idx, st:tc, :49]
        feat_seq = compute_volatility_features(raw_seq) if args.use_volatility else raw_seq
        seq_data_base.append(feat_seq)
        v_seq = v_vals_all[original_idx, st:tc]
        seq_data_with_v.append(np.concatenate([feat_seq, v_seq], axis=-1))

    seq_data_dict = {
        "no_v": seq_data_base,
        "with_v": seq_data_with_v
    }
        
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    print(f"Starting Optuna Study for Target '{args.model_target}' over {args.n_trials} trials...")
    study.optimize(lambda trial: objective(trial, seq_data_dict, y_cohort, c_indices, args), n_trials=args.n_trials)
    
    print("\n=== Optuna Study Complete ===")
    print(f"Target: {args.model_target}")
    print(f"Best Trial Score ({args.metric.upper()} at 9h): {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.model_target == "all":
        param_filename = "best_params.yaml"
        hist_filename = "optuna_optimization_history.png"
    else:
        param_filename = f"best_params_{args.model_target}.yaml"
        hist_filename = f"optuna_history_{args.model_target}.png"
        
    best_yaml = out_dir / param_filename
    with open(best_yaml, "w") as f:
        yaml.dump(study.best_params, f)
    print(f"Saved best parameters to {best_yaml}")

    # Plot Optuna optimization history
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        trial_numbers = [t.number for t in study.trials if t.value is not None]
        trial_values = [t.value for t in study.trials if t.value is not None]
        best_values = np.maximum.accumulate(trial_values)
        
        ax.plot(trial_numbers, trial_values, 'o', color='tab:blue', alpha=0.6, label=f'Trial {args.metric.upper()}')
        ax.plot(trial_numbers, best_values, '-', color='tab:red', linewidth=2.5, label=f'Best Cumulative {args.metric.upper()}')
        ax.set_title(f"Optuna Optimization History ({args.model_target} - \u03c4=9h)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Trial Number", fontsize=11)
        ax.set_ylabel(f"Validation {args.metric.upper()}", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=10)
        plt.tight_layout()
        hist_path = out_dir / hist_filename
        plt.savefig(hist_path, dpi=200)
        plt.close()
        print(f"Saved optimization history plot to {hist_path}")
    except Exception as e:
        print(f"Warning: Could not save optimization history plot: {e}")

