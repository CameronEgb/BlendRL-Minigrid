import os
import sys
import argparse
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

def objective(trial, X, y, mask, patient_lengths, v_vals_all, args):
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

    tau = trial.suggest_categorical("tau", [9]) # Benchmark at tau=9h early prediction
    steps_early = 2 * tau
    w_steps = 2 * args.window_hours
    
    min_stay_steps = 2 * 36 + w_steps
    c_indices = np.array([i for i in range(len(X)) if patient_lengths[i] >= min_stay_steps])
    t_cutoffs = patient_lengths[c_indices] - steps_early
    y_cohort = y[c_indices]
    
    seq_data = []
    for i, original_idx in enumerate(c_indices):
        tc = t_cutoffs[i]
        st = max(0, tc - w_steps)
        raw_seq = X[original_idx, st:tc, :49]
        feat_seq = compute_volatility_features(raw_seq) if args.use_volatility else raw_seq
        if use_v_feat and v_vals_all is not None:
            v_seq = v_vals_all[original_idx, st:tc]
            seq_data.append(np.concatenate([feat_seq, v_seq], axis=-1))
        else:
            seq_data.append(feat_seq)
            
    input_dim = seq_data[0].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.4)
    epochs = trial.suggest_int("epochs", 15, 25)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    use_focal_loss = trial.suggest_categorical("use_focal_loss", [True, False])
    use_tcn_conv = trial.suggest_categorical("use_tcn_conv", [True, False])
    
    split_auprcs = []
    n_eval_splits = 5 # 5 stratified splits per trial for speed
    
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
            precisions, recalls, _ = precision_recall_curve(y_te, probs)
            auprc_val = float(auc(recalls, precisions))
            split_auprcs.append(auprc_val)
            
    else: # lstm
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        
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
                device=device, seed=seed_val
            )
            probs = evaluate_lstm_model(model, X_te, input_dim, device=device)
            precisions, recalls, _ = precision_recall_curve(y_te, probs)
            auprc_val = float(auc(recalls, precisions))
            split_auprcs.append(auprc_val)
            
    mean_auprc = float(np.mean(split_auprcs))
    return mean_auprc

def main():
    parser = argparse.ArgumentParser(description="Modular Optuna Hyperparameter Search for Early Prediction Models")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--model-target", type=str, default="all", help="Target architecture to tune: lstm_no_v, lstm_with_v, transformer_no_v, transformer_with_v, or all")
    parser.add_argument("--dataset-path", type=str, default=find_default_mimic_npz())
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/mimic/tune_mimic_cql")
    parser.add_argument("--window-hours", type=int, default=12)
    parser.add_argument("--use-volatility", action="store_true", default=True)
    parser.add_argument("--out-dir", type=str, default="results/plots/early_prediction/tune_early_pred")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading dataset from: {args.dataset_path}")
    data = np.load(args.dataset_path, allow_pickle=True)
    X = data['X']
    y = data['y'].squeeze()
    mask = data['mask']
    patient_lengths = np.array([(mask[i].squeeze() != -1).sum() for i in range(len(X))])
    
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
                    batch_x = torch.tensor(X[i:i+128, :, :46], dtype=torch.float32).to(device)
                    B_curr = batch_x.size(0)
                    flat_q = cql_agent.q_network(batch_x.view(-1, 46))
                    v_vals = torch.max(flat_q.view(B_curr, 240, 2), dim=-1)[0].unsqueeze(-1).cpu().numpy()
                    v_vals_all[i:i+128] = v_vals
        except Exception as e:
            print(f"Notice: CQL loading skipped: {e}")
            v_vals_all = np.zeros((len(X), 240, 1), dtype=np.float32)
            
    if v_vals_all is None:
        v_vals_all = np.zeros((len(X), 240, 1), dtype=np.float32)
        
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    print(f"Starting Optuna Study for Target '{args.model_target}' over {args.n_trials} trials...")
    study.optimize(lambda trial: objective(trial, X, y, mask, patient_lengths, v_vals_all, args), n_trials=args.n_trials)
    
    print("\n=== Optuna Study Complete ===")
    print(f"Target: {args.model_target}")
    print(f"Best Trial Score (AUPRC at 9h): {study.best_value:.4f}")
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
        
        ax.plot(trial_numbers, trial_values, 'o', color='tab:blue', alpha=0.6, label='Trial AUPRC')
        ax.plot(trial_numbers, best_values, '-', color='tab:red', linewidth=2.5, label='Best Cumulative AUPRC')
        ax.set_title(f"Optuna Optimization History ({args.model_target} - \u03c4=9h)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Trial Number", fontsize=11)
        ax.set_ylabel("Validation AUPRC", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=10)
        plt.tight_layout()
        hist_path = out_dir / hist_filename
        plt.savefig(hist_path, dpi=200)
        plt.close()
        print(f"Saved optimization history plot to {hist_path}")
    except Exception as e:
        print(f"Warning: Could not save optimization history plot: {e}")

if __name__ == "__main__":
    main()
