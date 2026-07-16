import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root and src to PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from src.methods.cql_agent import CQLAgent

# PyTorch LSTM Model for Early Prediction
class SepsisLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        # x shape: (batch_size, max_seq_len, input_dim)
        # lengths shape: (batch_size,)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hn, _) = self.lstm(packed)
        # hn shape: (1, batch_size, hidden_dim)
        logits = self.fc(hn.squeeze(0))
        return logits

def train_lstm_model(X_train, y_train, input_dim, epochs=15, batch_size=64, device="cpu"):
    # Convert lists/arrays to padded tensors
    lengths_train = torch.tensor([len(seq) for seq in X_train], dtype=torch.long)
    max_len = max(lengths_train).item()
    
    X_train_padded = np.zeros((len(X_train), max_len, input_dim), dtype=np.float32)
    for idx, seq in enumerate(X_train):
        X_train_padded[idx, :len(seq), :] = seq
        
    X_train_tensor = torch.tensor(X_train_padded, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    
    model = SepsisLSTM(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    # Train loop
    model.train()
    dataset_size = len(X_train)
    for epoch in range(epochs):
        permutation = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_train_tensor[indices]
            batch_y = y_train_tensor[indices]
            batch_lengths = lengths_train[indices]
            
            optimizer.zero_grad()
            logits = model(batch_x, batch_lengths)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
    return model

def evaluate_lstm_model(model, X_test, input_dim, device="cpu"):
    model.eval()
    lengths_test = torch.tensor([len(seq) for seq in X_test], dtype=torch.long)
    max_len = max(lengths_test).item()
    
    X_test_padded = np.zeros((len(X_test), max_len, input_dim), dtype=np.float32)
    for idx, seq in enumerate(X_test):
        X_test_padded[idx, :len(seq), :] = seq
        
    X_test_tensor = torch.tensor(X_test_padded, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(X_test_tensor, lengths_test)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze(1)
        preds = (probs > 0.5).astype(np.int32)
        
    return probs, preds

def extract_flat_features(X_seqs):
    # Extract mean and last step features for flat classifiers like RF or LR
    flat_feats = []
    for seq in X_seqs:
        mean_feat = np.mean(seq, axis=0)
        last_feat = seq[-1]
        flat_feats.append(np.concatenate([mean_feat, last_feat]))
    return np.array(flat_feats, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description="Septic Shock Early Prediction Evaluation (varying tau)")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/sepsis/mimic_cql/cql/0/best_model.ckpt", help="Path to CQL agent checkpoint")
    parser.add_argument("--dataset-path", type=str, default="/Users/cameronegbert/Documents/NCSU/Research/datasets/MIMIC 2/mimic_lazy_12_clean_with_interventions_corrected.npz", help="Path to MIMIC dataset")
    parser.add_argument("--models", type=str, default="rf,lstm", help="Comma-separated models to evaluate (rf, lr, lstm)")
    parser.add_argument("--tau-min", type=int, default=1, help="Minimum tau in hours")
    parser.add_argument("--tau-max", type=int, default=36, help="Maximum tau in hours")
    parser.add_argument("--tau-step", type=int, default=4, help="Step size for tau sweep in hours")
    parser.add_argument("--lstm-epochs", type=int, default=15, help="Number of training epochs for LSTM")
    parser.add_argument("--output-dir", type=str, default="results/plots/early_prediction", help="Directory to save plots")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}")
    data = np.load(args.dataset_path, allow_pickle=True)
    X = data['X']
    y = data['y'].squeeze()
    mask = data['mask']
    
    # Load CQL agent to extract Q-values & action probabilities
    print(f"Loading CQL agent from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
    cql_agent = CQLAgent.load_from_checkpoint(args.checkpoint, map_location=device, weights_only=False)
    cql_agent.eval()

    # Pre-compute learned features from CQL agent for all patients and all timesteps to speed up processing
    print("Pre-computing CQL learned features...")
    q_vals_all = []
    action_probs_all = []
    
    # Process in batches to save memory / time
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_x = X[i:i+batch_size, :, :46] # Extract states (first 46 features)
            batch_x_tensor = torch.tensor(batch_x, dtype=torch.float32).to(device)
            
            # Compute Q-values shape: (B, 240, 2)
            B_curr = batch_x_tensor.size(0)
            flat_x = batch_x_tensor.view(-1, 46)
            flat_q = cql_agent.q_network(flat_x)
            q_vals = flat_q.view(B_curr, 240, 2).cpu().numpy()
            
            # Compute action probabilities shape: (B, 240, 2)
            flat_probs = cql_agent.actor.get_action_probs(flat_x)
            action_probs = flat_probs.view(B_curr, 240, 2).cpu().numpy()
            
            q_vals_all.append(q_vals)
            action_probs_all.append(action_probs)
            
    q_vals_all = np.concatenate(q_vals_all, axis=0)
    action_probs_all = np.concatenate(action_probs_all, axis=0)
    
    # Define feature configurations
    # 1. Raw features (states + actions) - size 49
    # 2. Raw + Q-values - size 51
    # 3. Raw + Action Probs - size 51
    # 4. Raw + Q-values + Action Probs - size 53
    feat_configs = {
        "Raw Features (Baseline)": lambda idx, t: X[idx, :t, :49],
        "Raw + Q-values": lambda idx, t: np.concatenate([X[idx, :t, :49], q_vals_all[idx, :t]], axis=-1),
        "Raw + Action Probs": lambda idx, t: np.concatenate([X[idx, :t, :49], action_probs_all[idx, :t]], axis=-1),
        "Raw + Q & Probs": lambda idx, t: np.concatenate([X[idx, :t, :49], q_vals_all[idx, :t], action_probs_all[idx, :t]], axis=-1),
    }

    # Set up tau sweep list
    tau_list = list(range(args.tau_min, args.tau_max + 1, args.tau_step))
    print(f"Sweeping tau (hours early): {tau_list}")

    models_to_eval = [m.strip().lower() for m in args.models.split(",")]
    
    # Store results: results[model_name][feat_config_name] = { 'tau': [], 'f1': [], 'auc': [] }
    results = {}
    for m_name in models_to_eval:
        results[m_name] = {}
        for cfg_name in feat_configs.keys():
            results[m_name][cfg_name] = {"tau": [], "f1": [], "auc": []}

    # Run the sweep
    for tau in tau_list:
        print(f"\n--- Evaluating Lead Time: {tau} Hours Early ---")
        
        # Step size is 30 mins, so tau hours is 2 * tau steps
        steps_early = 2 * tau
        
        # Build dataset for this specific tau
        valid_patients = []
        for i in range(len(X)):
            valid_steps = np.where(mask[i].squeeze() != -1)[0]
            if len(valid_steps) > steps_early:
                # Keep patient stay but cut off the last steps_early steps
                t_cutoff = len(valid_steps) - steps_early
                valid_patients.append((i, t_cutoff))
                
        if len(valid_patients) < 100:
            print(f"Skipping tau={tau} due to insufficient valid patients ({len(valid_patients)}).")
            continue
            
        print(f"Valid patients at tau={tau}: {len(valid_patients)}")
        
        patient_indices = np.array([vp[0] for vp in valid_patients])
        t_cutoffs = np.array([vp[1] for vp in valid_patients])
        y_curr = y[patient_indices]
        
        # Split train/test indices
        train_idxs, test_idxs = train_test_split(
            np.arange(len(patient_indices)), test_size=0.2, random_state=42, stratify=y_curr
        )
        
        # Evaluate each feature configuration
        for cfg_name, feat_func in feat_configs.items():
            print(f"  Feature Config: {cfg_name}")
            
            # Construct sequence dataset
            seq_data = [feat_func(patient_indices[i], t_cutoffs[i]) for i in range(len(patient_indices))]
            input_dim = seq_data[0].shape[-1]
            
            # Split datasets
            X_train = [seq_data[i] for i in train_idxs]
            X_test = [seq_data[i] for i in test_idxs]
            y_train = y_curr[train_idxs]
            y_test = y_curr[test_idxs]
            
            # Model: Random Forest
            if "rf" in models_to_eval:
                X_train_flat = extract_flat_features(X_train)
                X_test_flat = extract_flat_features(X_test)
                
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_train_flat, y_train)
                probs = rf.predict_proba(X_test_flat)[:, 1]
                preds = (probs > 0.5).astype(np.int32)
                
                f1 = f1_score(y_test, preds, zero_division=0)
                auc = roc_auc_score(y_test, probs)
                
                results["rf"][cfg_name]["tau"].append(tau)
                results["rf"][cfg_name]["f1"].append(f1)
                results["rf"][cfg_name]["auc"].append(auc)
                print(f"    RF  -> F1: {f1:.4f}, AUC: {auc:.4f}")
                
            # Model: Logistic Regression
            if "lr" in models_to_eval:
                X_train_flat = extract_flat_features(X_train)
                X_test_flat = extract_flat_features(X_test)
                
                lr = LogisticRegression(max_iter=1000, random_state=42)
                lr.fit(X_train_flat, y_train)
                probs = lr.predict_proba(X_test_flat)[:, 1]
                preds = (probs > 0.5).astype(np.int32)
                
                f1 = f1_score(y_test, preds, zero_division=0)
                auc = roc_auc_score(y_test, probs)
                
                results["lr"][cfg_name]["tau"].append(tau)
                results["lr"][cfg_name]["f1"].append(f1)
                results["lr"][cfg_name]["auc"].append(auc)
                print(f"    LR  -> F1: {f1:.4f}, AUC: {auc:.4f}")

            # Model: LSTM
            if "lstm" in models_to_eval:
                model = train_lstm_model(
                    X_train, y_train, input_dim, epochs=args.lstm_epochs, device=device
                )
                probs, preds = evaluate_lstm_model(model, X_test, input_dim, device=device)
                
                f1 = f1_score(y_test, preds, zero_division=0)
                auc = roc_auc_score(y_test, probs)
                
                results["lstm"][cfg_name]["tau"].append(tau)
                results["lstm"][cfg_name]["f1"].append(f1)
                results["lstm"][cfg_name]["auc"].append(auc)
                print(f"    LSTM -> F1: {f1:.4f}, AUC: {auc:.4f}")

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save text results
    results_file = out_dir / "early_prediction_results.txt"
    with open(results_file, "w") as f:
        f.write("=== Septic Shock Early Prediction Sweep Results ===\n\n")
        for m_name in models_to_eval:
            f.write(f"Model: {m_name.upper()}\n")
            for cfg_name in feat_configs.keys():
                f.write(f"  Features: {cfg_name}\n")
                f.write(f"    Taus: {results[m_name][cfg_name]['tau']}\n")
                f.write(f"    F1s:  {results[m_name][cfg_name]['f1']}\n")
                f.write(f"    AUCs: {results[m_name][cfg_name]['auc']}\n\n")

    # Plot results
    print(f"Plotting results and saving to {out_dir}...")
    for m_name in models_to_eval:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # F1 Plot
        for cfg_name in feat_configs.keys():
            res = results[m_name][cfg_name]
            axes[0].plot(res["tau"], res["f1"], marker='o', label=cfg_name)
        axes[0].set_title(f"F1-Score for Septic Shock Early Prediction ({m_name.upper()})")
        axes[0].set_xlabel("Lead Time (hours early - \u03c4)")
        axes[0].set_ylabel("F1-Score")
        axes[0].grid(True)
        axes[0].legend()
        
        # AUC Plot
        for cfg_name in feat_configs.keys():
            res = results[m_name][cfg_name]
            axes[1].plot(res["tau"], res["auc"], marker='s', label=cfg_name)
        axes[1].set_title(f"AUC-ROC for Septic Shock Early Prediction ({m_name.upper()})")
        axes[1].set_xlabel("Lead Time (hours early - \u03c4)")
        axes[1].set_ylabel("AUC-ROC")
        axes[1].grid(True)
        axes[1].legend()
        
        plt.tight_layout()
        plot_path = out_dir / f"early_prediction_{m_name}_comparison.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot: {plot_path}")
        
    print("Early prediction evaluation finished successfully!")

if __name__ == "__main__":
    main()
