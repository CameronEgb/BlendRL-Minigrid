import os
import sys
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
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

# --- Positional Encoding for Transformer ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=240):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        sin_vals = torch.sin(position * div_term)
        cos_vals = torch.cos(position * div_term)
        
        pe[:, 0::2] = sin_vals[:, :pe[:, 0::2].shape[1]]
        pe[:, 1::2] = cos_vals[:, :pe[:, 1::2].shape[1]]
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (B, T, d_model)
        return x + self.pe[:, :x.size(1)]

# --- Transformer Classifier Model ---
class SepsisTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, padding_mask):
        # x shape: (B, T, input_dim)
        # padding_mask shape: (B, T), True indicates padding (to ignore)
        x = self.embedding(x) # (B, T, d_model)
        x = self.pos_encoder(x) # (B, T, d_model)
        
        # Transformer forward pass
        out = self.transformer_encoder(x, src_key_padding_mask=padding_mask) # (B, T, d_model)
        
        # Mean pool over non-padding steps
        valid_lens = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1) # (B, 1)
        out_masked = out * (~padding_mask).unsqueeze(-1)
        pooled = out_masked.sum(dim=1) / valid_lens # (B, d_model)
        
        logits = self.fc(pooled)
        return logits

# --- PyTorch LSTM Model ---
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
        logits = self.fc(hn.squeeze(0))
        return logits

# --- LSTM Training and Evaluation Helper Functions ---
def train_lstm_model(X_train, y_train, input_dim, epochs=15, batch_size=64, device="cpu"):
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

# --- Transformer Training and Evaluation Helper Functions ---
def train_transformer_model(X_train, y_train, input_dim, epochs=15, batch_size=64, device="cpu"):
    lengths_train = torch.tensor([len(seq) for seq in X_train], dtype=torch.long)
    max_len = max(lengths_train).item()
    
    X_train_padded = np.zeros((len(X_train), max_len, input_dim), dtype=np.float32)
    mask_train = np.ones((len(X_train), max_len), dtype=bool)
    
    for idx, seq in enumerate(X_train):
        X_train_padded[idx, :len(seq), :] = seq
        mask_train[idx, :len(seq)] = False
        
    X_train_tensor = torch.tensor(X_train_padded, dtype=torch.float32).to(device)
    mask_train_tensor = torch.tensor(mask_train, dtype=torch.bool).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    
    model = SepsisTransformer(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    dataset_size = len(X_train)
    for epoch in range(epochs):
        permutation = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_train_tensor[indices]
            batch_mask = mask_train_tensor[indices]
            batch_y = y_train_tensor[indices]
            
            optimizer.zero_grad()
            logits = model(batch_x, batch_mask)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
    return model

def evaluate_transformer_model(model, X_test, input_dim, device="cpu"):
    model.eval()
    lengths_test = torch.tensor([len(seq) for seq in X_test], dtype=torch.long)
    max_len = max(lengths_test).item()
    
    X_test_padded = np.zeros((len(X_test), max_len, input_dim), dtype=np.float32)
    mask_test = np.ones((len(X_test), max_len), dtype=bool)
    for idx, seq in enumerate(X_test):
        X_test_padded[idx, :len(seq), :] = seq
        mask_test[idx, :len(seq)] = False
        
    X_test_tensor = torch.tensor(X_test_padded, dtype=torch.float32).to(device)
    mask_test_tensor = torch.tensor(mask_test, dtype=torch.bool).to(device)
    
    with torch.no_grad():
        logits = model(X_test_tensor, mask_test_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze(1)
        preds = (probs > 0.5).astype(np.int32)
        
    return probs, preds

def main():
    parser = argparse.ArgumentParser(description="Septic Shock Early Prediction with LSTM & Transformer (varying tau)")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/sepsis/mimic_cql/cql/0/best_model.ckpt", help="Path to CQL agent checkpoint")
    parser.add_argument("--dataset-path", type=str, default="/Users/cameronegbert/Documents/NCSU/Research/datasets/MIMIC 2/mimic_lazy_12_clean_with_interventions_corrected.npz", help="Path to MIMIC dataset")
    parser.add_argument("--tau-min", type=int, default=1, help="Minimum tau in hours")
    parser.add_argument("--tau-max", type=int, default=36, help="Maximum tau in hours")
    parser.add_argument("--tau-step", type=int, default=4, help="Step size for tau sweep in hours")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs for PyTorch models")
    parser.add_argument("--output-dir", type=str, default="results/plots/early_prediction", help="Directory to save plots")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}")
    data = np.load(args.dataset_path, allow_pickle=True)
    X = data['X']
    y = data['y'].squeeze()
    mask = data['mask']
    
    # Load CQL agent to extract Q-values
    print(f"Loading CQL agent from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
    cql_agent = CQLAgent.load_from_checkpoint(args.checkpoint, map_location=device, weights_only=False)
    cql_agent.eval()

    # Pre-compute learned V-values from CQL agent for all patients and all timesteps
    print("Pre-computing CQL state-value functions V(s) = max_a Q(s, a)...")
    v_vals_all = []
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_x = X[i:i+batch_size, :, :46] # Extract states (first 46 features)
            batch_x_tensor = torch.tensor(batch_x, dtype=torch.float32).to(device)
            
            B_curr = batch_x_tensor.size(0)
            flat_x = batch_x_tensor.view(-1, 46)
            flat_q = cql_agent.q_network(flat_x)
            q_vals = flat_q.view(B_curr, 240, 2)
            
            # V(s) = max_a Q(s, a)
            v_vals = torch.max(q_vals, dim=-1)[0].unsqueeze(-1).cpu().numpy()
            v_vals_all.append(v_vals)
            
    v_vals_all = np.concatenate(v_vals_all, axis=0) # shape: (N, 240, 1)
    
    # Feature configurations to evaluate
    feat_configs = {
        "no_v": lambda idx, t: X[idx, :t, :49], # Baseline clinical features
        "with_v": lambda idx, t: np.concatenate([X[idx, :t, :49], v_vals_all[idx, :t]], axis=-1), # Baseline + V(s)
    }

    # Set up tau sweep list
    tau_list = list(range(args.tau_min, args.tau_max + 1, args.tau_step))
    print(f"Sweeping tau (hours early): {tau_list}")

    # Models we want to run:
    # 1. LSTM (without V)
    # 2. LSTM (with V)
    # 3. Transformer (without V)
    # 4. Transformer (with V)
    model_configs = [
        ("LSTM (no V)", "lstm", "no_v"),
        ("LSTM (with V)", "lstm", "with_v"),
        ("Transformer (no V)", "transformer", "no_v"),
        ("Transformer (with V)", "transformer", "with_v"),
    ]
    
    # Initialize results
    results = {}
    for m_cfg, _, _ in model_configs:
        results[m_cfg] = {"tau": [], "f1": [], "auc": []}

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
                t_cutoff = len(valid_steps) - steps_early
                valid_patients.append((i, t_cutoff))
                
        if len(valid_patients) < 100:
            print(f"Skipping tau={tau} due to insufficient valid patients.")
            continue
            
        print(f"Valid patients at tau={tau}: {len(valid_patients)}")
        
        patient_indices = np.array([vp[0] for vp in valid_patients])
        t_cutoffs = np.array([vp[1] for vp in valid_patients])
        y_curr = y[patient_indices]
        
        # Split train/test indices
        train_idxs, test_idxs = train_test_split(
            np.arange(len(patient_indices)), test_size=0.2, random_state=42, stratify=y_curr
        )
        
        # Train and evaluate each configuration
        for m_cfg_name, m_type, feat_key in model_configs:
            print(f"  Running model configuration: {m_cfg_name}")
            
            # Construct dataset
            feat_func = feat_configs[feat_key]
            seq_data = [feat_func(patient_indices[i], t_cutoffs[i]) for i in range(len(patient_indices))]
            input_dim = seq_data[0].shape[-1]
            
            X_train = [seq_data[i] for i in train_idxs]
            X_test = [seq_data[i] for i in test_idxs]
            y_train = y_curr[train_idxs]
            y_test = y_curr[test_idxs]
            
            if m_type == "lstm":
                model = train_lstm_model(
                    X_train, y_train, input_dim, epochs=args.epochs, device=device
                )
                probs, preds = evaluate_lstm_model(model, X_test, input_dim, device=device)
            elif m_type == "transformer":
                model = train_transformer_model(
                    X_train, y_train, input_dim, epochs=args.epochs, device=device
                )
                probs, preds = evaluate_transformer_model(model, X_test, input_dim, device=device)
            
            f1 = f1_score(y_test, preds, zero_division=0)
            auc = roc_auc_score(y_test, probs)
            
            results[m_cfg_name]["tau"].append(tau)
            results[m_cfg_name]["f1"].append(f1)
            results[m_cfg_name]["auc"].append(auc)
            print(f"    -> F1: {f1:.4f}, AUC: {auc:.4f}")

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save text results
    results_file = out_dir / "early_prediction_dl_results.txt"
    with open(results_file, "w") as f:
        f.write("=== Septic Shock Early Prediction DL Sweep Results ===\n\n")
        for m_cfg_name, _, _ in model_configs:
            f.write(f"Model Configuration: {m_cfg_name}\n")
            f.write(f"  Taus: {results[m_cfg_name]['tau']}\n")
            f.write(f"  F1s:  {results[m_cfg_name]['f1']}\n")
            f.write(f"  AUCs: {results[m_cfg_name]['auc']}\n\n")

    # Plot results
    print(f"Plotting results and saving to {out_dir}...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    markers = ['o', 's', '^', 'D']
    
    # F1 Plot
    for idx, (m_cfg_name, _, _) in enumerate(model_configs):
        res = results[m_cfg_name]
        axes[0].plot(res["tau"], res["f1"], marker=markers[idx], color=colors[idx], label=m_cfg_name, linewidth=2)
    axes[0].set_title("F1-Score for Septic Shock Early Prediction (\u03c4 \u2208 [1, 36])", fontsize=12)
    axes[0].set_xlabel("Lead Time (hours early - \u03c4)", fontsize=11)
    axes[0].set_ylabel("F1-Score", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)
    
    # AUC Plot
    for idx, (m_cfg_name, _, _) in enumerate(model_configs):
        res = results[m_cfg_name]
        axes[1].plot(res["tau"], res["auc"], marker=markers[idx], color=colors[idx], label=m_cfg_name, linewidth=2)
    axes[1].set_title("AUC-ROC for Septic Shock Early Prediction (\u03c4 \u2208 [1, 36])", fontsize=12)
    axes[1].set_xlabel("Lead Time (hours early - \u03c4)", fontsize=11)
    axes[1].set_ylabel("AUC-ROC", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    plot_path = out_dir / "early_prediction_dl_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved plot: {plot_path}")
    
    print("Early prediction deep learning evaluation finished successfully!")

if __name__ == "__main__":
    main()
