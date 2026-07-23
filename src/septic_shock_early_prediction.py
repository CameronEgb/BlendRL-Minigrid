import os
import sys
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc
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
        return x + self.pe[:, :x.size(1)]

# --- Lightweight Transformer Classifier Model ---
class SepsisTransformer(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=2, num_layers=1, dim_feedforward=64):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, padding_mask):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Last non-padding step representation per sequence
        valid_lens = (~padding_mask).sum(dim=1).clamp(min=1)
        last_indices = valid_lens - 1
        pooled = out[torch.arange(out.size(0)), last_indices]
        
        logits = self.fc(pooled)
        return logits

# --- Lightweight PyTorch LSTM Model ---
class SepsisLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hn, _) = self.lstm(packed)
        logits = self.fc(hn.squeeze(0))
        return logits

# --- Training Helper Functions ---
def train_lstm_model(X_train, y_train, input_dim, epochs=10, batch_size=64, device="cpu", seed=42, plot_convergence=False, plot_path=None):
    torch.manual_seed(seed)
    
    if plot_convergence:
        sub_train_idx, sub_val_idx = train_test_split(
            np.arange(len(X_train)), test_size=0.2, random_state=seed, stratify=y_train
        )
        X_sub_train = [X_train[i] for i in sub_train_idx]
        X_sub_val = [X_train[i] for i in sub_val_idx]
        y_sub_train = y_train[sub_train_idx]
        y_sub_val = y_train[sub_val_idx]
    else:
        X_sub_train = X_train
        y_sub_train = y_train
        X_sub_val = []
        y_sub_val = []

    lengths_train = torch.tensor([len(seq) for seq in X_sub_train], dtype=torch.long)
    max_len = max(lengths_train).item()
    
    X_train_padded = np.zeros((len(X_sub_train), max_len, input_dim), dtype=np.float32)
    for idx, seq in enumerate(X_sub_train):
        X_train_padded[idx, :len(seq), :] = seq
        
    X_train_tensor = torch.tensor(X_train_padded, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_sub_train, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Class-weighted loss
    n_pos = (y_sub_train == 1).sum()
    n_neg = (y_sub_train == 0).sum()
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32).to(device)
    
    model = SepsisLSTM(input_dim=input_dim, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    train_losses = []
    val_losses = []
    
    dataset_size = len(X_sub_train)
    
    # Pre-pad val set if plotting convergence
    if plot_convergence:
        lengths_val = torch.tensor([len(seq) for seq in X_sub_val], dtype=torch.long)
        max_len_val = max(lengths_val).item()
        X_val_padded = np.zeros((len(X_sub_val), max_len_val, input_dim), dtype=np.float32)
        for idx, seq in enumerate(X_sub_val):
            X_val_padded[idx, :len(seq), :] = seq
        X_val_tensor = torch.tensor(X_val_padded, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_sub_val, dtype=torch.float32).unsqueeze(1).to(device)
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(dataset_size)
        epoch_loss = 0.0
        batches = 0
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
            
            epoch_loss += loss.item()
            batches += 1
            
        train_losses.append(epoch_loss / max(1, batches))
        
        if plot_convergence:
            model.eval()
            with torch.no_grad():
                logits_val = model(X_val_tensor, lengths_val)
                loss_val = criterion(logits_val, y_val_tensor)
                val_losses.append(loss_val.item())
                
    if plot_convergence and plot_path:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", color="tab:red")
        plt.plot(range(1, epochs + 1), val_losses, label="Val Loss", color="tab:blue", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("LSTM Predictor Model Convergence (Septic Shock Prediction)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved LSTM convergence plot to: {plot_path}")
        
    # Re-train on FULL training dataset to maintain the exact same behavior as original code!
    if plot_convergence:
        lengths_full = torch.tensor([len(seq) for seq in X_train], dtype=torch.long)
        max_len_full = max(lengths_full).item()
        X_full_padded = np.zeros((len(X_train), max_len_full, input_dim), dtype=np.float32)
        for idx, seq in enumerate(X_train):
            X_full_padded[idx, :len(seq), :] = seq
        X_full_tensor = torch.tensor(X_full_padded, dtype=torch.float32).to(device)
        y_full_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        
        n_pos_full = (y_train == 1).sum()
        n_neg_full = (y_train == 0).sum()
        pos_weight_full = torch.tensor([n_neg_full / max(1, n_pos_full)], dtype=torch.float32).to(device)
        
        model = SepsisLSTM(input_dim=input_dim, hidden_dim=32).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        criterion_full = nn.BCEWithLogitsLoss(pos_weight=pos_weight_full)
        
        model.train()
        full_size = len(X_train)
        for epoch in range(epochs):
            permutation = torch.randperm(full_size)
            for i in range(0, full_size, batch_size):
                indices = permutation[i:i+batch_size]
                batch_x = X_full_tensor[indices]
                batch_y = y_full_tensor[indices]
                batch_lengths = lengths_full[indices]
                
                optimizer.zero_grad()
                logits = model(batch_x, batch_lengths)
                loss = criterion_full(logits, batch_y)
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
    return probs

def train_transformer_model(X_train, y_train, input_dim, epochs=10, batch_size=64, device="cpu", seed=42, plot_convergence=False, plot_path=None):
    torch.manual_seed(seed)
    
    if plot_convergence:
        sub_train_idx, sub_val_idx = train_test_split(
            np.arange(len(X_train)), test_size=0.2, random_state=seed, stratify=y_train
        )
        X_sub_train = [X_train[i] for i in sub_train_idx]
        X_sub_val = [X_train[i] for i in sub_val_idx]
        y_sub_train = y_train[sub_train_idx]
        y_sub_val = y_train[sub_val_idx]
    else:
        X_sub_train = X_train
        y_sub_train = y_train
        X_sub_val = []
        y_sub_val = []

    lengths_train = torch.tensor([len(seq) for seq in X_sub_train], dtype=torch.long)
    max_len = max(lengths_train).item()
    
    X_train_padded = np.zeros((len(X_sub_train), max_len, input_dim), dtype=np.float32)
    mask_train = np.ones((len(X_sub_train), max_len), dtype=bool)
    
    for idx, seq in enumerate(X_sub_train):
        X_train_padded[idx, :len(seq), :] = seq
        mask_train[idx, :len(seq)] = False
        
    X_train_tensor = torch.tensor(X_train_padded, dtype=torch.float32).to(device)
    mask_train_tensor = torch.tensor(mask_train, dtype=torch.bool).to(device)
    y_train_tensor = torch.tensor(y_sub_train, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Class-weighted loss
    n_pos = (y_sub_train == 1).sum()
    n_neg = (y_sub_train == 0).sum()
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32).to(device)
    
    model = SepsisTransformer(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    train_losses = []
    val_losses = []
    
    dataset_size = len(X_sub_train)
    
    if plot_convergence:
        lengths_val = torch.tensor([len(seq) for seq in X_sub_val], dtype=torch.long)
        max_len_val = max(lengths_val).item()
        X_val_padded = np.zeros((len(X_sub_val), max_len_val, input_dim), dtype=np.float32)
        mask_val = np.ones((len(X_sub_val), max_len_val), dtype=bool)
        for idx, seq in enumerate(X_sub_val):
            X_val_padded[idx, :len(seq), :] = seq
            mask_val[idx, :len(seq)] = False
        X_val_tensor = torch.tensor(X_val_padded, dtype=torch.float32).to(device)
        mask_val_tensor = torch.tensor(mask_val, dtype=torch.bool).to(device)
        y_val_tensor = torch.tensor(y_sub_val, dtype=torch.float32).unsqueeze(1).to(device)
        
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(dataset_size)
        epoch_loss = 0.0
        batches = 0
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
            
            epoch_loss += loss.item()
            batches += 1
            
        train_losses.append(epoch_loss / max(1, batches))
        
        if plot_convergence:
            model.eval()
            with torch.no_grad():
                logits_val = model(X_val_tensor, mask_val_tensor)
                loss_val = criterion(logits_val, y_val_tensor)
                val_losses.append(loss_val.item())
                
    if plot_convergence and plot_path:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", color="tab:red")
        plt.plot(range(1, epochs + 1), val_losses, label="Val Loss", color="tab:blue", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("Transformer Predictor Model Convergence (Septic Shock Prediction)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved Transformer convergence plot to: {plot_path}")
        
    # Re-train on FULL training dataset to maintain the exact same behavior as original code!
    if plot_convergence:
        lengths_full = torch.tensor([len(seq) for seq in X_train], dtype=torch.long)
        max_len_full = max(lengths_full).item()
        X_full_padded = np.zeros((len(X_train), max_len_full, input_dim), dtype=np.float32)
        mask_full = np.ones((len(X_train), max_len_full), dtype=bool)
        for idx, seq in enumerate(X_train):
            X_full_padded[idx, :len(seq), :] = seq
            mask_full[idx, :len(seq)] = False
            
        X_full_tensor = torch.tensor(X_full_padded, dtype=torch.float32).to(device)
        mask_full_tensor = torch.tensor(mask_full, dtype=torch.bool).to(device)
        y_full_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        
        n_pos_full = (y_train == 1).sum()
        n_neg_full = (y_train == 0).sum()
        pos_weight_full = torch.tensor([n_neg_full / max(1, n_pos_full)], dtype=torch.float32).to(device)
        
        model = SepsisTransformer(input_dim=input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion_full = nn.BCEWithLogitsLoss(pos_weight=pos_weight_full)
        
        model.train()
        full_size = len(X_train)
        for epoch in range(epochs):
            permutation = torch.randperm(full_size)
            for i in range(0, full_size, batch_size):
                indices = permutation[i:i+batch_size]
                batch_x = X_full_tensor[indices]
                batch_mask = mask_full_tensor[indices]
                batch_y = y_full_tensor[indices]
                
                optimizer.zero_grad()
                logits = model(batch_x, batch_mask)
                loss = criterion_full(logits, batch_y)
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
    return probs

def find_default_mimic_npz():
    env_dir = os.environ.get("MIMIC_DATASET_DIR", "")
    if env_dir and os.path.exists(os.path.join(env_dir, "mimic_lazy_12_clean_with_interventions_corrected.npz")):
        return os.path.join(env_dir, "mimic_lazy_12_clean_with_interventions_corrected.npz")
    for candidate_dir in [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../in/datasets/MIMIC 2")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../in/datasets")),
        os.path.abspath(os.path.join(os.getcwd(), "in/datasets/MIMIC 2")),
        os.path.abspath(os.path.join(os.getcwd(), "in/datasets")),
        "/Users/cameronegbert/Documents/NCSU/Research/datasets/MIMIC 2",
        "/mnt/beegfs/cegbert/NeSyRL/in/datasets/MIMIC 2",
        "/mnt/beegfs/cegbert/NeSyRL/in/datasets",
        "/mnt/beegfs/cegbert/MIMIC 2"
    ]:
        candidate_file = os.path.join(candidate_dir, "mimic_lazy_12_clean_with_interventions_corrected.npz")
        if os.path.exists(candidate_file):
            return candidate_file
    return "/Users/cameronegbert/Documents/NCSU/Research/datasets/MIMIC 2/mimic_lazy_12_clean_with_interventions_corrected.npz"

def main():
    parser = argparse.ArgumentParser(description="Controlled Septic Shock Early Prediction Sweep with Fixed Cohort")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/mimic/tune_mimic_cql", help="Path to CQL agent checkpoints")
    parser.add_argument("--dataset-path", type=str, default=find_default_mimic_npz(), help="Path to MIMIC dataset")
    parser.add_argument("--tau-min", type=int, default=1, help="Minimum tau in hours")
    parser.add_argument("--tau-max", type=int, default=36, help="Maximum tau in hours")
    parser.add_argument("--tau-step", type=int, default=4, help="Step size for tau sweep in hours")
    parser.add_argument("--window-hours", type=int, default=12, help="Fixed observation window length in hours (default: 12)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs for each model")
    parser.add_argument("--n-splits", "--n-models", type=int, dest="n_splits", default=20, help="Number of data splits to evaluate (mean and SEM will be computed across splits)")
    parser.add_argument("--output-dir", type=str, default="results/plots/early_prediction", help="Directory to save plots")
    args = parser.parse_args()

    w_steps = 2 * args.window_hours
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Observation window: {args.window_hours} hours ({w_steps} steps)")
    print(f"Evaluation configuration: training and averaging across {args.n_splits} splits per setup.")

    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}")
    data = np.load(args.dataset_path, allow_pickle=True)
    X = data['X']
    y = data['y'].squeeze()
    mask = data['mask']
    
    # Filter cohort globally: Only keep patients with stays >= 2 * tau_max + w_steps to ensure a full window at max tau
    min_stay_steps = 2 * args.tau_max + w_steps
    cohort_indices = []
    cohort_t_lengths = []
    
    for i in range(len(X)):
        valid_steps = np.where(mask[i].squeeze() != -1)[0]
        if len(valid_steps) >= min_stay_steps:
            cohort_indices.append(i)
            cohort_t_lengths.append(len(valid_steps))
            
    cohort_indices = np.array(cohort_indices)
    cohort_t_lengths = np.array(cohort_t_lengths)
    y_cohort = y[cohort_indices]
    
    print(f"Global Cohort Filter (stays >= {args.tau_max}h + {args.window_hours}h window = {min_stay_steps} steps): {len(cohort_indices)} patients remaining (out of {len(X)})")
    
    # Load CQL agent robustly
    checkpoint_arg = Path(args.checkpoint)
    if checkpoint_arg.is_dir():
        candidates = list(checkpoint_arg.glob("best_model*.ckpt"))
        if not candidates:
            candidates = list(checkpoint_arg.rglob("best_model*.ckpt"))
        if candidates:
            def extract_version(p):
                name = p.stem
                if "-" in name:
                    parts = name.split("-v")
                    if len(parts) > 1 and parts[1].isdigit():
                        return int(parts[1])
                return -1
            candidates.sort(key=extract_version)
            checkpoint_path = candidates[-1]
            print(f"Directory mode: Selected latest checkpoint {checkpoint_path}")
        else:
            raise FileNotFoundError(f"No best_model*.ckpt files found under {checkpoint_arg}")
    else:
        checkpoint_path = checkpoint_arg
        if not checkpoint_path.exists():
            dirpath = checkpoint_path.parent
            if dirpath.exists():
                candidates = list(dirpath.glob("best_model*.ckpt"))
                if candidates:
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    checkpoint_path = candidates[0]
                    print(f"Redirecting to candidate: {checkpoint_path}")
                else:
                    raise FileNotFoundError(f"No best_model*.ckpt files under {dirpath}")
            else:
                raise FileNotFoundError(f"No checkpoint or directory found at {checkpoint_path}")

    # Allow loading dictconfig safelist for torch.load
    torch.serialization.add_safe_globals([
        getattr(sys.modules.get('omegaconf.dictconfig', None), 'DictConfig', None)
    ])

    print(f"Loading CQL agent from: {checkpoint_path}")
    cql_agent = CQLAgent.load_from_checkpoint(str(checkpoint_path), map_location=device, weights_only=False)
    cql_agent.eval()

    # Pre-compute CQL state values V(s) = max_a Q(s, a) for the cohort
    print("Pre-computing CQL state-value functions V(s)...")
    v_vals_all = {}
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(cohort_indices), batch_size):
            batch_indices = cohort_indices[i:i+batch_size]
            batch_x = X[batch_indices, :, :46]
            batch_x_tensor = torch.tensor(batch_x, dtype=torch.float32).to(device)
            B_curr = batch_x_tensor.size(0)
            flat_x = batch_x_tensor.view(-1, 46)
            flat_q = cql_agent.q_network(flat_x)
            q_vals = flat_q.view(B_curr, 240, 2)
            v_vals = torch.max(q_vals, dim=-1)[0].unsqueeze(-1).cpu().numpy()
            
            for idx_in_batch, original_idx in enumerate(batch_indices):
                v_vals_all[original_idx] = v_vals[idx_in_batch]
    
    # Feature configurations with fixed window W ending at t: X[idx, max(0, t-w_steps):t, :]
    feat_configs = {
        "no_v": lambda idx, t: X[idx, max(0, t - w_steps):t, :49],
        "with_v": lambda idx, t: np.concatenate([X[idx, max(0, t - w_steps):t, :49], v_vals_all[idx][max(0, t - w_steps):t]], axis=-1),
    }

    # Set up tau sweep list
    tau_list = list(range(args.tau_min, args.tau_max + 1, args.tau_step))
    print(f"Sweeping tau (hours early): {tau_list}")

    model_configs = [
        ("LSTM (no V)", "lstm", "no_v"),
        ("LSTM (with V)", "lstm", "with_v"),
        ("Transformer (no V)", "transformer", "no_v"),
        ("Transformer (with V)", "transformer", "with_v"),
    ]
    
    results = {}
    for m_cfg, _, _ in model_configs:
        results[m_cfg] = {
            "tau": [],
            "auc": [], "auc_sem": [],
            "auprc": [], "auprc_sem": [],
            "f1_max": [], "f1_max_sem": [],
            "f1_05": [], "f1_05_sem": []
        }
        
    results_losses = {}

    # Run the sweep
    for tau in tau_list:
        print(f"\n--- Evaluating Lead Time: {tau} Hours Early ---")
        steps_early = 2 * tau
        
        # Slices are constructed using the pre-filtered cohort details
        # For each patient in the cohort, the cutoff time is len(valid_steps) - steps_early
        t_cutoffs = cohort_t_lengths - steps_early
        results_losses[tau] = {}
        
        for m_cfg_name, m_type, feat_key in model_configs:
            print(f"  Evaluating: {m_cfg_name} across {args.n_splits} splits")
            feat_func = feat_configs[feat_key]
            
            # Construct patient sliced sequences
            seq_data = [feat_func(cohort_indices[i], t_cutoffs[i]) for i in range(len(cohort_indices))]
            input_dim = seq_data[0].shape[-1]
            
            split_aucs = []
            split_auprcs = []
            split_f1_maxes = []
            split_f1_05s = []
            results_losses[tau][m_cfg_name] = []
            
            for m_idx in range(args.n_splits):
                seed_val = 42 + m_idx
                # Split train/test (80/20) for this specific split
                train_cohort_idxs, test_cohort_idxs = train_test_split(
                    np.arange(len(cohort_indices)), test_size=0.2, random_state=seed_val, stratify=y_cohort
                )
                
                X_train = [seq_data[i] for i in train_cohort_idxs]
                X_test = [seq_data[i] for i in test_cohort_idxs]
                y_train = y_cohort[train_cohort_idxs]
                y_test = y_cohort[test_cohort_idxs]
                
                if m_type == "lstm":
                    model, train_losses = train_lstm_model(
                        X_train, y_train, input_dim, epochs=args.epochs, device=device, seed=seed_val
                    )
                    probs = evaluate_lstm_model(model, X_test, input_dim, device=device)
                elif m_type == "transformer":
                    model, train_losses = train_transformer_model(
                        X_train, y_train, input_dim, epochs=args.epochs, device=device, seed=seed_val
                    )
                    probs = evaluate_transformer_model(model, X_test, input_dim, device=device)
                
                results_losses[tau][m_cfg_name].append(train_losses)
                
                # Metrics for this split
                auc_roc = float(roc_auc_score(y_test, probs))
                precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
                auprc_val = float(auc(recalls, precisions))
                
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
                f1_max_val = float(np.max(f1_scores))
                
                preds_05 = (probs > 0.5).astype(np.int32)
                f1_05_val = float(f1_score(y_test, preds_05, zero_division=0))
                
                split_aucs.append(auc_roc)
                split_auprcs.append(auprc_val)
                split_f1_maxes.append(f1_max_val)
                split_f1_05s.append(f1_05_val)
                
            auc_mean = float(np.mean(split_aucs))
            auc_sem = float(np.std(split_aucs) / np.sqrt(args.n_splits))
            
            auprc_mean = float(np.mean(split_auprcs))
            auprc_sem = float(np.std(split_auprcs) / np.sqrt(args.n_splits))
            
            f1_max_mean = float(np.mean(split_f1_maxes))
            f1_max_sem = float(np.std(split_f1_maxes) / np.sqrt(args.n_splits))
            
            f1_05_mean = float(np.mean(split_f1_05s))
            f1_05_sem = float(np.std(split_f1_05s) / np.sqrt(args.n_splits))
            
            results[m_cfg_name]["tau"].append(tau)
            results[m_cfg_name]["auc"].append(auc_mean)
            results[m_cfg_name]["auc_sem"].append(auc_sem)
            results[m_cfg_name]["auprc"].append(auprc_mean)
            results[m_cfg_name]["auprc_sem"].append(auprc_sem)
            results[m_cfg_name]["f1_max"].append(f1_max_mean)
            results[m_cfg_name]["f1_max_sem"].append(f1_max_sem)
            results[m_cfg_name]["f1_05"].append(f1_05_mean)
            results[m_cfg_name]["f1_05_sem"].append(f1_05_sem)
            
            print(f"    AUC-ROC: {auc_mean:.4f} ± {auc_sem:.4f}, AUPRC: {auprc_mean:.4f} ± {auprc_sem:.4f}, F1-Max (optimal): {f1_max_mean:.4f} ± {f1_max_sem:.4f}, F1-0.5: {f1_05_mean:.4f} ± {f1_05_sem:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save text results
    results_file = out_dir / "early_prediction_dl_results.txt"
    with open(results_file, "w") as f:
        f.write(f"=== Septic Shock Early Prediction DL Sweep Results over {args.n_splits} Splits ===\n")
        f.write(f"Cohort Constraint: Patients with stays >= {args.tau_max}h + {args.window_hours}h window (Fixed size = {len(cohort_indices)})\n\n")
        for m_cfg_name, _, _ in model_configs:
            f.write(f"Model Configuration: {m_cfg_name}\n")
            f.write(f"  Taus:   {results[m_cfg_name]['tau']}\n")
            f.write(f"  AUCs:   {results[m_cfg_name]['auc']} (SEMs: {results[m_cfg_name]['auc_sem']})\n")
            f.write(f"  AUPRCs: {results[m_cfg_name]['auprc']} (SEMs: {results[m_cfg_name]['auprc_sem']})\n")
            f.write(f"  F1_max: {results[m_cfg_name]['f1_max']} (SEMs: {results[m_cfg_name]['f1_max_sem']})\n")
            f.write(f"  F1_0.5: {results[m_cfg_name]['f1_05']} (SEMs: {results[m_cfg_name]['f1_05_sem']})\n\n")

    # Plot 4-panel results
    print(f"Plotting 4-panel results and saving to {out_dir}...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    markers = ['o', 's', '^', 'D']
    
    # Plot 1: AUC-ROC
    for idx, (m_cfg_name, _, _) in enumerate(model_configs):
        res = results[m_cfg_name]
        tau_arr = np.array(res["tau"])
        mean_arr = np.array(res["auc"])
        sem_arr = np.array(res["auc_sem"])
        axes[0, 0].plot(tau_arr, mean_arr, marker=markers[idx], color=colors[idx], label=m_cfg_name, linewidth=2)
        axes[0, 0].fill_between(tau_arr, mean_arr - sem_arr, mean_arr + sem_arr, color=colors[idx], alpha=0.15)
    axes[0, 0].set_title(f"AUC-ROC vs. Lead Time (\u03c4)", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Lead Time (hours early - \u03c4)", fontsize=11)
    axes[0, 0].set_ylabel("AUC-ROC", fontsize=11)
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    axes[0, 0].legend(fontsize=10)
    
    # Plot 2: AUPRC
    for idx, (m_cfg_name, _, _) in enumerate(model_configs):
        res = results[m_cfg_name]
        tau_arr = np.array(res["tau"])
        mean_arr = np.array(res["auprc"])
        sem_arr = np.array(res["auprc_sem"])
        axes[0, 1].plot(tau_arr, mean_arr, marker=markers[idx], color=colors[idx], label=m_cfg_name, linewidth=2)
        axes[0, 1].fill_between(tau_arr, mean_arr - sem_arr, mean_arr + sem_arr, color=colors[idx], alpha=0.15)
    axes[0, 1].set_title(f"AUPRC (PR-AUC) vs. Lead Time (\u03c4)", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Lead Time (hours early - \u03c4)", fontsize=11)
    axes[0, 1].set_ylabel("AUPRC", fontsize=11)
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)
    axes[0, 1].legend(fontsize=10)
    
    # Plot 3: F1-Max (Optimal Threshold)
    for idx, (m_cfg_name, _, _) in enumerate(model_configs):
        res = results[m_cfg_name]
        tau_arr = np.array(res["tau"])
        mean_arr = np.array(res["f1_max"])
        sem_arr = np.array(res["f1_max_sem"])
        axes[1, 0].plot(tau_arr, mean_arr, marker=markers[idx], color=colors[idx], label=m_cfg_name, linewidth=2)
        axes[1, 0].fill_between(tau_arr, mean_arr - sem_arr, mean_arr + sem_arr, color=colors[idx], alpha=0.15)
    axes[1, 0].set_title(f"F1-Max (Optimal Threshold \u03b8*) vs. Lead Time (\u03c4)", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Lead Time (hours early - \u03c4)", fontsize=11)
    axes[1, 0].set_ylabel("Optimal F1-Score", fontsize=11)
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    axes[1, 0].legend(fontsize=10)

    # Plot 4: F1 at 0.5 Threshold
    for idx, (m_cfg_name, _, _) in enumerate(model_configs):
        res = results[m_cfg_name]
        tau_arr = np.array(res["tau"])
        mean_arr = np.array(res["f1_05"])
        sem_arr = np.array(res["f1_05_sem"])
        axes[1, 1].plot(tau_arr, mean_arr, marker=markers[idx], color=colors[idx], label=m_cfg_name, linewidth=2)
        axes[1, 1].fill_between(tau_arr, mean_arr - sem_arr, mean_arr + sem_arr, color=colors[idx], alpha=0.15)
    axes[1, 1].set_title(f"F1-Score (Standard Threshold \u03b8=0.5) vs. Lead Time (\u03c4)", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Lead Time (hours early - \u03c4)", fontsize=11)
    axes[1, 1].set_ylabel("F1-Score (\u03b8=0.5)", fontsize=11)
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)
    axes[1, 1].legend(fontsize=10)
    
    plt.tight_layout()
    plot_path = out_dir / "early_prediction_dl_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved 4-panel comparison plot: {plot_path}")
    
    # Plot composite training convergence curves for all tau values on one single PNG
    print("Plotting composite training convergence curves...")
    num_taus = len(tau_list)
    cols = 3
    rows = math.ceil(num_taus / cols)
    fig_conv, axes_conv = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    
    # Ensure axes_conv is a flat array even if rows/cols = 1
    if num_taus == 1:
        axes_conv_flat = [axes_conv]
    else:
        axes_conv_flat = axes_conv.flatten()
        
    for tau_idx, tau in enumerate(tau_list):
        ax = axes_conv_flat[tau_idx]
        for m_idx, (m_cfg_name, _, _) in enumerate(model_configs):
            loss_lists = results_losses[tau][m_cfg_name]
            mean_losses = np.mean(loss_lists, axis=0)
            ax.plot(range(1, args.epochs + 1), mean_losses, color=colors[m_idx], label=m_cfg_name, linewidth=1.5)
            
        ax.set_title(f"Lead Time \u03c4 = {tau}h", fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Avg BCE Loss", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.6)
        if tau_idx == 0:
            ax.legend(fontsize=8, loc="upper right")
            
    # Hide unused subplots
    for idx in range(num_taus, len(axes_conv_flat)):
        fig_conv.delaxes(axes_conv_flat[idx])
        
    plt.tight_layout()
    conv_plot_path = out_dir / "predictor_convergence_all_taus.png"
    plt.savefig(conv_plot_path, dpi=200)
    plt.close()
    print(f"Saved composite convergence plot: {conv_plot_path}")
    
    print("Early prediction deep learning evaluation finished successfully!")

if __name__ == "__main__":
    main()
