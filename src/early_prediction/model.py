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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

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

# --- Improved Transformer Classifier Model with Dual Pooling & LayerNorm ---
class SepsisTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, use_dual_pooling=True):
        super().__init__()
        self.use_dual_pooling = use_dual_pooling
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.input_layer_norm = nn.LayerNorm(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        in_features = d_model * 2 if use_dual_pooling else d_model
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x, padding_mask):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.input_layer_norm(x)
        out = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        valid_lens = (~padding_mask).sum(dim=1).clamp(min=1)
        last_indices = valid_lens - 1
        last_repr = out[torch.arange(out.size(0)), last_indices]
        
        if self.use_dual_pooling:
            valid_mask_expanded = (~padding_mask).unsqueeze(-1).float()
            sum_repr = (out * valid_mask_expanded).sum(dim=1)
            mean_repr = sum_repr / valid_lens.unsqueeze(1).float()
            pooled = torch.cat([last_repr, mean_repr], dim=-1)
        else:
            pooled = last_repr
            
        logits = self.classifier(pooled)
        return logits

# --- Improved PyTorch LSTM Model with Dual Pooling & Multi-layer ---
class SepsisLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2, use_dual_pooling=True):
        super().__init__()
        self.use_dual_pooling = use_dual_pooling
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        in_features = hidden_dim * 2 if use_dual_pooling else hidden_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, (hn, _) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        
        last_hn = hn[-1]
        
        if self.use_dual_pooling:
            B, L, H = out.size()
            mask_t = torch.arange(L, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            valid_mask = mask_t.unsqueeze(-1).float()
            mean_out = (out * valid_mask).sum(dim=1) / lengths.unsqueeze(1).float()
            pooled = torch.cat([last_hn, mean_out], dim=-1)
        else:
            pooled = last_hn
            
        logits = self.classifier(pooled)
        return logits

def normalize_features(X_train_list, X_test_list):
    all_steps = np.concatenate([s for s in X_train_list], axis=0)
    mean = np.mean(all_steps, axis=0, keepdims=True)
    std = np.std(all_steps, axis=0, keepdims=True) + 1e-6
    return [(s - mean) / std for s in X_train_list], [(s - mean) / std for s in X_test_list]

# --- Training Helper Functions ---
def train_lstm_model(X_train, y_train, input_dim, hidden_dim=64, num_layers=2, epochs=15, batch_size=64, lr=1e-3, device="cpu", seed=42, use_norm=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    lengths_train = torch.tensor([len(seq) for seq in X_train], dtype=torch.long)
    max_len = max(lengths_train).item()
    
    X_train_padded = np.zeros((len(X_train), max_len, input_dim), dtype=np.float32)
    for idx, seq in enumerate(X_train):
        X_train_padded[idx, :len(seq), :] = seq
        
    X_train_tensor = torch.tensor(X_train_padded, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Class-weighted loss
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32).to(device)
    
    model = SepsisLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    train_losses = []
    dataset_size = len(X_train)
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        train_losses.append(epoch_loss / max(1, batches))
        
    return model, train_losses

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

def train_transformer_model(X_train, y_train, input_dim, d_model=64, nhead=4, num_layers=2, epochs=15, batch_size=64, lr=1e-3, device="cpu", seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
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
    
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32).to(device)
    
    model = SepsisTransformer(input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=d_model*2, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    train_losses = []
    dataset_size = len(X_train)
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        train_losses.append(epoch_loss / max(1, batches))
        
    return model, train_losses

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
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../in/datasets/MIMIC 2")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../in/datasets")),
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
    return "in/datasets/MIMIC 2/mimic_lazy_12_clean_with_interventions_corrected.npz"

def main():
    parser = argparse.ArgumentParser(description="Controlled Septic Shock Early Prediction Sweep with Fixed Cohort")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/mimic/tune_mimic_cql", help="Path to CQL agent checkpoints (optional for V(s))")
    parser.add_argument("--dataset-path", type=str, default=find_default_mimic_npz(), help="Path to MIMIC dataset")
    parser.add_argument("--tau-min", type=int, default=1, help="Minimum tau in hours")
    parser.add_argument("--tau-max", type=int, default=36, help="Maximum tau in hours")
    parser.add_argument("--tau-step", type=int, default=4, help="Step size for tau sweep in hours")
    parser.add_argument("--window-hours", type=int, default=12, help="Observation window length in hours if use-all-history is false (default: 12)")
    parser.add_argument("--use-all-history", action="store_true", default=True, help="Use full observation sequence from t=0 to cutoff instead of fixed window (default: True)")
    parser.add_argument("--fixed-window-history", dest="use_all_history", action="store_false", help="Use fixed observation window instead of full history")
    parser.add_argument("--use-all-trajectories", action="store_true", default=True, help="Use all valid trajectories for each tau instead of restricted cohort filter (default: True)")
    parser.add_argument("--restricted-cohort", dest="use_all_trajectories", action="store_false", help="Restrict cohort to stays >= tau_max + window_hours")
    parser.add_argument("--use-norm", action="store_true", default=True, help="Apply feature standardization per split (default: True)")
    parser.add_argument("--no-norm", dest="use_norm", action="store_false", help="Disable feature standardization")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs for each model (default: 20)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer embedding dimension (default: 64)")
    parser.add_argument("--nhead", type=int, default=4, help="Transformer attention heads (default: 4)")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers for Transformer / LSTM (default: 2)")
    parser.add_argument("--hidden-dim", type=int, default=64, help="LSTM hidden dimension (default: 64)")
    parser.add_argument("--n-splits", "--n-models", type=int, dest="n_splits", default=100, help="Number of data splits to evaluate (default: 100)")
    parser.add_argument("--output-dir", type=str, default="results/plots/early_prediction", help="Directory to save plots")
    args = parser.parse_args()

    w_steps = 2 * args.window_hours
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Observation configuration: {'Full History (t=0 to cutoff)' if args.use_all_history else f'{args.window_hours}h window ({w_steps} steps)'}")
    print(f"Cohort selection: {'Dynamic per-tau (All valid trajectories)' if args.use_all_trajectories else f'Restricted global cohort (stays >= {args.tau_max}h + {args.window_hours}h window)'}")
    print(f"Evaluation configuration: training and averaging across {args.n_splits} splits per setup.")

    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}")
    data = np.load(args.dataset_path, allow_pickle=True)
    X = data['X']
    y = data['y'].squeeze()
    mask = data['mask']
    
    patient_lengths = np.array([(mask[i].squeeze() != -1).sum() for i in range(len(X))])
    
    # Pre-compute CQL state values V(s) if checkpoint exists
    v_vals_all = None
    checkpoint_arg = Path(args.checkpoint)
    cql_ckpt_path = None
    if checkpoint_arg.is_dir():
        candidates = list(checkpoint_arg.glob("best_model*.ckpt"))
        if candidates:
            cql_ckpt_path = str(candidates[-1])
    elif checkpoint_arg.exists():
        cql_ckpt_path = str(checkpoint_arg)
        
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
            batch_sz = 128
            with torch.no_grad():
                for i in range(0, len(X), batch_sz):
                    batch_x = torch.tensor(X[i:i+batch_sz, :, :46], dtype=torch.float32).to(device)
                    B_curr = batch_x.size(0)
                    flat_x = batch_x.view(-1, 46)
                    flat_q = cql_agent.q_network(flat_x)
                    q_vals = flat_q.view(B_curr, 240, 2)
                    v_vals = torch.max(q_vals, dim=-1)[0].unsqueeze(-1).cpu().numpy()
                    v_vals_all[i:i+batch_sz] = v_vals
            print("CQL V(s) pre-computation complete.")
        except Exception as e:
            print(f"Warning: Could not compute V(s) from checkpoint: {e}")

    # Set up tau sweep list
    tau_list = list(range(args.tau_min, args.tau_max + 1, args.tau_step))
    print(f"Sweeping tau (hours early): {tau_list}")

    model_configs = [
        ("LSTM (no V)", "lstm", False),
        ("LSTM (with V)", "lstm", True),
        ("Transformer (no V)", "transformer", False),
        ("Transformer (with V)", "transformer", True),
    ]
    
    results = {}
    for m_cfg, _, _ in model_configs:
        results[m_cfg] = {
            "tau": [],
            "auc": [], "auc_sem": [],
            "auprc": [], "auprc_sem": [],
            "f1_opt": [], "f1_opt_sem": [],
            "f1_max": [], "f1_max_sem": [],
            "f1_05": [], "f1_05_sem": []
        }
        
    results_losses = {}

    # Run the sweep
    for tau in tau_list:
        print(f"\n--- Evaluating Lead Time: {tau} Hours Early ---")
        steps_early = 2 * tau
        
        # Select cohort dynamically or fixed
        if args.use_all_trajectories:
            c_indices = np.array([i for i in range(len(X)) if patient_lengths[i] - steps_early >= 1])
        else:
            min_stay_steps = 2 * args.tau_max + w_steps
            c_indices = np.array([i for i in range(len(X)) if patient_lengths[i] >= min_stay_steps])
            
        t_cutoffs = patient_lengths[c_indices] - steps_early
        y_cohort = y[c_indices]
        print(f"  Cohort size at tau={tau}h: {len(c_indices)} patients (positive ratio: {y_cohort.mean():.3f})")
        results_losses[tau] = {}
        
        for m_cfg_name, m_type, use_v_feat in model_configs:
            print(f"  Evaluating: {m_cfg_name} across {args.n_splits} splits")
            
            # Construct patient sliced sequences
            seq_data = []
            for i, original_idx in enumerate(c_indices):
                tc = t_cutoffs[i]
                st = 0 if args.use_all_history else max(0, tc - w_steps)
                raw_seq = X[original_idx, st:tc, :49]
                if use_v_feat and v_vals_all is not None:
                    v_seq = v_vals_all[original_idx, st:tc]
                    seq_data.append(np.concatenate([raw_seq, v_seq], axis=-1))
                else:
                    seq_data.append(raw_seq)
                    
            input_dim = seq_data[0].shape[-1]
            
            split_aucs = []
            split_auprcs = []
            split_f1_opts = []
            split_f1_maxes = []
            split_f1_05s = []
            results_losses[tau][m_cfg_name] = []
            
            for m_idx in range(args.n_splits):
                seed_val = 42 + m_idx
                train_cohort_idxs, test_cohort_idxs = train_test_split(
                    np.arange(len(c_indices)), test_size=0.2, random_state=seed_val, stratify=y_cohort
                )
                
                X_train = [seq_data[i] for i in train_cohort_idxs]
                X_test = [seq_data[i] for i in test_cohort_idxs]
                y_train = y_cohort[train_cohort_idxs]
                y_test = y_cohort[test_cohort_idxs]
                
                if args.use_norm:
                    X_train, X_test = normalize_features(X_train, X_test)
                
                if m_type == "lstm":
                    model, train_losses = train_lstm_model(
                        X_train, y_train, input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device, seed=seed_val
                    )
                    probs_test = evaluate_lstm_model(model, X_test, input_dim, device=device)
                    probs_train = evaluate_lstm_model(model, X_train, input_dim, device=device)
                elif m_type == "transformer":
                    model, train_losses = train_transformer_model(
                        X_train, y_train, input_dim, d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
                        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device, seed=seed_val
                    )
                    probs_test = evaluate_transformer_model(model, X_test, input_dim, device=device)
                    probs_train = evaluate_transformer_model(model, X_train, input_dim, device=device)
                
                results_losses[tau][m_cfg_name].append(train_losses)
                
                # Metrics for this split
                auc_roc = float(roc_auc_score(y_test, probs_test))
                precisions, recalls, thresholds = precision_recall_curve(y_test, probs_test)
                auprc_val = float(auc(recalls, precisions))
                
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
                f1_max_val = float(np.max(f1_scores))
                
                # Calculate optimal threshold on train set
                tr_prec, tr_rec, tr_thresh = precision_recall_curve(y_train, probs_train)
                tr_f1 = 2 * (tr_prec * tr_rec) / (tr_prec + tr_rec + 1e-8)
                best_tr_idx = np.argmax(tr_f1)
                opt_thresh = tr_thresh[best_tr_idx] if best_tr_idx < len(tr_thresh) else 0.5
                
                preds_opt = (probs_test >= opt_thresh).astype(np.int32)
                f1_opt_val = float(f1_score(y_test, preds_opt, zero_division=0))
                
                preds_05 = (probs_test >= 0.5).astype(np.int32)
                f1_05_val = float(f1_score(y_test, preds_05, zero_division=0))
                
                split_aucs.append(auc_roc)
                split_auprcs.append(auprc_val)
                split_f1_opts.append(f1_opt_val)
                split_f1_maxes.append(f1_max_val)
                split_f1_05s.append(f1_05_val)
                
            auc_mean = float(np.mean(split_aucs))
            auc_sem = float(np.std(split_aucs) / np.sqrt(args.n_splits))
            
            auprc_mean = float(np.mean(split_auprcs))
            auprc_sem = float(np.std(split_auprcs) / np.sqrt(args.n_splits))
            
            f1_opt_mean = float(np.mean(split_f1_opts))
            f1_opt_sem = float(np.std(split_f1_opts) / np.sqrt(args.n_splits))
            
            f1_max_mean = float(np.mean(split_f1_maxes))
            f1_max_sem = float(np.std(split_f1_maxes) / np.sqrt(args.n_splits))
            
            f1_05_mean = float(np.mean(split_f1_05s))
            f1_05_sem = float(np.std(split_f1_05s) / np.sqrt(args.n_splits))
            
            results[m_cfg_name]["tau"].append(tau)
            results[m_cfg_name]["auc"].append(auc_mean)
            results[m_cfg_name]["auc_sem"].append(auc_sem)
            results[m_cfg_name]["auprc"].append(auprc_mean)
            results[m_cfg_name]["auprc_sem"].append(auprc_sem)
            results[m_cfg_name]["f1_opt"].append(f1_opt_mean)
            results[m_cfg_name]["f1_opt_sem"].append(f1_opt_sem)
            results[m_cfg_name]["f1_max"].append(f1_max_mean)
            results[m_cfg_name]["f1_max_sem"].append(f1_max_sem)
            results[m_cfg_name]["f1_05"].append(f1_05_mean)
            results[m_cfg_name]["f1_05_sem"].append(f1_05_sem)
            
            print(f"    AUC-ROC: {auc_mean:.4f} ± {auc_sem:.4f}, AUPRC: {auprc_mean:.4f} ± {auprc_sem:.4f}, F1-Opt: {f1_opt_mean:.4f} ± {f1_opt_sem:.4f}, F1-Max: {f1_max_mean:.4f} ± {f1_max_sem:.4f}, F1-0.5: {f1_05_mean:.4f} ± {f1_05_sem:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save text results
    results_file = out_dir / "early_prediction_dl_results.txt"
    with open(results_file, "w") as f:
        f.write(f"=== Septic Shock Early Prediction DL Sweep Results over {args.n_splits} Splits ===\n")
        f.write(f"Observation: {'Full History' if args.use_all_history else f'{args.window_hours}h window'}\n")
        f.write(f"Cohort: {'All valid trajectories per tau' if args.use_all_trajectories else 'Restricted global cohort'}\n\n")
        for m_cfg_name, _, _ in model_configs:
            f.write(f"Model Configuration: {m_cfg_name}\n")
            f.write(f"  Taus:   {results[m_cfg_name]['tau']}\n")
            f.write(f"  AUCs:   {results[m_cfg_name]['auc']} (SEMs: {results[m_cfg_name]['auc_sem']})\n")
            f.write(f"  AUPRCs: {results[m_cfg_name]['auprc']} (SEMs: {results[m_cfg_name]['auprc_sem']})\n")
            f.write(f"  F1_opt: {results[m_cfg_name]['f1_opt']} (SEMs: {results[m_cfg_name]['f1_opt_sem']})\n")
            f.write(f"  F1_max: {results[m_cfg_name]['f1_max']} (SEMs: {results[m_cfg_name]['f1_max_sem']})\n")
            f.write(f"  F1_0.5: {results[m_cfg_name]['f1_05']} (SEMs: {results[m_cfg_name]['f1_05_sem']})\n\n")

    # Save summary markdown report
    summary_file = out_dir / "early_prediction_results_summary.md"
    with open(summary_file, "w") as f:
        f.write("# Septic Shock Early Prediction Deep Learning Results\n\n")
        f.write(f"Evaluating across **{args.n_splits} dataset splits** per setup.\n\n")
        f.write("| Model | Metric | " + " | ".join([f"&tau;={t}h" for t in tau_list]) + " |\n")
        f.write("| :--- | :--- | " + " | ".join([":---:" for _ in tau_list]) + " |\n")
        for m_cfg_name, _, _ in model_configs:
            res = results[m_cfg_name]
            auc_str = " | ".join([f"{m:.4f}±{s:.4f}" for m, s in zip(res['auc'], res['auc_sem'])])
            auprc_str = " | ".join([f"{m:.4f}±{s:.4f}" for m, s in zip(res['auprc'], res['auprc_sem'])])
            f1_opt_str = " | ".join([f"{m:.4f}±{s:.4f}" for m, s in zip(res['f1_opt'], res['f1_opt_sem'])])
            f1_05_str = " | ".join([f"{m:.4f}±{s:.4f}" for m, s in zip(res['f1_05'], res['f1_05_sem'])])
            
            f.write(f"| **{m_cfg_name}** | AUC-ROC | {auc_str} |\n")
            f.write(f"| | AUPRC | {auprc_str} |\n")
            f.write(f"| | F1-Opt | {f1_opt_str} |\n")
            f.write(f"| | F1-0.5 | {f1_05_str} |\n")

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
    
    # Plot 3: F1-Opt (Learned Optimal Threshold)
    for idx, (m_cfg_name, _, _) in enumerate(model_configs):
        res = results[m_cfg_name]
        tau_arr = np.array(res["tau"])
        mean_arr = np.array(res["f1_opt"])
        sem_arr = np.array(res["f1_opt_sem"])
        axes[1, 0].plot(tau_arr, mean_arr, marker=markers[idx], color=colors[idx], label=m_cfg_name, linewidth=2)
        axes[1, 0].fill_between(tau_arr, mean_arr - sem_arr, mean_arr + sem_arr, color=colors[idx], alpha=0.15)
    axes[1, 0].set_title(f"Optimal F1-Score (\u03b8*) vs. Lead Time (\u03c4)", fontsize=12, fontweight='bold')
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
    axes[1, 1].set_title(f"Standard F1-Score (\u03b8=0.5) vs. Lead Time (\u03c4)", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Lead Time (hours early - \u03c4)", fontsize=11)
    axes[1, 1].set_ylabel("F1-Score (\u03b8=0.5)", fontsize=11)
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)
    axes[1, 1].legend(fontsize=10)
    
    plt.tight_layout()
    plot_path = out_dir / "early_prediction_dl_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # Plot 2-panel results (AUC-ROC and Optimal F1)
    print("Plotting 2-panel results (AUC-ROC & Optimal F1)...")
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (m_cfg_name, _, _) in enumerate(model_configs):
        res = results[m_cfg_name]
        tau_arr = np.array(res["tau"])
        mean_arr = np.array(res["auc"])
        sem_arr = np.array(res["auc_sem"])
        axes2[0].plot(tau_arr, mean_arr, marker=markers[idx], color=colors[idx], label=m_cfg_name, linewidth=2)
        axes2[0].fill_between(tau_arr, mean_arr - sem_arr, mean_arr + sem_arr, color=colors[idx], alpha=0.15)
    axes2[0].set_title(f"AUC-ROC vs. Lead Time (\u03c4)", fontsize=13, fontweight='bold')
    axes2[0].set_xlabel("Lead Time (hours early - \u03c4)", fontsize=12)
    axes2[0].set_ylabel("AUC-ROC", fontsize=12)
    axes2[0].grid(True, linestyle="--", alpha=0.6)
    axes2[0].legend(fontsize=10)
    
    for idx, (m_cfg_name, _, _) in enumerate(model_configs):
        res = results[m_cfg_name]
        tau_arr = np.array(res["tau"])
        mean_arr = np.array(res["f1_opt"])
        sem_arr = np.array(res["f1_opt_sem"])
        axes2[1].plot(tau_arr, mean_arr, marker=markers[idx], color=colors[idx], label=m_cfg_name, linewidth=2)
        axes2[1].fill_between(tau_arr, mean_arr - sem_arr, mean_arr + sem_arr, color=colors[idx], alpha=0.15)
    axes2[1].set_title(f"Optimal F1-Score vs. Lead Time (\u03c4)", fontsize=13, fontweight='bold')
    axes2[1].set_xlabel("Lead Time (hours early - \u03c4)", fontsize=12)
    axes2[1].set_ylabel("F1-Score (\u03b8*)", fontsize=12)
    axes2[1].grid(True, linestyle="--", alpha=0.6)
    axes2[1].legend(fontsize=10)
    
    plt.tight_layout()
    plot_path_2panel = out_dir / "early_prediction_dl_comparison_2panel.png"
    plt.savefig(plot_path_2panel, dpi=200)
    plt.close()
    
    print(f"Early prediction evaluation finished successfully! Results saved to {out_dir}")

if __name__ == "__main__":
    main()
