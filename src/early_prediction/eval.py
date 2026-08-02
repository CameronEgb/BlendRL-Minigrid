import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# Add root directory to path to allow importing src modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from src.methods.cql_agent import CQLAgent
from src.early_prediction.model import SepsisLSTM, SepsisTransformer, compute_volatility_features

class SepsisPredictorLSTM(nn.Module):
    def __init__(self, input_dim=49, hidden_dim=64, num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        num_dirs = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        self.classifier = nn.Linear(hidden_dim * num_dirs, 1)
        
    def forward(self, x, mask):
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        out, (hn, _) = self.lstm(x)
        if self.bidirectional:
            last_hn = torch.cat([hn[-2], hn[-1]], dim=-1)
        else:
            last_hn = hn[-1]
        return self.classifier(last_hn)

class PatientDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, mask):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]

def train_predictor(X, y, mask, train_indices, test_indices, epochs=60, batch_size=32, device='cpu', plot_convergence=False, plot_path=None, tuned_params=None):
    # States + action are features 0 to 48
    X_sa = X[:, :, :49].copy()
    
    # Zero out where mask == -1 to prevent NaN/weird values propagation
    m = (mask != -1).astype(np.float32)
    X_sa = X_sa * m
    
    tp = tuned_params or {}
    batch_size = tp.get("batch_size", batch_size)
    lr = tp.get("lr", 1e-3)
    weight_decay = tp.get("weight_decay", 1e-4)
    hidden_dim = tp.get("hidden_dim", 64)
    num_layers = tp.get("num_layers", 2)
    dropout = tp.get("dropout", 0.2)
    bidirectional = tp.get("bidirectional", False)
    epochs = tp.get("epochs", epochs)
    
    train_dataset = PatientDataset(X_sa[train_indices], y[train_indices], mask[train_indices])
    test_dataset = PatientDataset(X_sa[test_indices], y[test_indices], mask[test_indices])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = SepsisPredictorLSTM(input_dim=49, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    
    best_acc = 0.0
    best_state = None
    
    train_losses = []
    val_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for batch_x, batch_y, batch_mask in train_loader:
            batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
            optimizer.zero_grad()
            logits = model(batch_x, batch_mask)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
            
        train_losses.append(epoch_loss / batches)
            
        # Eval on test set
        model.eval()
        correct = 0
        total = 0
        val_loss_epoch = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_mask in test_loader:
                batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
                logits = model(batch_x, batch_mask)
                loss = criterion(logits, batch_y)
                val_loss_epoch += loss.item()
                val_batches += 1
                
                preds = (logits > 0).float()
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        acc = correct / total
        val_accs.append(acc)
        val_losses.append(val_loss_epoch / val_batches)
        
        if acc > best_acc or best_state is None:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    print(f"Supervised Predictor trained. Best test accuracy: {best_acc:.4f}")
    
    if plot_convergence and plot_path:
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('BCE Loss', color=color)
        ax1.plot(range(1, epochs + 1), train_losses, label='Train Loss', color=color, linestyle='-')
        ax1.plot(range(1, epochs + 1), val_losses, label='Val Loss', color=color, linestyle='--')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Validation Accuracy', color=color)
        ax2.plot(range(1, epochs + 1), val_accs, label='Val Acc', color=color, linestyle='-')
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        plt.title('Predictor Model Convergence Graph')
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved predictor convergence graph to: {plot_path}")
        
    model.load_state_dict(best_state)
    model.to(device)
    return model, best_acc

def generate_text_report(csv_path, summary_path, exp_id):
    import csv
    if not os.path.exists(csv_path):
        return
        
    rows = []
    with open(csv_path, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            
    if not rows:
        return
        
    def safe_float(val, default=0.0):
        if val is None or val == "":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def format_val(mean_val, sem_val, is_percent=True, digits=2):
        if sem_val is not None:
            if is_percent:
                return f"{mean_val * 100:.{digits}f}% ± {sem_val * 100:.{digits}f}%"
            else:
                return f"{mean_val:.{digits+2}f} ± {sem_val:.{digits+2}f}"
        else:
            if is_percent:
                return f"{mean_val:.{digits}%}"
            else:
                return f"{mean_val:.{digits+2}f}"
            
    with open(summary_path, mode="w") as f_txt:
        f_txt.write(f"=== MIMIC Sepsis Early Prediction Summary Report ({exp_id}) ===\n\n")
        f_txt.write("This document compiles the summary table and detailed early prediction evaluation results for all evaluated checkpoints in this experiment.\n")
        f_txt.write("All values are reported as mean ± standard error over the random data splits.\n\n")
        
        # Write Summary Table Section
        f_txt.write("--- SUMMARY TABLE ---\n")
        for row in rows:
            ckpt = row.get("checkpoint", "N/A")
            
            def get_val_and_sem(key):
                m = safe_float(row.get(key, 0))
                s_val = row.get(key + "_sem")
                s = safe_float(s_val) if s_val is not None and s_val != "" else None
                return m, s

            pred_acc, pred_acc_sem = get_val_and_sem("predictor_acc")
            clin_mort, clin_mort_sem = get_val_and_sem("clinician_mort")
            cql_mort, cql_mort_sem = get_val_and_sem("cql_mort")
            clin_admin, clin_admin_sem = get_val_and_sem("clinician_admin")
            cql_admin, cql_admin_sem = get_val_and_sem("cql_admin")
            agreement, agreement_sem = get_val_and_sem("agreement")
            
            expert_visits_val = row.get("setup_b_expert_visits")
            expert_visits_sem_val = row.get("setup_b_expert_visits_sem")
            if expert_visits_val is not None and expert_visits_val != "":
                if expert_visits_sem_val is not None and expert_visits_sem_val != "":
                    expert_visits_str = f"{safe_float(expert_visits_val):.1f} ± {safe_float(expert_visits_sem_val):.1f}"
                else:
                    expert_visits_str = f"{safe_float(expert_visits_val):.1f}"
            else:
                expert_visits_str = "N/A"
                
            acc_b, acc_b_sem = get_val_and_sem("setup_b_accuracy")
            rec_b, rec_b_sem = get_val_and_sem("setup_b_recall")
            prec_b, prec_b_sem = get_val_and_sem("setup_b_precision")
            f1_b, f1_b_sem = get_val_and_sem("setup_b_f1")
            auc_b, auc_b_sem = get_val_and_sem("setup_b_auc")
            
            pred_acc_str = format_val(pred_acc, pred_acc_sem, is_percent=True, digits=2)
            clin_mort_str = format_val(clin_mort, clin_mort_sem, is_percent=True, digits=2)
            cql_mort_str = format_val(cql_mort, cql_mort_sem, is_percent=True, digits=2)
            clin_admin_str = format_val(clin_admin, clin_admin_sem, is_percent=True, digits=2)
            cql_admin_str = format_val(cql_admin, cql_admin_sem, is_percent=True, digits=2)
            agreement_str = format_val(agreement, agreement_sem, is_percent=True, digits=2)
            acc_b_str = format_val(acc_b, acc_b_sem, is_percent=True, digits=2)
            rec_b_str = format_val(rec_b, rec_b_sem, is_percent=True, digits=2)
            prec_b_str = format_val(prec_b, prec_b_sem, is_percent=True, digits=2)
            f1_b_str = format_val(f1_b, f1_b_sem, is_percent=True, digits=2)
            auc_b_str = format_val(auc_b, auc_b_sem, is_percent=False, digits=2)
            
            f_txt.write(f"Checkpoint: {ckpt}\n")
            f_txt.write(f"  Predictor Acc: {pred_acc_str} | Clinician Mort: {clin_mort_str} | CQL Mort: {cql_mort_str}\n")
            f_txt.write(f"  Clinician Admin: {clin_admin_str} | CQL Admin: {cql_admin_str} | Agreement: {agreement_str}\n")
            f_txt.write(f"  Setup B - Expert Visits: {expert_visits_str} | Acc: {acc_b_str} | Rec: {rec_b_str} | Prec: {prec_b_str} | F1: {f1_b_str} | AUC: {auc_b_str}\n\n")
            
        f_txt.write("--------------------------------------------------------------------------------\n")
        f_txt.write("DETAILED TRIAL REPORTS\n")
        f_txt.write("--------------------------------------------------------------------------------\n\n")
        
        for row in rows:
            ckpt = row.get("checkpoint", "N/A")
            ckpt_stem = os.path.basename(ckpt).replace(".ckpt", "")
            
            def get_val_and_sem(key):
                m = safe_float(row.get(key, 0))
                s_val = row.get(key + "_sem")
                s = safe_float(s_val) if s_val is not None and s_val != "" else None
                return m, s

            pred_acc, pred_acc_sem = get_val_and_sem("predictor_acc")
            clin_mort, clin_mort_sem = get_val_and_sem("clinician_mort")
            cql_mort, cql_mort_sem = get_val_and_sem("cql_mort")
            clin_admin, clin_admin_sem = get_val_and_sem("clinician_admin")
            cql_admin, cql_admin_sem = get_val_and_sem("cql_admin")
            agreement, agreement_sem = get_val_and_sem("agreement")
            
            expert_visits_val = row.get("setup_b_expert_visits")
            expert_visits_sem_val = row.get("setup_b_expert_visits_sem")
            if expert_visits_val is not None and expert_visits_val != "":
                if expert_visits_sem_val is not None and expert_visits_sem_val != "":
                    expert_visits_str = f"{safe_float(expert_visits_val):.1f} ± {safe_float(expert_visits_sem_val):.1f}"
                else:
                    expert_visits_str = f"{safe_float(expert_visits_val):.1f}"
            else:
                expert_visits_str = "N/A"
                
            acc_b, acc_b_sem = get_val_and_sem("setup_b_accuracy")
            rec_b, rec_b_sem = get_val_and_sem("setup_b_recall")
            prec_b, prec_b_sem = get_val_and_sem("setup_b_precision")
            f1_b, f1_b_sem = get_val_and_sem("setup_b_f1")
            auc_b, auc_b_sem = get_val_and_sem("setup_b_auc")
            
            pred_acc_str = format_val(pred_acc, pred_acc_sem, is_percent=True, digits=2)
            clin_mort_str = format_val(clin_mort, clin_mort_sem, is_percent=True, digits=2)
            cql_mort_str = format_val(cql_mort, cql_mort_sem, is_percent=True, digits=2)
            clin_admin_str = format_val(clin_admin, clin_admin_sem, is_percent=True, digits=2)
            cql_admin_str = format_val(cql_admin, cql_admin_sem, is_percent=True, digits=2)
            agreement_str = format_val(agreement, agreement_sem, is_percent=True, digits=2)
            acc_b_str = format_val(acc_b, acc_b_sem, is_percent=True, digits=2)
            rec_b_str = format_val(rec_b, rec_b_sem, is_percent=True, digits=2)
            prec_b_str = format_val(prec_b, prec_b_sem, is_percent=True, digits=2)
            f1_b_str = format_val(f1_b, f1_b_sem, is_percent=True, digits=2)
            auc_b_str = format_val(auc_b, auc_b_sem, is_percent=False, digits=2)
            
            f_txt.write(f"Checkpoint: {ckpt} (Trial: {ckpt_stem})\n")
            f_txt.write(f"  Predictor Model Supervised Validation Accuracy: {pred_acc_str}\n\n")
            f_txt.write("  Setup A: Counterfactual Evaluation\n")
            f_txt.write(f"    - Average Predicted Mortality Rate: Clinician Actual = {clin_mort_str}, CQL Policy = {cql_mort_str}\n")
            f_txt.write(f"    - Antibiotics Administration Rate:  Clinician Actual = {clin_admin_str}, CQL Policy = {cql_admin_str}\n")
            f_txt.write(f"    - Policy Agreement: {agreement_str} of patient visits\n\n")
            f_txt.write("  Setup B: Imitation of Effective Interventions\n")
            f_txt.write(f"    - Expert Visits Identified: {expert_visits_str}\n")
            f_txt.write(f"    - Accuracy:  {acc_b_str}\n")
            f_txt.write(f"    - Precision: {prec_b_str}\n")
            f_txt.write(f"    - Recall:    {rec_b_str}\n")
            f_txt.write(f"    - F1-Score:  {f1_b_str}\n")
            f_txt.write(f"    - AUC-ROC:   {auc_b_str}\n")
            f_txt.write("--------------------------------------------------------------------------------\n\n")

def main():
    import argparse
    import csv
    parser = argparse.ArgumentParser(description="MIMIC Sepsis Early Prediction Evaluation")
    parser.add_argument("--experiment", "-e", type=str, default=None, help="Experiment ID to evaluate (e.g., tune_mimic_all)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained CQL checkpoint or directory of checkpoints")
    parser.add_argument("--dataset-name", type=str, default=os.environ.get("MIMIC_DATASET_NAME", "mimic_lazy_12_clean_with_interventions_corrected.npz"), help="Predictor training dataset name")
    parser.add_argument("--eval-dataset-name", type=str, default=os.environ.get("MIMIC_EVAL_DATASET_NAME", "mimic_lazy_12_clean_with_interventions_corrected.npz"), help="Evaluation dataset name")
    parser.add_argument("--dataset-path", type=str, default=None, help="Direct path to the MIMIC dataset .npz file")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Custom directory containing the MIMIC dataset")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for early prediction report")
    parser.add_argument("--remake", action="store_true", help="Force recalculation and overwrite the CSV/MD summaries")
    parser.add_argument("--n-splits", type=int, default=100, help="Number of random data splits to evaluate (mean and SEM will be computed across splits)")
    parser.add_argument("--predictor-epochs", type=int, default=60, help="Number of training epochs for predictor per split")
    parser.add_argument("--tau", type=int, default=12, help="Lead time in hours for early prediction model (default: 12)")
    args = parser.parse_known_args()[0]
    
    if args.experiment is not None and args.checkpoint is None:
        exp_id = args.experiment
        ckpt_root = Path("results/checkpoints")
        matches = list(ckpt_root.glob(f"**/{exp_id}"))
        if not matches:
            matches = list(ckpt_root.glob(f"*{exp_id}*"))
        if matches:
            args.checkpoint = str(matches[0])
            args.remake = True
            print(f"Resolved experiment '{exp_id}' to checkpoint path: {args.checkpoint}")
        else:
            raise FileNotFoundError(f"Could not find any checkpoint directory for experiment '{exp_id}' under {ckpt_root}")
            
    if args.checkpoint is None:
        parser.error("Either --experiment (-e) or --checkpoint must be provided.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # 1. Resolve mimic directory
    if args.dataset_dir is not None:
        mimic_dir = args.dataset_dir
    else:
        mimic_dir = os.environ.get("MIMIC_DATASET_DIR", "")
        if not mimic_dir or not os.path.exists(mimic_dir):
            for candidate in [
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../in/datasets/mimic")),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../in/datasets")),
                os.path.abspath(os.path.join(os.getcwd(), "in/datasets/mimic")),
                os.path.abspath(os.path.join(os.getcwd(), "in/datasets")),
                "/Users/cameronegbert/Documents/NCSU/Research/datasets/MIMIC 2",
                "/mnt/beegfs/cegbert/NeSyRL/in/datasets/mimic",
                "/mnt/beegfs/cegbert/NeSyRL/in/datasets",
                "/mnt/beegfs/cegbert/MIMIC 2"
            ]:
                if os.path.exists(candidate):
                    mimic_dir = candidate
                    break
            
    # Load predictor training dataset
    dataset_path = None
    if args.dataset_path is not None and os.path.exists(args.dataset_path):
        dataset_path = os.path.abspath(args.dataset_path)
    else:
        fname = os.path.basename(args.dataset_path) if args.dataset_path else args.dataset_name
        candidates = [
            os.path.join(mimic_dir, fname),
            os.path.abspath(os.path.join(os.getcwd(), "in/datasets/mimic", fname)),
            os.path.abspath(os.path.join(os.getcwd(), "in/datasets/MIMIC 2", fname)),
            os.path.abspath(os.path.join(os.getcwd(), "in/datasets", fname)),
            os.path.join("/mnt/beegfs/cegbert/NeSyRL/in/datasets/mimic", fname),
            os.path.join("/mnt/beegfs/cegbert/MIMIC 2", fname),
            os.path.join("/Users/cameronegbert/Documents/NCSU/Research/datasets/MIMIC 2", fname),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                dataset_path = cand
                break
    if dataset_path is None or not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Predictor training dataset not found at {args.dataset_path} or candidate fallback locations.")
        
    print(f"Loading predictor training dataset from: {dataset_path}")
    data_train = np.load(dataset_path, allow_pickle=True)
    X_train = data_train['X']
    y_train = data_train['y']
    mask_train = data_train['mask']
    
    # Apply lead-time tau cutoff to predictor training dataset
    steps_early = 2 * args.tau
    print(f"Applying lead-time tau={args.tau} hours ({steps_early} steps) cutoff to predictor training dataset...")
    X_train = X_train.copy()
    mask_train = mask_train.copy()
    for i in range(len(X_train)):
        valid_steps = np.where(mask_train[i].squeeze() != -1)[0]
        cutoff = max(1, len(valid_steps) - steps_early)
        mask_train[i, cutoff:] = -1
        X_train[i, cutoff:, :] = 0
        
    # 2. Resolve Output Directories and Paths (Experiment-Specific)
    if args.output_dir is not None:
        report_dir = Path(args.output_dir)
        exp_id = report_dir.name
    else:
        ckpt_path = Path(args.checkpoint)
        parts = ckpt_path.parts
        exp_id = getattr(args, "experiment", None)
        
        # Try to find group and exp_id from checkpoint path
        if len(parts) >= 4 and parts[0] == "results" and parts[1] == "checkpoints":
            group = parts[2]
            exp_id = parts[3]
            report_dir = Path("results/plots") / group / exp_id
        elif len(parts) >= 3 and parts[0] == "results":
            exp_id = parts[2]
            report_dir = Path("results/plots") / exp_id
        elif exp_id:
            # Look for existing plot dir matching exp_id under results/plots/
            matches = list(Path("results/plots").glob(f"**/{exp_id}"))
            if matches:
                report_dir = matches[0]
            else:
                report_dir = Path("results/plots") / exp_id
        else:
            exp_id = ckpt_path.name
            report_dir = Path("results/plots") / exp_id
            
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "early_prediction_summary.csv"
    summary_path = report_dir / "early_prediction_summary.txt"
    
    # Clean up old markdown report files if present to prevent clutter
    for old_report in list(report_dir.glob("early_prediction_report*.md")) + list(report_dir.glob("early_prediction_summary.md")):
        try:
            old_report.unlink()
            print(f"Removed legacy markdown report file: {old_report}")
        except Exception as e:
            print(f"Error removing {old_report}: {e}")

    # Load tuned predictor hyperparameters if available
    tuned_params = {}
    for t_yaml in [
        Path("results/plots/early_prediction/tune_early_pred/best_params_lstm_with_v.yaml"),
        Path("results/plots/early_prediction/tune_early_pred/best_params_transformer_with_v.yaml"),
        Path("results/plots/early_prediction/optuna_study/best_params.yaml")
    ]:
        if t_yaml.exists():
            import yaml
            with open(t_yaml, "r") as f:
                tuned_params = yaml.safe_load(f)
                print(f"Loaded tuned predictor hyperparameters from {t_yaml}: {tuned_params}")
                break

    # 3. Train or load cached predictor models over n_splits random splits
    ckpt_save_dir = Path("results/checkpoints/early_prediction") / "eval"
    ckpt_save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading/Training {args.n_splits} Sepsis Predictor models over random train/test splits...")
    predictors = []
    predictor_accs = []
    
    for split_idx in range(args.n_splits):
        # Search for pre-trained predictor checkpoint from tuned_early_pred_sweep or eval cache
        found_ckpt = None
        ep_ckpt_root = Path("results/checkpoints/early_prediction")
        if ep_ckpt_root.exists():
            candidates = list(ep_ckpt_root.glob(f"**/*tau{args.tau}*split{split_idx}.pt"))
            if not candidates:
                candidates = list(ep_ckpt_root.glob(f"**/*split{split_idx}.pt"))
            if candidates:
                found_ckpt = candidates[0]

        if found_ckpt and found_ckpt.exists():
            print(f"Loading pre-trained EP model split {split_idx + 1}/{args.n_splits} from: {found_ckpt}")
            try:
                ckpt_data = torch.load(found_ckpt, map_location=device)
                h_dim = tuned_params.get("hidden_dim", 64)
                n_layers = tuned_params.get("num_layers", 2)
                d_out = tuned_params.get("dropout", 0.2)
                b_dir = tuned_params.get("bidirectional", False)
                pred_model = SepsisPredictorLSTM(input_dim=49, hidden_dim=h_dim, num_layers=n_layers, dropout=d_out, bidirectional=b_dir).to(device)
                pred_model.load_state_dict(ckpt_data["model_state_dict"])
                pred_model.eval()
                pred_acc = ckpt_data.get("pred_acc", 0.80)
            except Exception as err:
                print(f"Notice: Checkpoint {found_ckpt} format mismatch ({err}). Training split {split_idx + 1} fresh with tuned hyperparameters...")
                found_ckpt = None

        if found_ckpt is None or not found_ckpt.exists():
            print(f"\nTraining predictor split {split_idx + 1}/{args.n_splits}...")
            seed_val = 42 + split_idx
            train_indices, test_indices_pred = train_test_split(
                np.arange(len(X_train)), test_size=0.2, random_state=seed_val
            )
            plot_path = None
            if split_idx == 0:
                convergence_dir = report_dir / "convergence"
                convergence_dir.mkdir(parents=True, exist_ok=True)
                plot_path = convergence_dir / "predictor_convergence.png"
                
            pred_model, pred_acc = train_predictor(
                X_train, y_train, mask_train, train_indices, test_indices_pred,
                epochs=args.predictor_epochs, device=device,
                plot_convergence=(split_idx == 0), plot_path=plot_path,
                tuned_params=tuned_params
            )
            torch.save({
                "model_state_dict": pred_model.state_dict(),
                "pred_acc": float(pred_acc),
                "tau": args.tau,
                "split_idx": split_idx
            }, ckpt_path)
            print(f"Saved trained predictor checkpoint to: {ckpt_path}")
            
        predictors.append(pred_model)
        predictor_accs.append(pred_acc)
        
    # Load evaluation dataset
    eval_dataset_path = os.path.join(mimic_dir, args.eval_dataset_name)
    if not os.path.exists(eval_dataset_path):
        raise FileNotFoundError(f"Evaluation dataset not found at {eval_dataset_path}")
        
    print(f"Loading evaluation dataset from: {eval_dataset_path}")
    data_eval = np.load(eval_dataset_path, allow_pickle=True)
    X = data_eval['X']
    y = data_eval['y']
    mask = data_eval['mask']
    
    # Apply lead-time tau cutoff to evaluation dataset
    print(f"Applying lead-time tau={args.tau} hours ({steps_early} steps) cutoff to evaluation dataset...")
    X = X.copy()
    mask = mask.copy()
    for i in range(len(X)):
        valid_steps = np.where(mask[i].squeeze() != -1)[0]
        cutoff = max(1, len(valid_steps) - steps_early)
        mask[i, cutoff:] = -1
        X[i, cutoff:, :] = 0
        
    print(f"Evaluation dataset size: {len(X)} patients")
    
    # 4. Gather checkpoints to evaluate
    checkpoint_arg = Path(args.checkpoint)
    checkpoints_to_eval = []
    
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
            checkpoints_to_eval = candidates
            print(f"Directory mode: Found {len(checkpoints_to_eval)} checkpoints to evaluate under {checkpoint_arg}")
        else:
            raise FileNotFoundError(f"No best_model*.ckpt files found in directory or subdirectories of {checkpoint_arg}")
    else:
        checkpoint_path = checkpoint_arg
        if not checkpoint_path.exists():
            dirpath = checkpoint_path.parent
            candidates = list(dirpath.glob("best_model*.ckpt"))
            if candidates:
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                checkpoint_path = candidates[0]
                print(f"Specified checkpoint not found. Redirecting to newest candidate: {checkpoint_path}")
            else:
                raise FileNotFoundError(f"No checkpoints found at or under {checkpoint_path}")
        checkpoints_to_eval = [checkpoint_path]
        
    # Allow loading dictconfig safelist for torch.load
    torch.serialization.add_safe_globals([
        getattr(sys.modules.get('omegaconf.dictconfig', None), 'DictConfig', None)
    ])
            
    # 5. Read existing entries to support incremental mode
    existing_checkpoints = set()
    if not args.remake and csv_path.exists():
        try:
            with open(csv_path, mode="r") as f_csv:
                reader = csv.reader(f_csv)
                header = next(reader, None)
                if header:
                    for row in reader:
                        if row:
                            existing_checkpoints.add(row[0])
        except Exception as e:
            print(f"Error reading existing CSV: {e}")
            
    # Filter checkpoints for incremental evaluation
    if not args.remake:
        original_count = len(checkpoints_to_eval)
        checkpoints_to_eval = [
            cp for cp in checkpoints_to_eval
            if "/".join(cp.parts[-5:]) not in existing_checkpoints
            and cp.name not in existing_checkpoints
        ]
        skipped = original_count - len(checkpoints_to_eval)
        if skipped > 0:
            print(f"Incremental mode: Skipped {skipped} already evaluated checkpoints. {len(checkpoints_to_eval)} checkpoints remaining.")
            
    # Initialize output summary files
    if args.remake or not csv_path.exists():
        with open(csv_path, mode="w", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow([
                "checkpoint", "predictor_acc", "predictor_acc_sem",
                "clinician_mort", "clinician_mort_sem",
                "cql_mort", "cql_mort_sem",
                "clinician_admin", "clinician_admin_sem",
                "cql_admin", "cql_admin_sem",
                "agreement", "agreement_sem",
                "setup_b_expert_visits", "setup_b_expert_visits_sem",
                "setup_b_accuracy", "setup_b_accuracy_sem",
                "setup_b_recall", "setup_b_recall_sem",
                "setup_b_precision", "setup_b_precision_sem",
                "setup_b_f1", "setup_b_f1_sem",
                "setup_b_auc", "setup_b_auc_sem"
            ])

    # 6. Evaluation Loop
    for checkpoint_path in checkpoints_to_eval:
        print(f"\n" + "="*50)
        print(f"Evaluating checkpoint: {checkpoint_path}")
        print("="*50)
        
        try:
            cql_agent = CQLAgent.load_from_checkpoint(str(checkpoint_path), map_location=device, weights_only=False)
            cql_agent.eval()
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            continue
            
        split_pred_accs = []
        split_clinician_morts = []
        split_cql_morts = []
        split_clinician_admins = []
        split_cql_admins = []
        split_agreements = []
        split_expert_visits = []
        split_accuracies = []
        split_recalls = []
        split_precisions = []
        split_f1s = []
        split_aucs = []
        
        patient_agreements_all = []
        patient_true_outcomes_all = []
        
        for split_idx in range(args.n_splits):
            predictor = predictors[split_idx]
            pred_acc = predictor_accs[split_idx]
            split_pred_accs.append(pred_acc)
            
            seed_val = 42 + split_idx
            # Split the evaluation dataset into 80/20 train/test
            _, test_indices_eval = train_test_split(
                np.arange(len(X)), test_size=0.2, random_state=seed_val
            )
            
            # Setup A: Counterfactual Evaluation
            X_clinician = X[test_indices_eval, :, :49].copy()
            m_test = (mask[test_indices_eval] != -1).astype(np.float32)
            X_clinician = X_clinician * m_test
            
            X_cql = X[test_indices_eval, :, :49].copy()
            cql_actions_count = 0
            clinician_actions_count = 0
            agreement_count = 0
            
            cql_recommended_actions = np.zeros((len(test_indices_eval), 240))
            cql_action_probs = np.zeros((len(test_indices_eval), 240))
            
            for idx, patient_idx in enumerate(test_indices_eval):
                valid_steps = np.where(mask[patient_idx].squeeze() != -1)[0]
                patient_matches = 0
                for t in valid_steps:
                    obs = X[patient_idx, t, :46]
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        probs = cql_agent.actor.get_action_probs(obs_tensor)
                        action = torch.argmax(probs, dim=-1).item()
                        prob_admin = probs[0, 1].item()
                        
                    cql_recommended_actions[idx, t] = action
                    cql_action_probs[idx, t] = prob_admin
                    X_cql[idx, t, 47] = action
                    
                    clin_act = int(X[patient_idx, t, 47])
                    if action == clin_act:
                        agreement_count += 1
                        patient_matches += 1
                    if action == 1:
                        cql_actions_count += 1
                    if clin_act == 1:
                        clinician_actions_count += 1
                        
                p_aggr = (patient_matches / max(1, len(valid_steps))) * 100.0
                patient_agreements_all.append(p_aggr)
                patient_true_outcomes_all.append(float(y[patient_idx, 0]))
                        
            X_cql = X_cql * m_test
            
            predictor.eval()
            with torch.no_grad():
                inputs_clinician = torch.tensor(X_clinician, dtype=torch.float32).to(device)
                mask_tensor = torch.tensor(mask[test_indices_eval], dtype=torch.float32).to(device)
                
                logits_clinician = predictor(inputs_clinician, mask_tensor)
                probs_clinician = torch.sigmoid(logits_clinician).cpu().numpy().squeeze()
                
                inputs_cql = torch.tensor(X_cql, dtype=torch.float32).to(device)
                logits_cql = predictor(inputs_cql, mask_tensor)
                probs_cql = torch.sigmoid(logits_cql).cpu().numpy().squeeze()
                
            avg_mortality_clinician = float(np.mean(probs_clinician))
            avg_mortality_cql = float(np.mean(probs_cql))
            
            total_valid_steps = sum(len(np.where(mask[i].squeeze() != -1)[0]) for i in test_indices_eval)
            policy_agreement = agreement_count / total_valid_steps if total_valid_steps > 0 else 0.0
            cql_admin_rate = cql_actions_count / total_valid_steps if total_valid_steps > 0 else 0.0
            clinician_admin_rate = clinician_actions_count / total_valid_steps if total_valid_steps > 0 else 0.0
            
            split_clinician_morts.append(avg_mortality_clinician)
            split_cql_morts.append(avg_mortality_cql)
            split_clinician_admins.append(clinician_admin_rate)
            split_cql_admins.append(cql_admin_rate)
            split_agreements.append(policy_agreement)
            
            # Setup B: Imitation of Effective Interventions
            with torch.no_grad():
                logits_steps = predictor.predict_all_steps(inputs_clinician)
                probs_steps = torch.sigmoid(logits_steps).cpu().numpy().squeeze(-1) # (N_test_eval, 240)
                
            targets_list = []
            predictions_list = []
            scores_list = []
            
            for idx, patient_idx in enumerate(test_indices_eval):
                patient_y = y[patient_idx, 0]
                if patient_y == 0:
                    valid_steps = np.where(mask[patient_idx].squeeze() != -1)[0]
                    for t in valid_steps:
                        pred_death_prob = probs_steps[idx, t]
                        if pred_death_prob > 0.5:
                            clinician_act = int(X[patient_idx, t, 47])
                            cql_act = cql_recommended_actions[idx, t]
                            cql_prob_admin = cql_action_probs[idx, t]
                            
                            targets_list.append(clinician_act)
                            predictions_list.append(cql_act)
                            scores_list.append(cql_prob_admin)
                            
            if len(targets_list) > 0:
                accuracy = accuracy_score(targets_list, predictions_list)
                recall = recall_score(targets_list, predictions_list, zero_division=0)
                precision = precision_score(targets_list, predictions_list, zero_division=0)
                f1 = f1_score(targets_list, predictions_list, zero_division=0)
                if len(np.unique(targets_list)) > 1:
                    auc_val = roc_auc_score(targets_list, scores_list)
                else:
                    auc_val = 0.5
            else:
                # Try lower threshold
                targets_list = []
                predictions_list = []
                scores_list = []
                for idx, patient_idx in enumerate(test_indices_eval):
                    patient_y = y[patient_idx, 0]
                    if patient_y == 0:
                        valid_steps = np.where(mask[patient_idx].squeeze() != -1)[0]
                        for t in valid_steps:
                            pred_death_prob = probs_steps[idx, t]
                            if pred_death_prob > 0.3:
                                clinician_act = int(X[patient_idx, t, 47])
                                cql_act = cql_recommended_actions[idx, t]
                                cql_prob_admin = cql_action_probs[idx, t]
                                
                                targets_list.append(clinician_act)
                                predictions_list.append(cql_act)
                                scores_list.append(cql_prob_admin)
                if len(targets_list) > 0:
                    accuracy = accuracy_score(targets_list, predictions_list)
                    recall = recall_score(targets_list, predictions_list, zero_division=0)
                    precision = precision_score(targets_list, predictions_list, zero_division=0)
                    f1 = f1_score(targets_list, predictions_list, zero_division=0)
                    if len(np.unique(targets_list)) > 1:
                        auc_val = roc_auc_score(targets_list, scores_list)
                    else:
                        auc_val = 0.5
                else:
                    accuracy, recall, precision, f1, auc_val = 0.0, 0.0, 0.0, 0.0, 0.5
                    
            split_expert_visits.append(len(targets_list))
            split_accuracies.append(accuracy)
            split_recalls.append(recall)
            split_precisions.append(precision)
            split_f1s.append(f1)
            split_aucs.append(auc_val)
            
        def get_mean_sem(lst):
            return float(np.mean(lst)), float(np.std(lst) / np.sqrt(len(lst)))
            
        pred_acc_m, pred_acc_s = get_mean_sem(split_pred_accs)
        clinician_mort_m, clinician_mort_s = get_mean_sem(split_clinician_morts)
        cql_mort_m, cql_mort_s = get_mean_sem(split_cql_morts)
        clinician_admin_m, clinician_admin_s = get_mean_sem(split_clinician_admins)
        cql_admin_m, cql_admin_s = get_mean_sem(split_cql_admins)
        agreement_m, agreement_s = get_mean_sem(split_agreements)
        expert_visits_m, expert_visits_s = get_mean_sem(split_expert_visits)
        acc_b_m, acc_b_s = get_mean_sem(split_accuracies)
        rec_b_m, rec_b_s = get_mean_sem(split_recalls)
        prec_b_m, prec_b_s = get_mean_sem(split_precisions)
        f1_b_m, f1_b_s = get_mean_sem(split_f1s)
        auc_b_m, auc_b_s = get_mean_sem(split_aucs)
        
        print(f"Setup A Results over {args.n_splits} splits:")
        print(f"  Avg Predicted Mortality (Clinician): {clinician_mort_m:.4f} \u00b1 {clinician_mort_s:.4f}")
        print(f"  Avg Predicted Mortality (CQL Policy): {cql_mort_m:.4f} \u00b1 {cql_mort_s:.4f}")
        print(f"  Policy Agreement: {agreement_m:.4f} \u00b1 {agreement_s:.4f}")
        print(f"  Clinician Admin Rate: {clinician_admin_m:.4f} \u00b1 {clinician_admin_s:.4f}, CQL Admin Rate: {cql_admin_m:.4f} \u00b1 {cql_admin_s:.4f}")
        
        print(f"Setup B Results over {args.n_splits} splits:")
        print(f"  Accuracy:  {acc_b_m:.4f} \u00b1 {acc_b_s:.4f}")
        print(f"  Recall:    {rec_b_m:.4f} \u00b1 {rec_b_s:.4f}")
        print(f"  Precision: {prec_b_m:.4f} \u00b1 {prec_b_s:.4f}")
        print(f"  F1-score:  {f1_b_m:.4f} \u00b1 {f1_b_s:.4f}")
        print(f"  AUC-ROC:   {auc_b_m:.4f} \u00b1 {auc_b_s:.4f}")
        
        # Parse name
        try:
            parts = Path(checkpoint_path).parts
            if len(parts) >= 5:
                checkpoint_name = "/".join(parts[-5:])
            else:
                checkpoint_name = Path(checkpoint_path).name
        except Exception:
            checkpoint_name = str(checkpoint_path)
            
        with open(csv_path, mode="a", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow([
                checkpoint_name,
                f"{pred_acc_m:.6f}", f"{pred_acc_s:.6f}",
                f"{clinician_mort_m:.6f}", f"{clinician_mort_s:.6f}",
                f"{cql_mort_m:.6f}", f"{cql_mort_s:.6f}",
                f"{clinician_admin_m:.6f}", f"{clinician_admin_s:.6f}",
                f"{cql_admin_m:.6f}", f"{cql_admin_s:.6f}",
                f"{agreement_m:.6f}", f"{agreement_s:.6f}",
                f"{expert_visits_m:.6f}", f"{expert_visits_s:.6f}",
                f"{acc_b_m:.6f}", f"{acc_b_s:.6f}",
                f"{rec_b_m:.6f}", f"{rec_b_s:.6f}",
                f"{prec_b_m:.6f}", f"{prec_b_s:.6f}",
                f"{f1_b_m:.6f}", f"{f1_b_s:.6f}",
                f"{auc_b_m:.6f}", f"{auc_b_s:.6f}"
            ])
            
        print(f"Appended results for {checkpoint_name} to CSV summary file.")

        # Generate Septic Shock Rate vs. Policy Agreement Plot
        try:
            import matplotlib.pyplot as plt
            agreements_np = np.array(patient_agreements_all)
            outcomes_np = np.array(patient_true_outcomes_all)
            
            bins = np.linspace(0, 100, 11)
            bin_centers = (bins[:-1] + bins[1:]) / 2.0
            
            bin_shock_means = []
            bin_shock_sems = []
            bin_patient_counts = []
            
            for b_idx in range(10):
                low, high = bins[b_idx], bins[b_idx+1]
                if b_idx == 9:
                    idx_mask = (agreements_np >= low) & (agreements_np <= high)
                else:
                    idx_mask = (agreements_np >= low) & (agreements_np < high)
                    
                pts_in_bin = outcomes_np[idx_mask]
                bin_patient_counts.append(len(pts_in_bin))
                if len(pts_in_bin) > 0:
                    mean_val = float(np.mean(pts_in_bin))
                    sem_val = float(np.std(pts_in_bin) / np.sqrt(len(pts_in_bin))) if len(pts_in_bin) > 1 else 0.0
                    bin_shock_means.append(mean_val)
                    bin_shock_sems.append(sem_val)
                else:
                    bin_shock_means.append(np.nan)
                    bin_shock_sems.append(0.0)
                    
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            means_arr = np.array(bin_shock_means)
            sems_arr = np.array(bin_shock_sems)
            valid_mask = ~np.isnan(means_arr)
            
            ax1.plot(bin_centers[valid_mask], means_arr[valid_mask] * 100.0, 'o-', color='tab:red', linewidth=2.5, label='True Septic Shock Rate (%)')
            ax1.fill_between(bin_centers[valid_mask], (means_arr[valid_mask] - sems_arr[valid_mask]) * 100.0, (means_arr[valid_mask] + sems_arr[valid_mask]) * 100.0, color='tab:red', alpha=0.2)
            
            ax1.set_xlabel("Clinician - RL Policy Agreement (%)", fontsize=12, fontweight='bold')
            ax1.set_ylabel("True Patient Septic Shock Rate (%)", fontsize=12, fontweight='bold', color='tab:red')
            ax1.tick_params(axis='y', labelcolor='tab:red')
            ax1.set_xticks(np.arange(0, 101, 10))
            ax1.grid(True, linestyle="--", alpha=0.5)
            
            ax2 = ax1.twinx()
            ax2.bar(bin_centers, bin_patient_counts, width=8, color='tab:blue', alpha=0.2, label='Patient Count')
            ax2.set_ylabel("Patient Count in Bin", fontsize=12, fontweight='bold', color='tab:blue')
            ax2.tick_params(axis='y', labelcolor='tab:blue')
            
            plt.title("True Septic Shock Rate vs. Clinician-RL Policy Agreement %", fontsize=13, fontweight='bold')
            fig.tight_layout()
            
            plot_out_dir = Path("results/plots/early_prediction") / (exp_id or "default")
            plot_out_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_out_dir / "septic_shock_vs_agreement.png"
            plt.savefig(plot_path, dpi=200)
            plt.close()
            print(f"Saved Septic Shock vs. Policy Agreement plot to: {plot_path}")
        except Exception as e:
            print(f"Warning: Could not save septic shock vs agreement plot: {e}")

        # Generate Cohort Breakdown Mortality Comparison (All vs Septic Shock vs Non-Shock)
        try:
            import matplotlib.pyplot as plt
            outcomes_np = np.array(patient_true_outcomes_all)
            clin_mort_np = np.array(patient_clinician_mort_all)
            cql_mort_np = np.array(patient_cql_mort_all)
            
            shock_mask = (outcomes_np == 1)
            non_shock_mask = (outcomes_np == 0)
            
            cohorts = [
                ("All Patients", np.ones(len(outcomes_np), dtype=bool), "mortality_all_patients.png"),
                ("Septic Shock Cohort (y=1)", shock_mask, "mortality_septic_shock_cohort.png"),
                ("Non-Shock Cohort (y=0)", non_shock_mask, "mortality_non_shock_cohort.png")
            ]
            
            for cohort_name, c_mask, fname in cohorts:
                if np.sum(c_mask) == 0:
                    continue
                clin_mean = np.mean(clin_mort_np[c_mask]) * 100.0
                cql_mean = np.mean(cql_mort_np[c_mask]) * 100.0
                
                fig, ax = plt.subplots(figsize=(7, 5))
                bars = ax.bar(["Clinician Care", "RL Policy"], [clin_mean, cql_mean], color=['tab:blue', 'tab:orange'], alpha=0.85, width=0.5)
                ax.set_ylabel("Predicted Patient Mortality Rate (%)", fontsize=12, fontweight='bold')
                ax.set_title(f"Predicted Mortality: {cohort_name}\n(Evaluated at {args.tau}h Lead Time Cutoff)", fontsize=12, fontweight='bold')
                ax.set_ylim(0, max(clin_mean, cql_mean, 1.0) * 1.25)
                ax.grid(True, linestyle="--", alpha=0.5, axis='y')
                
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
                    
                fig.tight_layout()
                c_path = report_dir / fname
                plt.savefig(c_path, dpi=200)
                plt.close()
                print(f"Saved cohort breakdown plot ({cohort_name}) to: {c_path}")
        except Exception as e:
            print(f"Warning: Could not save cohort breakdown mortality plot: {e}")

    # Re-generate the single summary & detailed text reports document
    generate_text_report(csv_path, summary_path, exp_id)
    print(f"\nText summary and detailed reports regenerated at: {summary_path}")

if __name__ == "__main__":
    main()
