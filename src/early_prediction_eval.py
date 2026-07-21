import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path

# Ensure project root and src are in PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.append(os.path.join(PROJECT_ROOT, "src"))

# Import CQLAgent from methods
from src.methods.cql_agent import CQLAgent

class SepsisPredictorLSTM(nn.Module):
    def __init__(self, input_dim=49, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, mask):
        # x: (batch_size, seq_len, input_dim)
        # mask: (batch_size, seq_len, 1)
        out, _ = self.lstm(x)
        
        # Determine lengths
        valid_mask = (mask.squeeze(-1) != -1).float()
        lengths = valid_mask.sum(dim=1).long() # (batch_size,)
        
        batch_size = x.size(0)
        idx = (lengths - 1).clamp(min=0)
        last_out = out[torch.arange(batch_size), idx] # (batch_size, hidden_dim)
        
        logits = self.fc(last_out)
        return logits

    def predict_all_steps(self, x):
        # x: (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x) # (batch_size, seq_len, hidden_dim)
        logits = self.fc(out) # (batch_size, seq_len, 1)
        return logits

class PatientDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, mask):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]

def train_predictor(X, y, mask, train_indices, test_indices, epochs=60, batch_size=32, device='cpu'):
    # States + action are features 0 to 48
    X_sa = X[:, :, :49].copy()
    
    # Zero out where mask == -1 to prevent NaN/weird values propagation
    m = (mask != -1).astype(np.float32)
    X_sa = X_sa * m
    
    train_dataset = PatientDataset(X_sa[train_indices], y[train_indices], mask[train_indices])
    test_dataset = PatientDataset(X_sa[test_indices], y[test_indices], mask[test_indices])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = SepsisPredictorLSTM(input_dim=49, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    best_acc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y, batch_mask in train_loader:
            batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
            optimizer.zero_grad()
            logits = model(batch_x, batch_mask)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
        # Eval on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_mask in test_loader:
                batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
                logits = model(batch_x, batch_mask)
                preds = (logits > 0).float()
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        acc = correct / total
        if acc > best_acc or best_state is None:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    print(f"Supervised Predictor trained. Best test accuracy: {best_acc:.4f}")
    
    model.load_state_dict(best_state)
    model.to(device)
    return model, best_acc

def generate_markdown_report(csv_path, summary_path, exp_id):
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
            
    with open(summary_path, mode="w") as f_md:
        f_md.write(f"# MIMIC Sepsis Early Prediction Summary Report ({exp_id})\n\n")
        f_md.write("This document compiles the summary table and detailed early prediction evaluation results for all evaluated checkpoints in this experiment.\n\n")
        
        # Write Summary Table
        f_md.write("## Summary Table\n\n")
        f_md.write("| Checkpoint | Predictor Acc | Clinician Mort | CQL Mort | Clinician Admin | CQL Admin | Agreement | Expert Visits | Accuracy (Setup B) | Recall | Precision | F1-Score | AUC-ROC |\n")
        f_md.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        
        for row in rows:
            # Format fields
            ckpt = row.get("checkpoint", "N/A")
            pred_acc = safe_float(row.get("predictor_acc", 0))
            clin_mort = safe_float(row.get("clinician_mort", 0))
            cql_mort = safe_float(row.get("cql_mort", 0))
            clin_admin = safe_float(row.get("clinician_admin", 0))
            cql_admin = safe_float(row.get("cql_admin", 0))
            agreement = safe_float(row.get("agreement", 0))
            expert_visits = row.get("setup_b_expert_visits")
            if expert_visits is None or expert_visits == "":
                expert_visits = "N/A"
            acc_b = safe_float(row.get("setup_b_accuracy", 0))
            rec_b = safe_float(row.get("setup_b_recall", 0))
            prec_b = safe_float(row.get("setup_b_precision", 0))
            f1_b = safe_float(row.get("setup_b_f1", 0))
            auc_b = safe_float(row.get("setup_b_auc", 0))
            
            f_md.write(f"| `{ckpt}` | {pred_acc:.2%} | {clin_mort:.2%} | {cql_mort:.2%} | {clin_admin:.2%} | {cql_admin:.2%} | {agreement:.2%} | {expert_visits} | {acc_b:.2%} | {rec_b:.2%} | {prec_b:.2%} | {f1_b:.2%} | {auc_b:.4f} |\n")
            
        f_md.write("\n---\n\n")
        f_md.write("# Detailed Trial Reports\n")
        
        for row in rows:
            ckpt = row.get("checkpoint", "N/A")
            ckpt_stem = os.path.basename(ckpt).replace(".ckpt", "")
            
            pred_acc = safe_float(row.get("predictor_acc", 0))
            clin_mort = safe_float(row.get("clinician_mort", 0))
            cql_mort = safe_float(row.get("cql_mort", 0))
            clin_admin = safe_float(row.get("clinician_admin", 0))
            cql_admin = safe_float(row.get("cql_admin", 0))
            agreement = safe_float(row.get("agreement", 0))
            expert_visits = row.get("setup_b_expert_visits")
            if expert_visits is None or expert_visits == "":
                expert_visits = "N/A"
            acc_b = safe_float(row.get("setup_b_accuracy", 0))
            rec_b = safe_float(row.get("setup_b_recall", 0))
            prec_b = safe_float(row.get("setup_b_precision", 0))
            f1_b = safe_float(row.get("setup_b_f1", 0))
            auc_b = safe_float(row.get("setup_b_auc", 0))
            
            f_md.write(f"""
## Checkpoint: `{ckpt}` (Trial `{ckpt_stem}`)

### Predictor Model Details
- **Architecture**: PyTorch LSTM Sepsis Predictor
- **Supervised Validation Accuracy**: **{pred_acc:.2%}**

### Setup A: Counterfactual Evaluation
| Metric | Clinician Actual | CQL Policy |
| :--- | :---: | :---: |
| **Average Predicted Mortality Rate** | **{clin_mort:.2%}** | **{cql_mort:.2%}** |
| **Antibiotics Administration Rate** | **{clin_admin:.2%}** | **{cql_admin:.2%}** |

- **Policy Agreement**: The CQL policy agreed with the clinician's decisions on **{agreement:.2%}** of all patient visits.

### Setup B: Imitation of Effective Interventions
- **Total Expert Visits Identified**: **{expert_visits}**

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **{acc_b:.2%}** |
| **Precision** | **{prec_b:.2%}** |
| **Recall (Sensitivity)** | **{rec_b:.2%}** |
| **F1-Score** | **{f1_b:.2%}** |
| **AUC-ROC** | **{auc_b:.4f}** |

---
""")

def main():
    import argparse
    import csv
    parser = argparse.ArgumentParser(description="MIMIC Sepsis Early Prediction Evaluation")
    parser.add_argument("--experiment", "-e", type=str, default=None, help="Experiment ID to evaluate (e.g., tune_mimic_blendrl_cql)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained CQL checkpoint or directory of checkpoints")
    parser.add_argument("--dataset-name", type=str, default=os.environ.get("MIMIC_DATASET_NAME", "mimic_lazy_12_clean_with_interventions_corrected.npz"), help="Predictor training dataset name")
    parser.add_argument("--eval-dataset-name", type=str, default=os.environ.get("MIMIC_EVAL_DATASET_NAME", "mimic_expert_demonstrations.npz"), help="Evaluation dataset name")
    parser.add_argument("--dataset-path", type=str, default=None, help="Direct path to the MIMIC dataset .npz file")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Custom directory containing the MIMIC dataset")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for early prediction report")
    parser.add_argument("--remake", action="store_true", help="Force recalculation and overwrite the CSV/MD summaries")
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
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../in/datasets/MIMIC 2")),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../in/datasets")),
                os.path.abspath(os.path.join(os.getcwd(), "in/datasets/MIMIC 2")),
                os.path.abspath(os.path.join(os.getcwd(), "in/datasets")),
                "/Users/cameronegbert/Documents/NCSU/Research/datasets/MIMIC 2",
                "/mnt/beegfs/cegbert/NeSyRL/in/datasets/MIMIC 2",
                "/mnt/beegfs/cegbert/NeSyRL/in/datasets",
                "/mnt/beegfs/cegbert/MIMIC 2"
            ]:
                if os.path.exists(candidate):
                    mimic_dir = candidate
                    break
            
    # Load predictor training dataset
    if args.dataset_path is not None:
        dataset_path = os.path.abspath(args.dataset_path)
    else:
        dataset_path = os.path.join(mimic_dir, args.dataset_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Predictor training dataset not found at {dataset_path}")
        
    print(f"Loading predictor training dataset from: {dataset_path}")
    data_train = np.load(dataset_path, allow_pickle=True)
    X_train = data_train['X']
    y_train = data_train['y']
    mask_train = data_train['mask']
    
    # 2. Split train/test (80/20) for predictor training
    train_indices, test_indices = train_test_split(np.arange(len(X_train)), test_size=0.2, random_state=42)
    print(f"Predictor Dataset split: Train={len(train_indices)}, Test={len(test_indices)}")
    
    # 3. Train predictor model (Trained only ONCE for all checkpoints!)
    predictor, predictor_acc = train_predictor(X_train, y_train, mask_train, train_indices, test_indices, device=device)
    
    # Load evaluation dataset
    eval_dataset_path = os.path.join(mimic_dir, args.eval_dataset_name)
    if not os.path.exists(eval_dataset_path):
        raise FileNotFoundError(f"Evaluation dataset not found at {eval_dataset_path}")
        
    print(f"Loading evaluation dataset from: {eval_dataset_path}")
    data_eval = np.load(eval_dataset_path, allow_pickle=True)
    X = data_eval['X']
    y = data_eval['y']
    mask = data_eval['mask']
    
    # Setup test_indices to cover the ENTIRE evaluation set
    test_indices = np.arange(len(X))
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
    
    # 5. Resolve Output Directories and Paths (Experiment-Specific)
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
    summary_path = report_dir / "early_prediction_summary.md"
    
    # Clean up old individual report files if present to prevent clutter
    for old_report in report_dir.glob("early_prediction_report*.md"):
        try:
            old_report.unlink()
            print(f"Removed legacy individual report file: {old_report}")
        except Exception as e:
            print(f"Error removing {old_report}: {e}")
            
    # 6. Read existing entries to support incremental mode
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
                "checkpoint", "predictor_acc", "clinician_mort", "cql_mort",
                "clinician_admin", "cql_admin", "agreement", "setup_b_expert_visits",
                "setup_b_accuracy", "setup_b_recall", "setup_b_precision",
                "setup_b_f1", "setup_b_auc"
            ])

    # 7. Evaluation Loop
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
            
        # Setup A: Counterfactual Evaluation
        print("Running Setup A: Counterfactual Evaluation...")
        X_clinician = X[test_indices, :, :49].copy()
        m_test = (mask[test_indices] != -1).astype(np.float32)
        X_clinician = X_clinician * m_test
        
        X_cql = X[test_indices, :, :49].copy()
        cql_actions_count = 0
        clinician_actions_count = 0
        agreement_count = 0
        
        cql_recommended_actions = np.zeros((len(test_indices), 240))
        cql_action_probs = np.zeros((len(test_indices), 240))
        
        for idx, patient_idx in enumerate(test_indices):
            valid_steps = np.where(mask[patient_idx].squeeze() != -1)[0]
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
                if action == 1:
                    cql_actions_count += 1
                if clin_act == 1:
                    clinician_actions_count += 1
                    
        X_cql = X_cql * m_test
        
        predictor.eval()
        with torch.no_grad():
            inputs_clinician = torch.tensor(X_clinician, dtype=torch.float32).to(device)
            mask_tensor = torch.tensor(mask[test_indices], dtype=torch.float32).to(device)
            
            logits_clinician = predictor(inputs_clinician, mask_tensor)
            probs_clinician = torch.sigmoid(logits_clinician).cpu().numpy().squeeze()
            
            inputs_cql = torch.tensor(X_cql, dtype=torch.float32).to(device)
            logits_cql = predictor(inputs_cql, mask_tensor)
            probs_cql = torch.sigmoid(logits_cql).cpu().numpy().squeeze()
            
        avg_mortality_clinician = float(np.mean(probs_clinician))
        avg_mortality_cql = float(np.mean(probs_cql))
        
        total_valid_steps = sum(len(np.where(mask[i].squeeze() != -1)[0]) for i in test_indices)
        policy_agreement = agreement_count / total_valid_steps if total_valid_steps > 0 else 0.0
        cql_admin_rate = cql_actions_count / total_valid_steps if total_valid_steps > 0 else 0.0
        clinician_admin_rate = clinician_actions_count / total_valid_steps if total_valid_steps > 0 else 0.0
        
        print(f"Setup A Results:")
        print(f"  Avg Predicted Mortality (Clinician): {avg_mortality_clinician:.4f}")
        print(f"  Avg Predicted Mortality (CQL Policy): {avg_mortality_cql:.4f}")
        print(f"  Policy Agreement: {policy_agreement:.4f}")
        print(f"  Clinician Admin Rate: {clinician_admin_rate:.4f}, CQL Admin Rate: {cql_admin_rate:.4f}")
        
        # Setup B: Imitation of Effective Interventions
        print("Running Setup B: Imitation of Effective Interventions...")
        with torch.no_grad():
            logits_steps = predictor.predict_all_steps(inputs_clinician)
            probs_steps = torch.sigmoid(logits_steps).cpu().numpy().squeeze(-1) # (N_test, 240)
            
        targets_list = []
        predictions_list = []
        scores_list = []
        
        for idx, patient_idx in enumerate(test_indices):
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
                        
        print(f"Setup B: Found {len(targets_list)} expert demonstration visits.")
        
        if len(targets_list) > 0:
            accuracy = accuracy_score(targets_list, predictions_list)
            recall = recall_score(targets_list, predictions_list, zero_division=0)
            precision = precision_score(targets_list, predictions_list, zero_division=0)
            f1 = f1_score(targets_list, predictions_list, zero_division=0)
            if len(np.unique(targets_list)) > 1:
                auc = roc_auc_score(targets_list, scores_list)
            else:
                auc = 0.5
        else:
            print("Warning: No visits met the predicted-to-crash threshold (> 0.5) for survivors. Trying lower threshold (0.3).")
            for idx, patient_idx in enumerate(test_indices):
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
            print(f"Setup B (threshold > 0.3): Found {len(targets_list)} expert demonstration visits.")
            if len(targets_list) > 0:
                accuracy = accuracy_score(targets_list, predictions_list)
                recall = recall_score(targets_list, predictions_list, zero_division=0)
                precision = precision_score(targets_list, predictions_list, zero_division=0)
                f1 = f1_score(targets_list, predictions_list, zero_division=0)
                if len(np.unique(targets_list)) > 1:
                    auc = roc_auc_score(targets_list, scores_list)
                else:
                    auc = 0.5
            else:
                accuracy, recall, precision, f1, auc = 0.0, 0.0, 0.0, 0.0, 0.5
                
        print(f"Setup B Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Recall:   {recall:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")
        
        # Parse name
        try:
            parts = Path(checkpoint_path).parts
            if len(parts) >= 5:
                checkpoint_name = "/".join(parts[-5:])
            else:
                checkpoint_name = Path(checkpoint_path).name
        except Exception:
            checkpoint_name = str(checkpoint_path)
            
        # Append to CSV summary file
        with open(csv_path, mode="a", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow([
                checkpoint_name, f"{predictor_acc:.6f}", f"{avg_mortality_clinician:.6f}", f"{avg_mortality_cql:.6f}",
                f"{clinician_admin_rate:.6f}", f"{cql_admin_rate:.6f}", f"{policy_agreement:.6f}",
                str(len(targets_list)),
                f"{accuracy:.6f}", f"{recall:.6f}", f"{precision:.6f}", f"{f1:.6f}", f"{auc:.6f}"
            ])
            
        print(f"Appended results for {checkpoint_name} to CSV summary file.")

    # Re-generate the single summary & detailed reports document
    generate_markdown_report(csv_path, summary_path, exp_id)
    print(f"\nMarkdown summary and detailed reports regenerated at: {summary_path}")

if __name__ == "__main__":
    main()
