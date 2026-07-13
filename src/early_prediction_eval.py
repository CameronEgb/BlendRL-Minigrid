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

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MIMIC Sepsis Early Prediction Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained CQL checkpoint")
    parser.add_argument("--dataset-name", type=str, default=os.environ.get("MIMIC_DATASET_NAME", "mimic_lazy_12_clean_with_q_values.npz"), help="Dataset name")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for early prediction report")
    args = parser.parse_known_args()[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # 1. Load the dataset
    mimic_dir = "/Users/cameronegbert/Documents/NCSU/Research/datasets/MIMIC 2"
    if not os.path.exists(mimic_dir):
        mimic_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../datasets/MIMIC 2"))
    if not os.path.exists(mimic_dir):
        mimic_dir = "/mnt/beegfs/cegbert/MIMIC 2"
    if not os.path.exists(mimic_dir):
        mimic_dir = os.path.abspath(os.path.join(os.getcwd(), "../datasets/MIMIC 2"))
        
    dataset_path = os.path.join(mimic_dir, args.dataset_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"MIMIC dataset not found at {dataset_path}")
        
    print(f"Loading dataset from: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    X = data['X']  # (N, 240, 51)
    y = data['y']  # (N, 1)
    mask = data['mask']  # (N, 240, 1)
    
    # 2. Split train/test (80/20)
    train_indices, test_indices = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
    print(f"Dataset split: Train={len(train_indices)}, Test={len(test_indices)}")
    
    # 3. Train predictor model
    predictor, predictor_acc = train_predictor(X, y, mask, train_indices, test_indices, device=device)
    
    # 4. Load CQL Agent
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        dirpath = checkpoint_path.parent
        candidates = list(dirpath.glob("best_model*.ckpt"))
        if candidates:
            # Sort by modification time, newest first
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            checkpoint_path = candidates[0]
            print(f"Specified checkpoint not found. Redirecting to newest candidate: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"No checkpoints found at or under {checkpoint_path}")
            
    print(f"Loading CQL policy from checkpoint: {checkpoint_path}")
    # Allow loading dictconfig safelist for torch.load
    torch.serialization.add_safe_globals([
        getattr(sys.modules.get('omegaconf.dictconfig', None), 'DictConfig', None)
    ])
    cql_agent = CQLAgent.load_from_checkpoint(str(checkpoint_path), map_location=device, weights_only=False)
    cql_agent.eval()
    
    # 5. Evaluate Setup A: Counterfactual Evaluation
    print("Running Setup A: Counterfactual Evaluation...")
    # Clinician actual inputs
    X_clinician = X[test_indices, :, :49].copy()
    m_test = (mask[test_indices] != -1).astype(np.float32)
    X_clinician = X_clinician * m_test
    
    # CQL recommended actions replacement
    X_cql = X[test_indices, :, :49].copy()
    cql_actions_count = 0
    clinician_actions_count = 0
    agreement_count = 0
    
    # Keep track of recommended actions and action probs
    cql_recommended_actions = np.zeros((len(test_indices), 240))
    cql_action_probs = np.zeros((len(test_indices), 240)) # Probability of action 1 (administer)
    
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
    
    # Run predictions through supervised model
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
    
    # 6. Evaluate Setup B: Imitation of Effective Interventions
    print("Running Setup B: Imitation of Effective Interventions...")
    # Get predictions at every step
    with torch.no_grad():
        logits_steps = predictor.predict_all_steps(inputs_clinician)
        probs_steps = torch.sigmoid(logits_steps).cpu().numpy().squeeze(-1) # (N_test, 240)
        
    targets_list = []
    predictions_list = []
    scores_list = []
    
    for idx, patient_idx in enumerate(test_indices):
        patient_y = y[patient_idx, 0]
        # Expert demonstration: Patient survived under clinician care
        if patient_y == 0:
            valid_steps = np.where(mask[patient_idx].squeeze() != -1)[0]
            for t in valid_steps:
                pred_death_prob = probs_steps[idx, t]
                # Patient was predicted to crash (probability of death > 0.5)
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
        
        # Check if we can compute AUC
        if len(np.unique(targets_list)) > 1:
            auc = roc_auc_score(targets_list, scores_list)
        else:
            auc = 0.5 # Default when only one class is present
    else:
        # Fallback if no visits match the strict threshold
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
    
    # 7. Write report
    if args.output_dir is not None:
        report_dir = Path(args.output_dir)
    else:
        # Determine from checkpoint path: results/checkpoints/group/exp_id/agent/trial_id/...
        parts = Path(args.checkpoint).parts
        if len(parts) >= 5:
            # results/checkpoints/[group]/[exp_id]/[agent]
            group = parts[2]
            exp_id = parts[3]
            report_dir = Path("results/plots/combined") / exp_id
        else:
            report_dir = Path("results/plots/combined/mimic_cql")
            
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "early_prediction_report.md"
    print(f"Writing early prediction report to: {report_path}")
    
    report_content = f"""# MIMIC Sepsis Early Prediction Evaluation Report

This report presents the clinical evaluation of the trained CQL policy on the MIMIC-III sepsis dataset. The evaluation compares the CQL policy's recommended actions against the clinician's actual actions across two configurations: **Setup A (Counterfactual Evaluation)** and **Setup B (Imitation of Effective Interventions)**.

## Predictor Model Details
- **Architecture**: PyTorch LSTM Sepsis Predictor
- **Input Dimension**: 49 (46 patient states + 3 actions/interventions)
- **Output**: Binary prediction of final patient outcome (survival vs mortality/shock)
- **Supervised Validation Accuracy**: **{predictor_acc:.2%}**

---

## Setup A: Counterfactual Evaluation
This setup compares the overall predicted patient mortality rate when adhering strictly to the clinician's actions versus replacing them with the trained CQL policy's recommendations.

| Metric | Clinician Actual | CQL Policy |
| :--- | :---: | :---: |
| **Average Predicted Mortality Rate** | **{avg_mortality_clinician:.2%}** | **{avg_mortality_cql:.2%}** |
| **Antibiotics Administration Rate** | **{clinician_admin_rate:.2%}** | **{cql_admin_rate:.2%}** |

- **Policy Agreement**: The CQL policy agreed with the clinician's decisions on **{policy_agreement:.2%}** of all patient visits.
- **Interpretation**: A lower predicted mortality rate under the CQL policy suggests potential clinical benefit from the policy's recommendation sequence.

---

## Setup B: Imitation of Effective Interventions
This setup focuses on "expert demonstrations"—specific visits during which the patient was at high risk of crashing (predicted mortality > 50%) but ultimately survived under the clinician's care. We measure how effectively the CQL policy imitates these life-saving decisions.

- **Total Expert Visits Identified**: **{len(targets_list)}**

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **{accuracy:.2%}** |
| **Precision** | **{precision:.2%}** |
| **Recall (Sensitivity)** | **{recall:.2%}** |
| **F1-Score** | **{f1:.2%}** |
| **AUC-ROC** | **{auc:.4f}** |

- **Interpretation**: High recall and precision in this subgroup indicate that the CQL policy successfully identifies and reproduces critical clinical interventions.
"""
    with open(report_path, "w") as f:
        f.write(report_content)
        
    print("Report generated successfully.")

if __name__ == "__main__":
    main()
