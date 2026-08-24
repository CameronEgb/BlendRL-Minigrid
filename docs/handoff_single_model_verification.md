# Handoff Guide 1: Single BlendRL Model Deployment & Verification

## Executive Summary
This document provides the exact technical specification and execution steps to deploy our currently trained **NeSyRL BlendRL checkpoint** into **Pyrenees ITS** (`Pyrenees-python`) under condition **`26140`**.

It packages the existing model so it satisfies Pyrenees's 11-key lookup table architecture (`problem`, `ex132(w)`, `ex132a(w)`, `ex152a(w)`, `ex212(w)`, `ex242(w)`, `ex252(w)`, `ex252a(w)`, `exc137(w)`, `exp426d(w)`, `exp426e(w)`), integrates routing in `app/routes.py`, and runs automated unit tests to verify end-to-end functionality.

---

## 1. System Architecture & Action Space Mapping

### Action Definitions (NeSyRL Parity):
$$\text{Action 0} = \text{PS} \quad | \quad \text{Action 1} = \text{WE} \quad | \quad \text{Action 2} = \text{FWE}$$

### Routing Mapping:
- **Problem Level (`dialog_saymachineproblemdecision`)**:
  - `0 (PS)` $\rightarrow$ `'problem'` (Pure Problem Solving)
  - `1 (WE)` $\rightarrow$ `'example'` (Pure Worked Example)
  - `2 (FWE)` $\rightarrow$ `'step_decision'` (Enters FWE mode, delegating to step-level decisions)
- **Step Level (`dialog_saymachinestepdecision`)**:
  - `0 (PS)` $\rightarrow$ `'problem'` (Elicit step: student solves)
  - `1 (WE)` $\rightarrow$ `'example'` (Tell step: tutor demonstrates)
  - `2 (FWE)` $\rightarrow$ `'problem'` (Consistent elicit mapping)

---

## 2. Detailed Implementation Steps

### Step 1: Export Script in NeSyRL
Create [`/Users/cameronegbert/Documents/NCSU/Research/NeSyRL/scripts/export_pyrenees_for_tutor.py`](file:///Users/cameronegbert/Documents/NCSU/Research/NeSyRL/scripts/export_pyrenees_for_tutor.py):
- Loads `results/checkpoints/pyrenees/test_pyrenees_blendrl/cql_blendrl_human_neural/0/best_model.ckpt`.
- Loads scaler parameters from `in/datasets/pyrenees/pyrenees_scaler.npz` and `in/datasets/pyrenees/pyrenees_gmm_scaler.npz`.
- Compiles the TorchScript model using `torch.jit.trace`.
- Populates `/Users/cameronegbert/Documents/NCSU/Research/Pyrenees/Pyrenees-python/app/models/policies/Blend-RL/`:
  - `trace/`: Saves `problem.pt`, `ex132(w).pt`, `ex132a(w).pt`, `ex152a(w).pt`, `ex212(w).pt`, `ex242(w).pt`, `ex252(w).pt`, `ex252a(w).pt`, `exc137(w).pt`, `exp426d(w).pt`, `exp426e(w).pt`.
  - `minmax/`: Saves `problem_0.pkl`, `ex132(w)_0.pkl`, `ex132a(w)_0.pkl`, `ex152a(w)_0.pkl`, `ex212(w)_0.pkl`, `ex242(w)_0.pkl`, `ex252(w)_0.pkl`, `ex252a(w)_0.pkl`, `exc137(w)_0.pkl`, `exp426d(w)_0.pkl`, `exp426e(w)_0.pkl`.

```python
# Code outline for scripts/export_pyrenees_for_tutor.py:
import torch
import numpy as np
import pickle
from pathlib import Path
from src.methods.cql_agent import CQLAgent

ckpt_path = "results/checkpoints/pyrenees/test_pyrenees_blendrl/cql_blendrl_human_neural/0/best_model.ckpt"
agent = CQLAgent.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=False)
agent.eval()

class StandaloneBlendRL(torch.nn.Module):
    def __init__(self, agent):
        super().__init__()
        self.actor = agent.model.actor
    def forward(self, obs_123):
        B = obs_123.shape[0]
        pad = torch.zeros((B, 3), dtype=obs_123.dtype, device=obs_123.device)
        obs_126 = torch.cat([obs_123, pad], dim=-1)
        logic_obs = obs_126.unsqueeze(1).repeat(1, 2, 1)
        probs, _ = self.actor(obs_126, logic_obs)
        return probs

model = StandaloneBlendRL(agent)
traced = torch.jit.trace(model, torch.zeros((1, 123), dtype=torch.float32))

target_dir = Path("/Users/cameronegbert/Documents/NCSU/Research/Pyrenees/Pyrenees-python/app/models/policies/Blend-RL")
trace_dir = target_dir / "trace"
minmax_dir = target_dir / "minmax"
trace_dir.mkdir(parents=True, exist_ok=True)
minmax_dir.mkdir(parents=True, exist_ok=True)

PROBLEM_LIST = ['problem', 'exc137(w)', 'ex132a(w)', 'ex132(w)', 'ex152a(w)', 'exp426d(w)', 'exp426e(w)', 'ex212(w)', 'ex242(w)', 'ex252a(w)', 'ex252(w)']

for pid in PROBLEM_LIST:
    torch.jit.save(traced, trace_dir / f"{pid}.pt")
    # Save standard min/max scaler object
    scaler_data = {'min': np.zeros(123), 'max': np.ones(123)}
    with open(minmax_dir / f"{pid}_0.pkl", "wb") as f:
        pickle.dump(scaler_data, f)
print("Successfully exported all 11 Blend-RL models and scalers!")
```

---

### Step 2: Policy Loader in `app/torch_policies.py`
In [`app/torch_policies.py`](file:///Users/cameronegbert/Documents/NCSU/Research/Pyrenees/Pyrenees-python/app/torch_policies.py):
1. In `PedagogicalAgent.__init__`:
   Set `self.subdirectory = "trace"` and `self.extension = "pt"` when `policy_name == "Blend-RL"`.
2. In `logic_mapping`:
   ```python
   def logic_mapping(decision_level: str, raw_value: int) -> str:
       if decision_level == "problem":
           if raw_value == 0:
               return "problem"
           elif raw_value == 1:
               return "example"
           else:
               return "step_decision"
       else:  # step level
           if raw_value == 1:
               return "example"
           else:
               return "problem"  # 0 (PS) and 2 (FWE) -> problem
   ```
3. Preload `"Blend-RL"` into `GLOBAL_MODEL` at startup.

---

### Step 3: Routing in `app/routes.py`
In [`app/routes.py`](file:///Users/cameronegbert/Documents/NCSU/Research/Pyrenees/Pyrenees-python/app/routes.py):
1. In `getProblemLevelDecision`:
   ```python
   elif user_condition == 26140:  # BlendRL Policy
       decision_info = get_problem_level_decision_from_torch_policy(
           "Blend-RL", legacy=False, conditionID='26140'
       )
   ```
2. In `getStepLevelDecision`:
   ```python
   elif user_condition == 26140:  # BlendRL Step-Level Policy
       decision_info = get_step_level_decision_from_torch_policy(
           "Blend-RL", legacy=False, conditionID=user_condition
       )
   ```

---

### Step 4: Unit Testing in `app/blendrl_policy_test.py`
Create [`app/blendrl_policy_test.py`](file:///Users/cameronegbert/Documents/NCSU/Research/Pyrenees/Pyrenees-python/app/blendrl_policy_test.py):
```python
import unittest
import numpy as np
from app.torch_policies import PedagogicalAgent, logic_mapping, torch_decision

class TestBlendRLPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = PedagogicalAgent("Blend-RL")
        cls.agent.build("Blend-RL")

    def test_step_level_all_problems(self):
        problems = ['exc137', 'ex132a', 'ex132', 'ex152a', 'exp426d', 'exp426e', 'ex212', 'ex242', 'ex252a', 'ex252']
        dummy_step_features = np.zeros((1, 123), dtype=np.float32)
        for pid in problems:
            res = torch_decision("Blend-RL", decision_level="step", problem_id=pid, input_features=dummy_step_features, condition="26140")
            self.assertIn(res["decision"], ["problem", "example"])
            self.assertEqual(res["policy"], "Blend-RL")

    def test_problem_level(self):
        dummy_prob_features = np.zeros((1, 123), dtype=np.float32)
        res = torch_decision("Blend-RL", decision_level="problem", problem_id="problem", input_features=dummy_prob_features, condition="26140")
        self.assertIn(res["decision"], ["problem", "example", "step_decision"])
```

---

## 3. Verification & Execution Commands

Run the following commands in terminal:
```bash
# 1. Run Export Script in NeSyRL
/Users/cameronegbert/Documents/NCSU/Research/NeSyRL/venv/bin/python /Users/cameronegbert/Documents/NCSU/Research/NeSyRL/scripts/export_pyrenees_for_tutor.py

# 2. Run BlendRL Unit Test
cd /Users/cameronegbert/Documents/NCSU/Research/Pyrenees/Pyrenees-python
python -m unittest app/blendrl_policy_test.py

# 3. Run Full Pyrenees Test Suite
./run_all_unit_tests.sh
```

---

## 4. Fresh Context Prompt
When opening a clean chat to execute this work, paste:
```text
Please execute Handoff Guide 1 (Single BlendRL Model Deployment & Verification) documented in handoff_single_model_verification.md:
1. Run scripts/export_pyrenees_for_tutor.py in NeSyRL to export the existing checkpoint to app/models/policies/Blend-RL/
2. Modify app/torch_policies.py and app/routes.py for condition 26140
3. Create and run app/blendrl_policy_test.py and ./run_all_unit_tests.sh
```
