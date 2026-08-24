import os
import sys
import pickle
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

# Ensure project root and src are in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.methods.cql_agent import CQLAgent


class StandalonePyreneesPolicy(torch.nn.Module):
    """
    Universal inference wrapper for Pyrenees policies (Neural, BlendRL, CEW).
    Accepts raw observations (123 or 130 features), slices the first 123,
    pads the 3 one-hot alternation slots (to 126 dims), and computes
    action probabilities.
    """
    def __init__(self, agent):
        super().__init__()
        self.is_modular = getattr(agent, "is_modular", False)
        if self.is_modular:
            self.actor = agent.model.actor
        elif hasattr(agent, "q_network"):
            self.q_network = agent.q_network
        else:
            self.model = agent.model

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        obs_123 = obs[:, :123]
        B = obs_123.shape[0]
        pad = torch.zeros((B, 3), dtype=obs_123.dtype, device=obs_123.device)
        obs_126 = torch.cat([obs_123, pad], dim=-1)

        if self.is_modular:
            logic_obs = obs_126.unsqueeze(1).repeat(1, 2, 1)
            probs, _ = self.actor(obs_126, logic_obs)
            return probs
        elif hasattr(self, "q_network"):
            q = self.q_network(obs_126)
            return torch.softmax(q, dim=-1)
        else:
            q = self.model.get_q_values(obs_126)
            return torch.softmax(q, dim=-1)


def export_models_and_scalers(ckpt_path=None, output_dir=None):
    if ckpt_path is None:
        if len(sys.argv) > 1:
            ckpt_path = Path(sys.argv[1])
        else:
            # Fall back to best discovered checkpoint
            candidates = [
                PROJECT_ROOT / "results/checkpoints/pyrenees/pyrenees_best/blendrl_human_dueling_resnet/best_model.ckpt",
                PROJECT_ROOT / "results/checkpoints/pyrenees/pyrenees_best/cql_dueling_resnet/best_model.ckpt",
                PROJECT_ROOT / "results/checkpoints/pyrenees/tune_pyrenees_scaled/cql_blendrl_human_dueling_resnet/0/best_model.ckpt",
                PROJECT_ROOT / "results/checkpoints/pyrenees/test_pyrenees_blendrl/cql_blendrl_human_neural/0/best_model.ckpt",
            ]
            for c in candidates:
                if c.exists():
                    ckpt_path = c
                    break
    ckpt_path = Path(ckpt_path) if ckpt_path else None
    scaler_npz_path = PROJECT_ROOT / "in/datasets/pyrenees/pyrenees_scaler.npz"

    if not ckpt_path or not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    if not scaler_npz_path.exists():
        raise FileNotFoundError(f"Scaler npz not found at: {scaler_npz_path}")

    print(f"Loading checkpoint from: {ckpt_path}")
    agent = CQLAgent.load_from_checkpoint(str(ckpt_path), map_location="cpu", weights_only=False)
    agent.eval()

    model = StandalonePyreneesPolicy(agent)
    model.eval()

    # Load scaler parameters
    print(f"Loading scaler parameters from: {scaler_npz_path}")
    scaler_npz = np.load(scaler_npz_path)
    mean_123 = scaler_npz["mean"].astype(np.float64)
    std_123 = scaler_npz["std"].astype(np.float64)
    std_123[std_123 == 0.0] = 1.0

    target_policy_dir = Path("/Users/cameronegbert/Documents/NCSU/Research/Pyrenees/Pyrenees-python/app/models/policies/Blend-RL")
    onnx_dir = target_policy_dir / "onnx"
    minmax_dir = target_policy_dir / "minmax"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    minmax_dir.mkdir(parents=True, exist_ok=True)

    problem_list = [
        "problem",
        "exc137(w)",
        "ex132a(w)",
        "ex132(w)",
        "ex152a(w)",
        "exp426d(w)",
        "exp426e(w)",
        "ex212(w)",
        "ex242(w)",
        "ex252a(w)",
        "ex252(w)",
    ]

    # Create temporary template ONNX
    template_onnx_path = onnx_dir / "_template.onnx"
    dummy_input = torch.zeros((1, 123), dtype=torch.float32)
    print("Exporting ONNX template...")
    torch.onnx.export(
        model,
        dummy_input,
        str(template_onnx_path),
        input_names=["input_0"],
        output_names=["output_0"],
        dynamic_axes={"input_0": {0: "batch", 1: "features"}, "output_0": {0: "batch"}},
        opset_version=14,
        dynamo=False,
    )

    with open(template_onnx_path, "rb") as f:
        onnx_bytes = f.read()

    # Clean up template
    template_onnx_path.unlink()

    # Build scalers
    step_scaler = MinMaxScaler()
    step_scaler.data_min_ = mean_123
    step_scaler.data_max_ = mean_123 + std_123

    # For problem level (130 features), pad extra 7 features
    mean_130 = np.pad(mean_123, (0, 7), "constant", constant_values=0.0)
    std_130 = np.pad(std_123, (0, 7), "constant", constant_values=1.0)
    problem_scaler = MinMaxScaler()
    problem_scaler.data_min_ = mean_130
    problem_scaler.data_max_ = mean_130 + std_130

    for pid in problem_list:
        # Save ONNX
        onnx_path = onnx_dir / f"{pid}.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_bytes)

        # Save Scaler
        scaler_path = minmax_dir / f"{pid}.pkl"
        scaler_to_save = problem_scaler if pid == "problem" else step_scaler
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler_to_save, f)

        print(f"Exported {pid}: ONNX -> {onnx_path.name}, Scaler -> {scaler_path.name}")

    print(f"\nSuccessfully exported all {len(problem_list)} Blend-RL ONNX models and scalers!")


if __name__ == "__main__":
    export_models_and_scalers()
