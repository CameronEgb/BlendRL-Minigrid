#!/usr/bin/env python3
"""
export_pyrenees_per_problem.py — Multi-Model ONNX and Scaler Exporter for Pyrenees ITS.

Loads trained checkpoints for all 11 problem types from NeSyRL,
compiles standalone ONNX graphs with matching action dimensions (3 for problem, 2 for step),
and saves drop-in ONNX models & scalers to Pyrenees-python/app/models/policies/Blend-RL/.

Usage:
  python scripts/export_pyrenees_per_problem.py [--ckpt-root CKPT_ROOT] [--verify]
"""

import os
import sys
import pickle
import argparse
import subprocess
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.methods.cql_agent import CQLAgent

ALL_PROBLEMS = [
    "problem",
    "ex132(w)",
    "ex132a(w)",
    "ex152a(w)",
    "ex212(w)",
    "ex242(w)",
    "ex252(w)",
    "ex252a(w)",
    "exc137(w)",
    "exp426d(w)",
    "exp426e(w)",
]


class StandaloneProblemPolicy(torch.nn.Module):
    """Standalone wrapper for Problem-Level Policy (130 in -> 3 actions out)."""
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
        obs_130 = obs[:, :130]
        B = obs_130.shape[0]
        pad = torch.zeros((B, 3), dtype=obs_130.dtype, device=obs_130.device)
        obs_133 = torch.cat([obs_130, pad], dim=-1)

        if self.is_modular:
            logic_obs = obs_133.unsqueeze(1).repeat(1, 2, 1)
            probs, _ = self.actor(obs_133, logic_obs)
            return probs
        elif hasattr(self, "q_network"):
            q = self.q_network(obs_133)
            return torch.softmax(q, dim=-1)
        else:
            q = self.model.get_q_values(obs_133)
            return torch.softmax(q, dim=-1)


class StandaloneStepPolicy(torch.nn.Module):
    """Standalone wrapper for Step-Level Policy (123 in -> 2 actions out)."""
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


def find_checkpoint(problem_id: str, ckpt_root: Path = None) -> Path:
    clean_id = problem_id.replace("/", "_").replace("(", "").replace(")", "")
    alt_ids = [problem_id, clean_id, problem_id.replace("(w)", "_w")]

    search_roots = []
    if ckpt_root:
        search_roots.append(ckpt_root)
    search_roots.extend([
        PROJECT_ROOT / "results" / "checkpoints" / "pyrenees" / "tune_pyrenees_3way",
        PROJECT_ROOT / "results" / "checkpoints" / "pyrenees" / "tune_pyrenees_blendrl_resnet",
        PROJECT_ROOT / "results" / "checkpoints" / "pyrenees" / "pyrenees_scaled",
        PROJECT_ROOT / "results" / "checkpoints" / "pyrenees" / "per_problem",
        PROJECT_ROOT / "results" / "checkpoints" / "pyrenees",
    ])

    for root in search_roots:
        if not root.exists():
            continue
        for aid in alt_ids:
            # 1. Direct filename match (e.g. cql_blendrl_human_neural_ex132_w.ckpt or ex132(w).ckpt)
            for p in root.rglob("*.ckpt"):
                stem = p.stem
                if stem.endswith(f"_{aid}") or stem == aid or f"_{aid}" in stem:
                    return p
            # 2. Directory match (e.g. cql_blendrl_human_neural_ex132_w/best_model.ckpt)
            for p in root.rglob("best_model*.ckpt"):
                parent_name = p.parent.parent.name if p.parent.name in ("0", "1", "2", "3", "4", "5") else p.parent.name
                if parent_name.endswith(f"_{aid}") or parent_name == aid or f"/{aid}/" in str(p):
                    return p

    return None


def export_all_models(output_dir: Path = None, ckpt_root: Path = None):
    if output_dir is None:
        output_dir = Path("/Users/cameronegbert/Documents/NCSU/Research/Pyrenees/Pyrenees-python/app/models/policies/Blend-RL")

    onnx_dir = output_dir / "onnx"
    minmax_dir = output_dir / "minmax"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    minmax_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("      EXPORTING 11 SPECIALIZED BLEND-RL ONNX MODELS & SCALERS")
    print("=" * 75)
    print(f"  Target Policy Directory: {output_dir}")

    exported_count = 0

    for problem_id in ALL_PROBLEMS:
        is_problem_level = (problem_id == "problem")
        n_in = 130 if is_problem_level else 123
        n_out = 3 if is_problem_level else 2

        # 1. Locate scaler
        scaler_path = PROJECT_ROOT / "in" / "datasets" / "pyrenees" / "per_problem" / problem_id / "scaler.npz"
        if not scaler_path.exists():
            scaler_path = PROJECT_ROOT / "in" / "datasets" / "pyrenees" / "pyrenees_scaler.npz"

        if not scaler_path.exists():
            print(f"  ❌ Scaler not found for {problem_id}. Run preprocess_pyrenees_per_problem.py first.")
            continue

        scaler_data = np.load(scaler_path)
        mean = scaler_data["mean"].astype(np.float64)[:n_in]
        std = scaler_data["std"].astype(np.float64)[:n_in]
        std[std == 0.0] = 1.0

        # Build MinMaxScaler
        scaler = MinMaxScaler()
        scaler.data_min_ = mean
        scaler.data_max_ = mean + std

        # Save Scaler
        out_scaler_file = minmax_dir / f"{problem_id}.pkl"
        with open(out_scaler_file, "wb") as f:
            pickle.dump(scaler, f)

        # 2. Locate Checkpoint
        ckpt_file = find_checkpoint(problem_id, ckpt_root)
        if not ckpt_file:
            print(f"  ⚠️ No checkpoint found for {problem_id}. Skipping ONNX export for this problem.")
            continue

        print(f"  [{problem_id}] Loading checkpoint: {ckpt_file.name} (In: {n_in}, Out: {n_out})")
        
        # Load agent
        orig_prob = os.environ.get("PYRENEES_PROBLEM_TYPE", None)
        os.environ["PYRENEES_PROBLEM_TYPE"] = problem_id
        try:
            agent = CQLAgent.load_from_checkpoint(str(ckpt_file), map_location="cpu", weights_only=False)
            agent.eval()

            if is_problem_level:
                policy_wrapper = StandaloneProblemPolicy(agent)
            else:
                policy_wrapper = StandaloneStepPolicy(agent)
            policy_wrapper.eval()

            # Export ONNX
            out_onnx_file = onnx_dir / f"{problem_id}.onnx"
            dummy_input = torch.zeros((1, n_in), dtype=torch.float32)

            torch.onnx.export(
                policy_wrapper,
                dummy_input,
                str(out_onnx_file),
                input_names=["input_0"],
                output_names=["output_0"],
                dynamic_axes={"input_0": {0: "batch", 1: "features"}, "output_0": {0: "batch"}},
                opset_version=14,
                dynamo=False,
            )

            print(f"  ✅ Exported {problem_id:12s} -> ONNX: {out_onnx_file.name}, Scaler: {out_scaler_file.name}")
            exported_count += 1
        finally:
            if orig_prob is not None:
                os.environ["PYRENEES_PROBLEM_TYPE"] = orig_prob

    print("=" * 75)
    print(f"  Export Complete: {exported_count} / {len(ALL_PROBLEMS)} models exported.")
    print("=" * 75)


def run_verification_tests():
    tutor_dir = Path("/Users/cameronegbert/Documents/NCSU/Research/Pyrenees/Pyrenees-python")
    if not tutor_dir.exists():
        print(f"Tutor directory {tutor_dir} not found; skipping verification.")
        return

    test_script = tutor_dir / "app" / "blendrl_policy_test.py"
    if not test_script.exists():
        print(f"Test script {test_script} not found; skipping verification.")
        return

    print("\n" + "=" * 75)
    print("      RUNNING PYRENEES ITS VERIFICATION UNIT TESTS")
    print("=" * 75)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{tutor_dir}:{tutor_dir}/app:{PROJECT_ROOT}:{PROJECT_ROOT}/src:" + env.get("PYTHONPATH", "")
    
    python_bin = sys.executable
    if (PROJECT_ROOT / "venv" / "bin" / "python").exists():
        python_bin = str(PROJECT_ROOT / "venv" / "bin" / "python")

    cmd = [python_bin, "-m", "unittest", "app/blendrl_policy_test.py"]
    res = subprocess.run(cmd, cwd=str(tutor_dir), env=env)
    if res.returncode == 0:
        print("\n✅ ALL PYRENEES BLENDRL UNIT TESTS PASSED!")
    else:
        print(f"\n❌ Unit tests returned code {res.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Export all 11 Pyrenees Blend-RL models to ITS.")
    parser.add_argument("--output-dir", type=str, default=None, help="Target policy directory in Pyrenees-python.")
    parser.add_argument("--ckpt-root", type=str, default=None, help="Root directory containing trained checkpoints.")
    parser.add_argument("--verify", action="store_true", default=True, help="Run unittest verification in Pyrenees-python.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None
    ckpt_root = Path(args.ckpt_root) if args.ckpt_root else None

    export_all_models(output_dir=out_dir, ckpt_root=ckpt_root)

    if args.verify:
        run_verification_tests()


if __name__ == "__main__":
    main()
