#!/usr/bin/env python3
"""
train_pyrenees_all_problems.py — Multi-Problem Parallel Training Orchestrator for Pyrenees ITS.

Trains 11 specialized models (1 problem-level model + 10 exercise step-level models):
  1. problem       (130 features, 3 actions: 0=PS, 1=WE, 2=FWE, rules=default)
  2. ex132(w)      (123 features, 2 actions: 0=Elicit, 1=Tell, rules=step)
  3. ex132a(w)     (123 features, 2 actions: 0=Elicit, 1=Tell, rules=step)
  4. ex152a(w)     (123 features, 2 actions: 0=Elicit, 1=Tell, rules=step)
  5. ex212(w)      (123 features, 2 actions: 0=Elicit, 1=Tell, rules=step)
  6. ex242(w)      (123 features, 2 actions: 0=Elicit, 1=Tell, rules=step)
  7. ex252(w)      (123 features, 2 actions: 0=Elicit, 1=Tell, rules=step)
  8. ex252a(w)     (123 features, 2 actions: 0=Elicit, 1=Tell, rules=step)
  9. exc137(w)     (123 features, 2 actions: 0=Elicit, 1=Tell, rules=step)
  10. exp426d(w)   (123 features, 2 actions: 0=Elicit, 1=Tell, rules=step)
  11. exp426e(w)   (123 features, 2 actions: 0=Elicit, 1=Tell, rules=step)

Supports parallel execution across multiple GPU workers (e.g. on RTX 4060 Ti 16GB).
Each checkpoint is named and structured by problem type and method.

Usage:
  python scripts/train_pyrenees_all_problems.py --method cql/blendrl_human_dueling_resnet --parallel --max-workers 11
  python scripts/train_pyrenees_all_problems.py --method cql/dueling_resnet --gpus 0,1
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

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


def train_single_problem(
    problem_id: str,
    method: str,
    exp_id: str,
    epochs: int,
    batch_size: int,
    lr: float,
    cql_alpha: float,
    gpu_id: str,
    python_bin: str,
):
    is_problem_level = (problem_id == "problem")
    ruleset = "default" if is_problem_level else "step"
    method_clean = method.replace("/", "_")

    dataset_path = str(PROJECT_ROOT / "in" / "datasets" / "pyrenees" / "per_problem" / problem_id / "cql")
    clean_npz_path = str(PROJECT_ROOT / "in" / "datasets" / "pyrenees" / "per_problem" / problem_id / "clean.npz")
    gmm_scaler_path = str(PROJECT_ROOT / "in" / "datasets" / "pyrenees" / "per_problem" / problem_id / "gmm_scaler.npz")

    # Destination directory for this specific problem model
    ckpt_dir = PROJECT_ROOT / "results" / "checkpoints" / "pyrenees" / exp_id / problem_id / method_clean
    log_dir = PROJECT_ROOT / "results" / "logs" / "pyrenees" / exp_id / problem_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{PROJECT_ROOT}/src:{PROJECT_ROOT}/src/fyd_repo/src:" + env.get("PYTHONPATH", "")
    env["PYRENEES_PROBLEM_TYPE"] = problem_id
    env["PYRENEES_GMM_PATH"] = gmm_scaler_path
    if gpu_id is not None and gpu_id != "":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_file = log_dir / f"train_{problem_id}_{method_clean}.log"

    cmd = [
        python_bin,
        "src/train.py",
        f"group=pyrenees/{exp_id}",
        f"experiment_id={problem_id}",
        f"agent={method}",
        "mode=offline",
        "env=pyrenees",
        f"env.rules={ruleset}",
        f"mode.dataset_path={dataset_path}",
        f"agent.epochs_per_interval={epochs}",
        f"agent.batch_size={batch_size}",
        f"agent.lr={lr}",
        f"agent.cql_alpha={cql_alpha}",
        f"agent.name={method_clean}",
        "agent.eval_interval_epochs=5",
        "+trainer.enable_progress_bar=False",
    ]

    print(f"[{problem_id}] Starting training on GPU={gpu_id or 'all/cpu'} -> Log: {log_file}")
    start_time = time.time()

    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(PROJECT_ROOT))

    elapsed = time.time() - start_time

    if proc.returncode != 0:
        print(f"[{problem_id}] ❌ FAILED after {elapsed:.1f}s (code {proc.returncode}). Check {log_file}")
        return {
            "problem_id": problem_id,
            "status": "failed",
            "elapsed": elapsed,
            "log": str(log_file),
        }

    # Locate generated checkpoint
    expected_ckpt = PROJECT_ROOT / "results" / "checkpoints" / f"pyrenees/{exp_id}" / problem_id / method_clean / "0" / "best_model.ckpt"
    if not expected_ckpt.exists():
        # Search recursively
        candidates = list(Path(PROJECT_ROOT / "results" / "checkpoints" / f"pyrenees/{exp_id}").glob(f"**/{problem_id}/**/best_model*.ckpt"))
        if candidates:
            expected_ckpt = candidates[0]

    # Create convenient per_problem symlink / copy for export script
    export_dest = PROJECT_ROOT / "results" / "checkpoints" / "pyrenees" / "per_problem" / problem_id / "best_model.ckpt"
    export_dest.parent.mkdir(parents=True, exist_ok=True)
    if expected_ckpt.exists():
        shutil.copy2(expected_ckpt, export_dest)

    print(f"[{problem_id}] ✅ COMPLETED in {elapsed:.1f}s. Checkpoint: {export_dest}")
    return {
        "problem_id": problem_id,
        "status": "success",
        "elapsed": elapsed,
        "ckpt": str(export_dest),
        "log": str(log_file),
    }


def main():
    parser = argparse.ArgumentParser(description="Train all 11 Pyrenees problem models in parallel.")
    parser.add_argument("--method", type=str, default="cql/blendrl_human_dueling_resnet", help="Agent method to train (e.g. 'cql/blendrl_human_dueling_resnet', 'cql/dueling_resnet').")
    parser.add_argument("--exp-id", type=str, default="pyrenees_multi_model", help="Experiment ID for grouping logs and checkpoints.")
    parser.add_argument("--problems", type=str, default=None, help="Comma-separated problem IDs. Default: all 11.")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per model.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--cql-alpha", type=float, default=1.0, help="CQL alpha parameter.")
    parser.add_argument("--parallel", action="store_true", default=True, help="Run trainings concurrently.")
    parser.add_argument("--max-workers", type=int, default=11, help="Max concurrent worker processes.")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU indices to allocate (e.g. '0,1,2,3,4,5' or '0').")
    args = parser.parse_args()

    python_bin = sys.executable
    if (PROJECT_ROOT / "venv" / "bin" / "python").exists():
        python_bin = str(PROJECT_ROOT / "venv" / "bin" / "python")

    problems = ALL_PROBLEMS
    if args.problems:
        problems = [p.strip() for p in args.problems.split(",") if p.strip()]

    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()] if args.gpus else [None]

    print("=" * 75)
    print("      PYRENEES MULTI-MODEL TRAINING PIPELINE (11 PROBLEMS)")
    print("=" * 75)
    print(f"  Method:       {args.method}")
    print(f"  Exp ID:       {args.exp_id}")
    print(f"  Problems:     {len(problems)} models ({', '.join(problems)})")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch Size:   {args.batch_size}")
    print(f"  Parallel:     {args.parallel} (Max workers: {args.max_workers})")
    print(f"  GPUs:         {gpu_list}")
    print(f"  Python:       {python_bin}")
    print("=" * 75)

    start_all = time.time()
    results = []

    if args.parallel and len(problems) > 1:
        with ProcessPoolExecutor(max_workers=min(args.max_workers, len(problems))) as executor:
            futures = {}
            for i, pid in enumerate(problems):
                assigned_gpu = gpu_list[i % len(gpu_list)]
                f = executor.submit(
                    train_single_problem,
                    problem_id=pid,
                    method=args.method,
                    exp_id=args.exp_id,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    cql_alpha=args.cql_alpha,
                    gpu_id=assigned_gpu,
                    python_bin=python_bin,
                )
                futures[f] = pid

            for f in as_completed(futures):
                res = f.result()
                results.append(res)
    else:
        for i, pid in enumerate(problems):
            assigned_gpu = gpu_list[i % len(gpu_list)]
            res = train_single_problem(
                problem_id=pid,
                method=args.method,
                exp_id=args.exp_id,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                cql_alpha=args.cql_alpha,
                gpu_id=assigned_gpu,
                python_bin=python_bin,
            )
            results.append(res)

    total_time = time.time() - start_all
    print("\n" + "=" * 75)
    print("      TRAINING SUMMARY")
    print("=" * 75)
    n_success = sum(1 for r in results if r["status"] == "success")
    print(f"  Completed: {n_success} / {len(problems)} models successfully in {total_time/60:.2f} minutes.")
    for r in results:
        status_sym = "✅" if r["status"] == "success" else "❌"
        print(f"    {status_sym} {r['problem_id']:15s}: {r['status']} ({r['elapsed']:.1f}s)")
    print("=" * 75)


if __name__ == "__main__":
    main()
