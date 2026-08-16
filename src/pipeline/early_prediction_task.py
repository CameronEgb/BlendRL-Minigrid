"""Early prediction task execution module.

Handles standalone early prediction tasks: deep learning sweeps,
checkpoint evaluation, and Optuna hyperparameter tuning.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from src.pipeline.datasets import run_plotting
from src.pipeline.optuna_utils import get_python_executable
from src.pipeline.slurm import generate_sbatch_header


def build_early_pred_args(cfg):
    """Construct CLI argument list from early_prediction config dict."""
    ep_c = cfg.get("early_prediction", {})
    a_list = []
    flag_map = {
        "checkpoint": "--checkpoint",
        "dataset_path": "--dataset-path",
        "output_dir": "--output-dir",
        "tune_dir": "--tune-dir",
        "use_tuned_params": "--use-tuned-params",
        "save_checkpoints": "--save-checkpoints",
        "n_splits": "--n-splits",
        "tau_min": "--tau-min",
        "tau_max": "--tau-max",
        "tau_step": "--tau-step",
        "window_hours": "--window-hours",
        "epochs": "--epochs",
        "batch_size": "--batch-size",
        "lr": "--lr",
        "d_model": "--d-model",
        "nhead": "--nhead",
        "n_layers": "--n-layers",
    }
    for k, flag in flag_map.items():
        if k in ep_c and ep_c[k] is not None:
            if isinstance(ep_c[k], bool):
                if ep_c[k]:
                    a_list.append(flag)
            else:
                a_list.extend([flag, str(ep_c[k])])
    if ep_c.get("use_all_history", False):
        a_list.append("--use-all-history")
    if ep_c.get("use_all_trajectories", False):
        a_list.append("--use-all-trajectories")
    if ep_c.get("no_norm", False):
        a_list.append("--no-norm")
    return a_list


def run_early_prediction_task(cfg, args, local_val, sanitized_extra_args, storage_url, is_sweep):
    """Handle standalone early_prediction task types (sweeps, evals, tuning).
    
    Returns True if a task was handled and completed, False otherwise.
    """
    task_name = cfg.get("task", "")
    if not task_name.startswith("early_prediction"):
        return False

    python_exe = get_python_executable()

    if task_name in ("early_prediction_sweep", "early_prediction"):
        print(f"\n=== Running Early Prediction Deep Learning Sweep ({cfg.experiment_id}) ===")
        ep_args = build_early_pred_args(cfg)
        if local_val:
            cmd = [python_exe, "-u", "src/early_prediction/model.py", "--exp-id", cfg.experiment_id] + ep_args
            print(f"Executing: {' '.join(cmd)}")
            res = subprocess.run(cmd)
            if not args.no_plot and res.returncode == 0:
                run_plotting(cfg.experiment_id, style=args.plot_style, base_experiment=args.experiment)
            sys.exit(res.returncode)
        else:
            slurm_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
            slurm_dir.mkdir(parents=True, exist_ok=True)
            
            target_models = ["lstm_no_v", "lstm_with_v", "transformer_no_v", "transformer_with_v"]
            print(f"Submitting 4 parallel SLURM model jobs ({target_models}) for {cfg.experiment_id}...")
            
            job_ids = []
            for tm in target_models:
                slurm_script_path = slurm_dir / f"early_pred_sweep_{tm}.slurm"
                cmd_args = ep_args + ["--target-model", tm]
                cmd_str = " ".join([f'"{arg}"' if " " in arg else arg for arg in cmd_args])
                log_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
                header = generate_sbatch_header(
                    job_name=f"ep_{tm}_{cfg.experiment_id}",
                    log_dir=log_dir,
                    partition=args.partition,
                    gpus=args.gpus,
                    cores=16
                )
                script_content = f"""{header}
echo "=== Sepsis Early Prediction Sweep Execution Start ({tm}) ==="
echo "Node: $(hostname)"
date

export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/fyd_repo/src:$PYTHONPATH

mkdir -p results/plots/early_prediction
mkdir -p results/logs

$PROJECT_ROOT/venv/bin/python3 -u src/early_prediction/model.py \\
    --exp-id "{cfg.experiment_id}" \\
    {cmd_str}

echo "=== Sepsis Early Prediction Sweep Execution End ({tm}) ==="
date
"""
                with open(slurm_script_path, "w") as f:
                    f.write(script_content)
                print(f"Submitting Model SLURM Job ({tm}): {slurm_script_path}")
                res = subprocess.run(["sbatch", str(slurm_script_path)], capture_output=True, text=True)
                print(res.stdout)
                if res.stderr:
                    print(res.stderr)
                out_text = res.stdout.strip()
                if "Submitted batch job" in out_text:
                    j_id = out_text.split()[-1]
                    job_ids.append(j_id)

            # Submit dependent SLURM job to run plotting after all parallel jobs finish
            if not args.no_plot and job_ids:
                plot_slurm_script = slurm_dir / "early_pred_sweep_plot.slurm"
                dep_str = ":".join(job_ids)
                plot_cmd_str = f"$PROJECT_ROOT/venv/bin/python3 plot/manager.py {cfg.experiment_id}"
                if args.plot_style:
                    plot_cmd_str += f" --style {args.plot_style}"

                log_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
                header = generate_sbatch_header(
                    job_name=f"ep_plot_{cfg.experiment_id}",
                    log_dir=log_dir,
                    partition=args.partition,
                    gpus=0,
                    cores=4
                )
                plot_script_content = f"""{header}
echo "=== Sepsis Early Prediction Plotting Execution Start ==="
echo "Node: $(hostname)"
date

export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/fyd_repo/src:$PYTHONPATH

{plot_cmd_str}

echo "=== Sepsis Early Prediction Plotting Execution End ==="
date
"""
                with open(plot_slurm_script, "w") as f:
                    f.write(plot_script_content)
                
                sbatch_cmd = ["sbatch", f"--dependency=afterok:{dep_str}", str(plot_slurm_script)]
                print(f"\nSubmitting Dependent Plotting SLURM Job (dependency afterok:{dep_str}): {' '.join(sbatch_cmd)}")
                res_plot = subprocess.run(sbatch_cmd, capture_output=True, text=True)
                print(res_plot.stdout)
                if res_plot.stderr:
                    print(res_plot.stderr)

            sys.exit(0)

    elif task_name == "early_prediction_eval":
        print(f"\n=== Running Early Prediction Checkpoint Evaluation ({cfg.experiment_id}) ===")
        ep_cfg = cfg.get("early_prediction", {})
        ckpt = ep_cfg.get("checkpoint", f"results/checkpoints/{cfg.group}/{cfg.experiment_id}")
        dataset_path = ep_cfg.get("dataset_path", "in/datasets/mimic/mimic_lazy_12_clean_with_interventions_corrected.npz")
        output_dir = ep_cfg.get("output_dir", f"results/plots/{cfg.group}/{cfg.experiment_id}")
        n_splits = ep_cfg.get("n_splits", 20)
        ep_ckpt_root = ep_cfg.get("ep_ckpt_root", "results/checkpoints/early_prediction")

        eval_cmd_args = [
            "--checkpoint", str(ckpt),
            "--dataset-path", str(dataset_path),
            "--output-dir", str(output_dir),
            "--n-splits", str(n_splits),
            "--ep-ckpt-root", str(ep_ckpt_root),
            "--remake",
        ]

        if local_val:
            cmd = [python_exe, "-u", "src/early_prediction/eval.py"] + eval_cmd_args
            print(f"Executing: {' '.join(cmd)}")
            res = subprocess.run(cmd)
            if not args.no_plot and res.returncode == 0:
                run_plotting(cfg.experiment_id, style=args.plot_style, base_experiment=args.experiment)
            sys.exit(res.returncode)
        else:
            slurm_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
            slurm_dir.mkdir(parents=True, exist_ok=True)
            slurm_script_path = slurm_dir / "early_pred_eval.slurm"
            eval_args_str = " ".join(eval_cmd_args)
            plot_cmd_str = f"$PROJECT_ROOT/venv/bin/python3 plot/manager.py {cfg.experiment_id}"
            if args.plot_style:
                plot_cmd_str += f" --style {args.plot_style}"
            log_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
            header = generate_sbatch_header(
                job_name=f"eval_pred_{cfg.experiment_id}",
                log_dir=log_dir,
                partition=args.partition,
                gpus=args.gpus,
                cores=16
            )
            script_content = f"""{header}
export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/fyd_repo/src:$PYTHONPATH

$PROJECT_ROOT/venv/bin/python3 -u src/early_prediction/eval.py {eval_args_str}
{plot_cmd_str if not args.no_plot else ""}
"""
            with open(slurm_script_path, "w") as f:
                f.write(script_content)
            print(f"Submitting Early Prediction Eval SLURM Job: {slurm_script_path}")
            res = subprocess.run(["sbatch", str(slurm_script_path)], capture_output=True, text=True)
            print(res.stdout)
            if res.stderr:
                print(res.stderr)
            sys.exit(res.returncode)

    elif task_name in ("early_prediction_tune", "early_prediction_optuna"):
        print(f"\n=== Running Early Prediction Modular Optuna Hyperparameter Search ({cfg.experiment_id}) ===")
        ep_cfg = cfg.get("early_prediction", {})
        n_trials = ep_cfg.get("n_trials", 50)
        ckpt = ep_cfg.get("checkpoint", "results/checkpoints/mimic/tune_mimic_all")
        dataset_path = ep_cfg.get("dataset_path", "in/datasets/mimic/mimic_lazy_12_clean_with_interventions_corrected.npz")
        out_dir = ep_cfg.get("output_dir", f"results/plots/early_prediction/{cfg.experiment_id}")
        
        metric = ep_cfg.get("metric", "auprc")
        n_eval_splits = ep_cfg.get("n_eval_splits", 5)
        target_models = ep_cfg.get("target_models", ["lstm_no_v", "lstm_with_v", "transformer_no_v", "transformer_with_v"])
        if isinstance(target_models, str):
            target_models = [m.strip() for m in target_models.split(",")]
            
        slurm_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
        slurm_dir.mkdir(parents=True, exist_ok=True)
        
        for m_target in target_models:
            print(f"\n--> Setting up Optuna Study for architecture target: [{m_target}] (metric: {metric}, eval_splits: {n_eval_splits})")
            
            stray_study_dir = Path(out_dir) / "optuna_study"
            if stray_study_dir.exists() and stray_study_dir.is_dir():
                shutil.rmtree(stray_study_dir)
                
            tune_args = [
                "--n-trials", str(n_trials),
                "--model-target", str(m_target),
                "--checkpoint", str(ckpt),
                "--dataset-path", str(dataset_path),
                "--out-dir", str(out_dir),
                "--metric", str(metric),
                "--n-eval-splits", str(n_eval_splits)
            ]
            if local_val:
                cmd = [python_exe, "-u", "src/early_prediction/tune_optuna.py"] + tune_args
                print(f"Executing local: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            else:
                slurm_script_path = slurm_dir / f"tune_pred_{m_target}.slurm"
                cmd_str = " ".join([f'"{arg}"' if " " in arg else arg for arg in tune_args])
                header = generate_sbatch_header(
                    job_name=f"tune_{m_target}_{cfg.experiment_id}",
                    log_dir=slurm_dir,
                    partition=args.partition,
                    gpus=args.gpus,
                    cores=16
                )
                script_content = f"""{header}
echo "=== Sepsis Early Prediction Optuna Search [{m_target}] Start ==="
echo "Node: $(hostname)"
date

export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/fyd_repo/src:$PYTHONPATH

mkdir -p {out_dir}
mkdir -p results/logs

$PROJECT_ROOT/venv/bin/python3 -u src/early_prediction/tune_optuna.py \\
    {cmd_str}

echo "=== Sepsis Early Prediction Optuna Search [{m_target}] End ==="
date
"""
                with open(slurm_script_path, "w") as f:
                    f.write(script_content)
                print(f"Submitting Early Prediction Optuna SLURM Job [{m_target}]: {slurm_script_path}")
                res = subprocess.run(["sbatch", str(slurm_script_path)], capture_output=True, text=True)
                print(res.stdout)
                if res.stderr:
                    print(res.stderr)
        sys.exit(0)
    return True
