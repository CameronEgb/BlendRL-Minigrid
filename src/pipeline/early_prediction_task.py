"""Early prediction task execution module.

Handles standalone early prediction tasks: deep learning sweeps,
checkpoint evaluation, and Optuna hyperparameter tuning.
"""
import shutil
import subprocess
import sys
from pathlib import Path

from src.pipeline.datasets import run_plotting
from src.pipeline.runtime import get_python_executable, get_shell_python_cmd, get_shell_env_block
from src.pipeline.slurm import generate_sbatch_header, submit_sbatch
from src.pipeline.task_registry import register_task

@register_task("early_prediction")
def run_early_prediction_task_wrapper(cfg, context):
    run_early_prediction_task(
        cfg, 
        context.get("is_interactive", context.get("local_val", True)), 
        context.get("sanitized_extra_args", []), 
        context.get("storage_url"), 
        context.get("is_sweep", False)
    )

def run_early_prediction_task(cfg, local_val, sanitized_extra_args, storage_url, is_sweep):
    """Handle standalone early_prediction task types (sweeps, evals, tuning).
    
    Returns True if a task was handled and completed, False otherwise.
    """
    task_name = cfg.get("task", "")
    if not task_name.startswith("early_prediction"):
        return False
        
    site_cfg = cfg.get("site", None)

    if task_name == "early_prediction_sweep":
        if local_val:
            print(f"Running Local Sweep for {cfg.experiment_id}...")
            venv_python = get_python_executable(site_cfg)
            cmd = [venv_python, "-u", "src/early_prediction/model.py", f"+experiment={cfg.experiment_id}"]
            if sanitized_extra_args:
                cmd.extend(sanitized_extra_args)
            print(f"Executing: {' '.join(cmd)}")
            res = subprocess.run(cmd)
            if not cfg.get("no_plot", False) and res.returncode == 0:
                run_plotting(cfg.experiment_id, style=cfg.get("plot_style", None), base_experiment=cfg.get("experiment_name", ""), site_cfg=site_cfg)
            sys.exit(res.returncode)
        else:
            slurm_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
            slurm_dir.mkdir(parents=True, exist_ok=True)
            
            target_models = ["lstm_no_v", "lstm_with_v", "transformer_no_v", "transformer_with_v"]
            print(f"Submitting 4 parallel SLURM model jobs ({target_models}) for {cfg.experiment_id}...")
            
            job_ids = []
            for tm in target_models:
                slurm_script_path = slurm_dir / f"early_pred_sweep_{tm}.slurm"
                
                log_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
                from omegaconf import OmegaConf
                local_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                if "resources" not in local_cfg:
                    local_cfg.resources = {}
                local_cfg.resources.cores = 16

                header = generate_sbatch_header(
                    job_name=f"ep_{tm}_{cfg.experiment_id}",
                    log_dir=log_dir,
                    cfg=local_cfg
                )
                
                env_block = get_shell_env_block(site_cfg)
                python_cmd = get_shell_python_cmd(site_cfg)
                
                extra_args_str = " ".join([f'"{a}"' if " " in a else a for a in sanitized_extra_args]) if sanitized_extra_args else ""
                
                script_content = f"""{header}
echo "=== Sepsis Early Prediction Sweep Execution Start ({tm}) ==="
echo "Node: $(hostname)"
date

{env_block}
mkdir -p results/plots/early_prediction
mkdir -p results/logs

{python_cmd} -u src/early_prediction/model.py \
    +experiment={cfg.experiment_id} \
    ++early_prediction.target_model={tm} \
    {extra_args_str}

echo "=== Sepsis Early Prediction Sweep Execution End ({tm}) ==="
date
"""
                with open(slurm_script_path, "w") as f:
                    f.write(script_content)
                print(f"Submitting Model SLURM Job ({tm}): {slurm_script_path}")
                
                job_id = submit_sbatch(script_content)
                if job_id:
                    job_ids.append(job_id)

            # Submit dependent SLURM job to run plotting after all parallel jobs finish
            if not cfg.get("no_plot", False) and job_ids:
                plot_slurm_script = slurm_dir / "early_pred_sweep_plot.slurm"
                dep_str = ":".join(job_ids)
                plot_cmd_str = f"{get_shell_python_cmd(site_cfg)} plot/manager.py {cfg.experiment_id}"
                if cfg.get("plot_style", None):
                    plot_cmd_str += f" --style {cfg.get('plot_style', None)}"

                log_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
                from omegaconf import OmegaConf
                plot_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                if "resources" not in plot_cfg:
                    plot_cfg.resources = {}
                plot_cfg.resources.gpus = 0
                plot_cfg.resources.cores = 4

                header = generate_sbatch_header(
                    job_name=f"ep_plot_{cfg.experiment_id}",
                    log_dir=log_dir,
                    cfg=plot_cfg,
                    dependency=dep_str,
                    dependency_type="afterok"
                )
                
                env_block = get_shell_env_block(site_cfg)
                
                plot_script_content = f"""{header}
echo "=== Early Prediction Sweep Plotting Start ==="
echo "Node: $(hostname)"
date

{env_block}

echo "Running plotting script..."
{plot_cmd_str}

echo "=== Early Prediction Sweep Plotting End ==="
date
"""
                with open(plot_slurm_script, "w") as f:
                    f.write(plot_script_content)
                
                print(f"Submitting Plotting SLURM Job (Dependent on {dep_str}): {plot_slurm_script}")
                submit_sbatch(plot_script_content)

            sys.exit(0)
    elif task_name == "early_prediction_eval":
        print(f"\n=== Running Early Prediction Checkpoint Evaluation ({cfg.experiment_id}) ===")
        if local_val:
            cmd = [get_python_executable(site_cfg), "-u", "plot/manager.py", str(cfg.experiment_id)]
            print(f"Executing: {' '.join(cmd)}")
            res = subprocess.run(cmd)
            sys.exit(res.returncode)
        else:
            slurm_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
            slurm_dir.mkdir(parents=True, exist_ok=True)
            slurm_script_path = slurm_dir / "early_pred_eval.slurm"
            
            log_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
            from omegaconf import OmegaConf
            local_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            if "resources" not in local_cfg:
                local_cfg.resources = {}
            local_cfg.resources.cores = 16

            header = generate_sbatch_header(
                job_name=f"eval_pred_{cfg.experiment_id}",
                log_dir=log_dir,
                cfg=local_cfg
            )
            
            env_block = get_shell_env_block(site_cfg)
            python_cmd = get_shell_python_cmd(site_cfg)
            
            script_content = f"""{header}
{env_block}

{python_cmd} -u plot/manager.py {cfg.experiment_id}
"""
            with open(slurm_script_path, "w") as f:
                f.write(script_content)
            print(f"Submitting Early Prediction Eval SLURM Job: {slurm_script_path}")
            submit_sbatch(script_content)
            sys.exit(0)

    elif task_name in ("early_prediction_tune", "early_prediction_optuna"):
        print(f"\n=== Running Early Prediction Modular Optuna Hyperparameter Search ({cfg.experiment_id}) ===")
        ep_cfg = cfg.get("early_prediction", {})
        target_models = ep_cfg.get("target_models", ["lstm_no_v", "lstm_with_v", "transformer_no_v", "transformer_with_v"])
        if isinstance(target_models, str):
            target_models = [m.strip() for m in target_models.split(",")]
            
        slurm_dir = Path("results/logs/slurm") / cfg.group / cfg.experiment_id
        slurm_dir.mkdir(parents=True, exist_ok=True)
        
        for m_target in target_models:
            print(f"\n--> Setting up Optuna Study for architecture target: [{m_target}]")
            
            out_dir = ep_cfg.get("output_dir", f"results/plots/early_prediction/{cfg.experiment_id}")
            stray_study_dir = Path(out_dir) / "optuna_study"
            if stray_study_dir.exists() and stray_study_dir.is_dir():
                shutil.rmtree(stray_study_dir)
                
            if local_val:
                cmd = [get_python_executable(site_cfg), "-u", "src/early_prediction/tune_optuna.py", f"+experiment={cfg.experiment_id}", f"++early_prediction.model_target={m_target}"]
                if sanitized_extra_args:
                    cmd.extend(sanitized_extra_args)
                print(f"Executing local: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            else:
                slurm_script_path = slurm_dir / f"tune_pred_{m_target}.slurm"
                from omegaconf import OmegaConf
                local_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                if "resources" not in local_cfg:
                    local_cfg.resources = {}
                local_cfg.resources.cores = 16

                header = generate_sbatch_header(
                    job_name=f"tune_{m_target}_{cfg.experiment_id}",
                    log_dir=slurm_dir,
                    cfg=local_cfg
                )
                
                env_block = get_shell_env_block(site_cfg)
                python_cmd = get_shell_python_cmd(site_cfg)
                
                extra_args_str = " ".join([f'"{a}"' if " " in a else a for a in sanitized_extra_args]) if sanitized_extra_args else ""
                
                script_content = f"""{header}
echo "=== Sepsis Early Prediction Optuna Search [{m_target}] Start ==="
echo "Node: $(hostname)"
date

{env_block}

mkdir -p {out_dir}
mkdir -p results/logs

{python_cmd} -u src/early_prediction/tune_optuna.py \
    +experiment={cfg.experiment_id} \
    ++early_prediction.model_target={m_target} \
    {extra_args_str}

echo "=== Sepsis Early Prediction Optuna Search [{m_target}] End ==="
date
"""
                with open(slurm_script_path, "w") as f:
                    f.write(script_content)
                print(f"Submitting Early Prediction Optuna SLURM Job [{m_target}]: {slurm_script_path}")
                submit_sbatch(script_content)
        sys.exit(0)
    return True
