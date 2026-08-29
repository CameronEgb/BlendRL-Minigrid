"""Slurm cluster utilities.

Provides functions for generating SBATCH scripts and submitting jobs.
All cluster-specific values (email, partition, cores) are read from site config.
"""
import re
import subprocess

from src.pipeline.runtime import get_shell_env_block, get_shell_python_cmd


from omegaconf import OmegaConf

def get_gres_header(res, site_cfg):
    """Generate the appropriate --gres header.
    
    Emits --gres=gpu:{gpus} automatically when gpus > 0 unless explicitly disabled.
    """
    gpus = res.get("gpus")
    gres = res.get("gres")
    gpu_type = res.get("gpu_type")
    no_gres = res.get("no_gres") or getattr(site_cfg, "no_gres", False)
    partition = str(res.get("partition", "")).lower()
    is_cpu_partition = partition in ("common", "serial", "cpu", "standard", "debug_cpu")

    if no_gres or is_cpu_partition or gres in ("none", "False", "false") or not gpus or int(gpus) == 0:
        return ""
    if gres:
        return f"#SBATCH --gres={gres}\n"
    
    # Optional site-specific gres format template (e.g., "gpu:{gpu_type}:{gpus}")
    gres_format = getattr(site_cfg, "gres_format", None)
    if gres_format and gpu_type:
        return f"#SBATCH --gres={gres_format.format(gpu_type=gpu_type, gpus=gpus)}\n"
    elif gres_format and not gpu_type and "{gpu_type}" not in gres_format:
        return f"#SBATCH --gres={gres_format.format(gpus=gpus)}\n"
        
    if gpu_type:
        return f"#SBATCH --gres=gpu:{gpu_type}:{gpus}\n"
    return f"#SBATCH --gres=gpu:{gpus}\n"


def generate_sbatch_header(job_name, log_dir, cfg, dependency=None, dependency_type="afterok", is_consolidated=False):
    """Generate standardized SBATCH script header using pure Hydra composition."""
    site_cfg = getattr(cfg, "site", None)
    
    # Merge site resources with experiment resources (experiment overrides site)
    site_res_raw = getattr(site_cfg, "resources", {}) if site_cfg else {}
    exp_res_raw = getattr(cfg, "resources", {}) if hasattr(cfg, "resources") else {}
    site_dict = OmegaConf.to_container(site_res_raw, resolve=True) if OmegaConf.is_config(site_res_raw) else (dict(site_res_raw) if isinstance(site_res_raw, dict) else {})
    exp_dict = OmegaConf.to_container(exp_res_raw, resolve=True) if OmegaConf.is_config(exp_res_raw) else (dict(exp_res_raw) if isinstance(exp_res_raw, dict) else {})
    res = {**site_dict, **exp_dict}
    
    partition = res.get("partition")
    if is_consolidated or cfg.get("consolidate", False):
        cores = res.get("consolidated_cores", res.get("cores", 16))
        memory = res.get("consolidated_memory", res.get("memory", "32G"))
    else:
        cores = res.get("standalone_cores", res.get("cores", 1))
        memory = res.get("standalone_memory", res.get("memory", "8G"))
    time = res.get("time")
    nodes = res.get("nodes", 1)
    
    mail_user = getattr(site_cfg, "mail_user", None)
    mail_type = getattr(site_cfg, "mail_type", "END,FAIL")

    script = "#!/bin/bash\n"
    script += f"#SBATCH --job-name={job_name}\n"
    if partition:
        script += f"#SBATCH --partition={partition}\n"
    script += get_gres_header(res, site_cfg)
    if time:
        script += f"#SBATCH --time={time}\n"
    if nodes:
        script += f"#SBATCH --nodes={nodes}\n"
    script += f"#SBATCH --ntasks=1\n"
    if cores:
        script += f"#SBATCH --cpus-per-task={cores}\n"
    if memory:
        script += f"#SBATCH --mem={memory}\n"
    script += f"#SBATCH --output={log_dir}/%x_%j.out\n"
    script += f"#SBATCH --error={log_dir}/%x_%j.err\n"

    # Extra site-specific sbatch flags
    for flag in getattr(site_cfg, "extra_sbatch_flags", []) or []:
        script += f"#SBATCH {flag}\n"

    if mail_user:
        script += f"#SBATCH --mail-type={mail_type}\n"
        script += f"#SBATCH --mail-user={mail_user}\n"
    if dependency:
        script += f"#SBATCH --dependency={dependency_type}:{dependency}\n"
    return script


def generate_sbatch_script(job_name, cmd_args, log_dir, cfg, dependency=None, is_consolidated=False):
    """Generate an SBATCH script string for Slurm submission."""
    import shlex

    script = generate_sbatch_header(
        job_name=job_name,
        log_dir=log_dir,
        cfg=cfg,
        dependency=dependency,
        is_consolidated=is_consolidated
    )
    script += "\n"
    
    site_cfg = getattr(cfg, "site", None)
    script += get_shell_env_block(site_cfg)
    script += "\n"

    # Construct the python command using site-aware venv path
    python_cmd = get_shell_python_cmd(site_cfg)
    cmd_str = python_cmd + " src/train.py " + " ".join(shlex.quote(arg) for arg in cmd_args)
    script += f'echo "Running: {cmd_str}"\n'
    script += f"{cmd_str}\n"
    
    return script


def submit_sbatch(script_content):
    """Submit an SBATCH script to Slurm and return the job ID.
    
    Uses --parsable for robust job ID extraction.
    Returns None if submission fails or sbatch is not available.
    """
    try:
        # Use --parsable for reliable job ID output (just the number)
        result = subprocess.run(
            ["sbatch", "--parsable"], 
            input=script_content, 
            capture_output=True, 
            text=True, 
            check=True
        )
        job_id = result.stdout.strip().split(";")[0]  # --parsable may include cluster name after ;
        if job_id.isdigit():
            print(f" -> Job ID: {job_id}", flush=True)
            return job_id
        else:
            # Fallback: try regex parsing for non-standard sbatch output
            match = re.search(r"(\d+)", result.stdout)
            if match:
                job_id = match.group(1)
                print(f" -> Job ID: {job_id}", flush=True)
                return job_id
            print(f" -> Could not parse job ID from: {result.stdout}", flush=True)
            return None
    except subprocess.CalledProcessError as e:
        print(f" -> Error submitting job: {e.stderr.strip()}", flush=True)
        return None
    except FileNotFoundError:
        print(" -> (sbatch not available locally)", flush=True)
        return None
