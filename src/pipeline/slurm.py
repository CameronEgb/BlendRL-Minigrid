"""Slurm cluster utilities.

Provides functions for generating SBATCH scripts and submitting jobs.
"""
import os
import re
import subprocess


def get_gres_header(partition, gpus, gpu_type=None, gres=None, no_gres=False):
    """Generate the appropriate --gres header.
    
    Emits --gres=gpu:{gpus} automatically when gpus > 0 unless explicitly disabled.
    """
    if no_gres or gres in ("none", "False", "false") or not gpus or int(gpus) == 0:
        return ""
    if gres:
        return f"#SBATCH --gres={gres}\n"
    if gpu_type:
        return f"#SBATCH --gres=gpu:{gpu_type}:{gpus}\n"
    return f"#SBATCH --gres=gpu:{gpus}\n"


def generate_sbatch_header(job_name, log_dir, partition="gpu", gpus=1, cores=16, nodes=1, time="01:00:00", gpu_type=None, gres=None, no_gres=False, dependency=None, dependency_type="afterok", mail_user="cegbert@ncsu.edu", mail_type="END,FAIL"):
    """Generate standardized SBATCH script header."""
    script = "#!/bin/bash\n"
    script += f"#SBATCH --job-name={job_name}\n"
    if partition:
        script += f"#SBATCH --partition={partition}\n"
    script += get_gres_header(partition, gpus, gpu_type=gpu_type, gres=gres, no_gres=no_gres)
    script += f"#SBATCH --time={time}\n"
    script += f"#SBATCH --ntasks-per-node={cores}\n"
    script += f"#SBATCH --nodes={nodes}\n"
    script += f"#SBATCH --output={log_dir}/%x_%j.out\n"
    script += f"#SBATCH --error={log_dir}/%x_%j.err\n"
    if mail_user:
        script += f"#SBATCH --mail-type={mail_type}\n"
        script += f"#SBATCH --mail-user={mail_user}\n"
    if dependency:
        script += f"#SBATCH --dependency={dependency_type}:{dependency}\n"
    return script


def generate_sbatch_script(job_name, cmd_args, log_dir, partition="gpu", gpus=1, cores=16, nodes=1, gpu_type=None, gres=None, no_gres=False, dependency=None, time="01:00:00"):
    """Generate an SBATCH script string for Slurm submission."""
    script = generate_sbatch_header(
        job_name=job_name,
        log_dir=log_dir,
        partition=partition,
        gpus=gpus,
        cores=cores,
        nodes=nodes,
        time=time,
        gpu_type=gpu_type,
        gres=gres,
        no_gres=no_gres,
        dependency=dependency
    )
    script += f"\n"
    script += f"export PROJECT_ROOT={os.getcwd()}\n"
    script += f"export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/src/fyd_repo/src:$PYTHONPATH\n"
    
    # Construct the python command with absolute venv path
    import shlex
    cmd_str = "$PROJECT_ROOT/venv/bin/python3 " + " ".join(shlex.quote(arg) for arg in cmd_args)
    script += f'echo "Running: {cmd_str}"\n'
    script += f"{cmd_str}\n"
    
    return script


def submit_sbatch(script_content):
    """Submit an SBATCH script to Slurm and return the job ID."""
    print(f"Submitting Slurm job via stdin...")
    try:
        # Pipe script_content directly to sbatch stdin
        result = subprocess.run(
            ["sbatch"], 
            input=script_content, 
            capture_output=True, 
            text=True, 
            check=True
        )
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            job_id = match.group(1)
            print(f"-> Job ID: {job_id}")
            return job_id
        else:
            print(f"Could not parse job ID from: {result.stdout}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: 'sbatch' command not found. Are you on the Slurm cluster?")
        return "99999"
