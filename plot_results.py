import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml
import pandas as pd
from collections import defaultdict

def get_style_info(label):
    l = label.lower()
    # If it's a specific tuning variant (v1, v2, etc.), return None to use cycle
    import re
    if re.search(r'_v\d+', l) or "tune" in l: 
        return None, "-", "o"
    
    # Check for the main algorithm name in the label for standard baselines
    if "ppo" in l and "(on" not in l: return "black", "--", "o"
    if "blendrl-iql" in l: return "#d62728", "-", "s"
    if "blendrl" in l and "iql" not in l and "(on" not in l: return "#2ca02c", "-", "^"
    if "iql" in l and "blendrl" not in l: return "#1f77b4", "-", "d"
    return None, "-", "o"

def moving_average(a, n=5):
    if len(a) == 0: return np.array([])
    n = min(len(a), n) if len(a) > 0 else 1
    a_padded = np.pad(a, (n-1, 0), mode='edge')
    ret = np.cumsum(a_padded, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def get_base_name(name):
    import re
    return re.sub(r'(_s\d+|_seed\d+|_v\d+)', '', name)

def load_run_data(run_folder, args):
    config = None
    # Try multiple places for config
    search_paths = [
        run_folder / "config.yaml",
        run_folder / ".hydra" / "config.yaml",
        run_folder / "hparams.yaml",
        run_folder.parent / "config.yaml",
        run_folder.parent / "hparams.yaml",
        run_folder.parent.parent / "config.yaml", # Up one more for Hydra
    ]
    
    for cp in search_paths:
        if cp.exists():
            with open(cp, "r") as f:
                if cp.suffix == ".yaml":
                    import yaml
                    try:
                        config = yaml.safe_load(f)
                        # Lightning's hparams.yaml often nests under 'cfg'
                        if config and "cfg" in config:
                            config = config["cfg"]
                        if config:
                            break
                    except:
                        continue
    
    # Better label detection - extract base algorithm name
    agent_name = "UNKNOWN"
    folder_str = str(run_folder).lower()
    
    # Priority 1: Use folder name for specific label if it's not a generic 'version_X'
    config_folder = run_folder.parent if run_folder.name.startswith("version_") else run_folder
    folder_label = config_folder.name
    
    # Check for the main algorithm name in the label
    if "blendrl_iql" in folder_str: agent_name = "BLENDRL_IQL"
    elif "blendrl" in folder_str: agent_name = "BLENDRL"
    elif "iql" in folder_str: agent_name = "IQL"
    elif "ppo" in folder_str: agent_name = "PPO"
    # Priority 2: Config-based detection if path fails
    elif config and "agent" in config:
        raw_name = config["agent"].get("name", "UNKNOWN").upper()
        if "BLENDRL_IQL" in raw_name: agent_name = "BLENDRL_IQL"
        elif "BLENDRL" in raw_name: agent_name = "BLENDRL"
        elif "IQL" in raw_name: agent_name = "IQL"
        elif "PPO" in raw_name: agent_name = "PPO"
        else: agent_name = raw_name
    
    # Mapping for cleaner names
    name_map = {
        "BLENDRL_IQL": "BlendRL-IQL",
        "BLENDRL": "BlendRL",
        "IQL": "IQL",
        "PPO": "PPO"
    }
    
    # Use the specific folder name if it's more descriptive than just the base algorithm
    if folder_label.lower() != agent_name.lower() and folder_label.lower() != name_map.get(agent_name, "").lower():
        agent_display_name = folder_label
    else:
        agent_display_name = name_map.get(agent_name, agent_name)
    
    source = "ONLINE"
    mode = config.get("mode", {}).get("type") if config else "unknown"
    
    if mode == "offline":
        # In run_pipeline.py, dataset_path is results/datasets/[EXP_ID]/ppo
        dataset_path = config.get("mode", {}).get("dataset_path") or config.get("dataset_path", "")
        if dataset_path:
            raw_source = Path(dataset_path).name.upper()
            # Clean source name too
            if "BLENDRL_IQL" in raw_source: source_key = "BLENDRL_IQL"
            elif "BLENDRL" in raw_source: source_key = "BLENDRL"
            elif "IQL" in raw_source: source_key = "IQL"
            elif "PPO" in raw_source: source_key = "PPO"
            else: source_key = raw_source
            
            source = name_map.get(source_key, source_key)
            label = f"{agent_display_name} (on {source})"
        else:
            label = agent_display_name
    else:
        label = agent_display_name
        # Clean source name for online agents too
        raw_source = agent_display_name.upper()
        if "BLENDRL_IQL" in raw_source: source_key = "BLENDRL_IQL"
        elif "BLENDRL" in raw_source: source_key = "BLENDRL"
        elif "IQL" in raw_source: source_key = "IQL"
        elif "PPO" in raw_source: source_key = "PPO"
        else: source_key = raw_source
        
        source = name_map.get(source_key, source_key)
    
    exp_id = "UNKNOWN"
    if config and "experiment_id" in config:
        exp_id = config["experiment_id"]
    else:
        # Try to extract from path results/logs/[EXP_ID]/[AGENT]
        parts = run_folder.parts
        if "logs" in parts:
            idx = parts.index("logs")
            if len(parts) > idx + 1:
                exp_id = parts[idx+1]

    data = defaultdict(list)
    metrics_path = run_folder / "metrics.csv"
    if not metrics_path.exists(): 
        # Check subdirectories for metrics.csv if not found
        for p in run_folder.glob("**/metrics.csv"):
            metrics_path = p
            break
        
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        
        # If 'transitions' exists, we use it for alignment but keep steps for convergence
        if "transitions" in df.columns:
            df['transitions'] = df['transitions'].ffill().bfill()
            df['transitions'] = df['transitions'].round().astype(int)

        for col in df.columns:
            if col in ["step", "transitions", "epoch"]: continue
            subset = df[df[col].notna()].copy()
            
            # Filter by total_timesteps if available for transition-based plots
            total_timesteps = config.get("total_timesteps") if config else None
            
            if not subset.empty:
                # Store the raw values and multiple potential x-axes
                data[col] = {
                    "values": subset[col].tolist(),
                    "step": subset["step"].tolist() if "step" in subset.columns else [],
                    "epoch": subset["epoch"].tolist() if "epoch" in subset.columns else [],
                    "transitions": subset["transitions"].tolist() if "transitions" in subset.columns else []
                }
    
    return {"folder": run_folder, "label": label, "exp_id": exp_id, "data": data, "base_name": get_base_name(exp_id), "source": source, "mode": mode}

def aggregate_runs(runs, metric_name, x_axis_col="transitions"):
    # Find all unique x-axis points
    all_x = sorted(list(set(s for r in runs for s in r["data"].get(metric_name, {}).get(x_axis_col, []))))
    if not all_x: return [], [], [], []
    
    means, stds, valid_x, logged_stds = [], [], [], []
    std_metric = f"{metric_name}_std"
    
    # Pre-index runs for speed
    run_data = []
    for r in runs:
        if metric_name in r["data"]:
            m_data = r["data"][metric_name]
            x_vals = m_data.get(x_axis_col, [])
            y_vals = m_data.get("values", [])
            
            # Standard deviation for within-run variance
            s_data = r["data"].get(std_metric, {})
            sx_vals = s_data.get(x_axis_col, [])
            sy_vals = s_data.get("values", [])
            
            # Map x to y and x to std
            mapping = dict(zip(x_vals, y_vals))
            s_mapping = dict(zip(sx_vals, sy_vals))
            run_data.append((mapping, s_mapping))
    
    for x in all_x:
        vals = []
        l_stds = []
        for mapping, s_mapping in run_data:
            if x in mapping:
                vals.append(mapping[x])
                if x in s_mapping:
                    l_stds.append(s_mapping[x])
        
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            valid_x.append(x)
            logged_stds.append(np.mean(l_stds) if l_stds else 0.0)
            
    return valid_x, means, stds, logged_stds

def create_plot(exp_groups, metric, title, ylabel, save_path, window=1, use_simple_labels=False, x_axis_col="transitions", xlabel="Transitions (Dataset Size)"):
    plt.figure(figsize=(12, 7))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Use a cycling palette if we have many runs or it's a tuning run
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_groups)))
    markers = ['o', 's', '^', 'v', '<', '>', 'd', 'p', '*', 'h']
    
    for i, ((exp_id, label), runs) in enumerate(sorted(exp_groups.items())):
        steps, means, stds, logged_stds = aggregate_runs(runs, metric, x_axis_col=x_axis_col)
        if not steps: continue
        print(f"  Plotting {label} in {exp_id} ({metric}) with {len(steps)} points")
        
        # Determine the shading values
        shading = np.array(stds) if len(runs) > 1 else np.array(logged_stds)
        
        if window > 1:
            means = moving_average(means, n=window)
            shading = moving_average(shading, n=window)
            steps = steps[:len(means)]
        
        # Priority 1: Hardcoded styles for standard comparison plots
        style = get_style_info(label)
        color, ls, marker = style if style else (None, "-", "o")
        
        # Priority 2: If color is None (tuning or unknown), use cycle
        if color is None:
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
        
        # Don't use markers for high-density training plots
        actual_marker = marker if x_axis_col == "transitions" else None
        
        legend_label = label if use_simple_labels else f"{label} ({exp_id})"
        
        plt.plot(steps, means, label=legend_label, color=color, linestyle=ls, marker=actual_marker, markersize=4)
        if np.any(shading > 0):
            plt.fill_between(steps, np.array(means)-shading, np.array(means)+shading, color=color, alpha=0.1)
            
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.2); plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"  Saved: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Experiment ID or substring to filter")
    parser.add_argument("--version", type=str, default=None, help="Specific version (e.g., 0), 'all' to aggregate, or latest by default.")
    parser.add_argument("--style", type=str, default=None, help="Path to a YAML config file defining plot styles and groupings.")
    args = parser.parse_args()

    search_dirs = [Path("results/logs"), Path("results/experiments")]
    
    experiment_filters = args.experiment.split(',')
    
    # Structure: {(exp_id, agent_path): {version_num: run_folder}}
    # Load configuration to get method lists for filtering
    try:
        from hydra import compose, initialize
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()
        # Filter out experiment names from commas for multi-exp plotting
        primary_exp = experiment_filters[0]
        initialize(version_base=None, config_path="conf")
        exp_cfg = compose(config_name="config", overrides=[f"+experiment={primary_exp}"])
        
        allowed_methods = set()
        for key in ["online_methods", "offline_methods"]:
            val = exp_cfg.get(key, "")
            if val:
                if isinstance(val, (list, tuple)): 
                    methods = val
                else:
                    methods = [item.strip() for item in str(val).split(",") if item.strip()]
                for m in methods:
                    allowed_methods.add(m)
                    allowed_methods.add(m.replace("/", "_"))
        
        print(f"Filtering plots to methods defined in {primary_exp}: {allowed_methods}")
    except Exception as e:
        print(f"Warning: Could not load experiment config for filtering ({e}). Plotting all found data.")
        allowed_methods = None

    agent_runs = defaultdict(dict)
    for base in search_dirs:
        if not base.exists(): continue
        for p in base.rglob("metrics.csv"):
            run_folder = p.parent
            config_folder = run_folder.parent if run_folder.name.startswith("version_") else run_folder
            exp_id_from_path = config_folder.parent.name
            agent_folder_name = config_folder.name
            
            # Apply filters
            if any(ef == exp_id_from_path for ef in experiment_filters):
                if allowed_methods is not None and agent_folder_name not in allowed_methods:
                    continue
                
                version = -1
                if run_folder.name.startswith("version_"):
                    try:
                        version = int(run_folder.name.split("_")[1])
                    except (ValueError, IndexError):
                        pass
                
                # Extract dataset from hparams.yaml to differentiate offline runs
                dataset_key = ""
                hparams_path = run_folder / "hparams.yaml"
                if not hparams_path.exists():
                    hparams_path = config_folder / "hparams.yaml"
                    
                if hparams_path.exists():
                    try:
                        with open(hparams_path, "r") as f:
                            # Simple parsing to avoid full yaml load overhead
                            for line in f:
                                if "dataset_path:" in line and "results/datasets/" in line:
                                    dataset_key = line.strip().split()[-1]
                                    break
                    except Exception:
                        pass
                
                agent_key = (exp_id_from_path, str(config_folder), dataset_key)
                agent_runs[agent_key][version] = run_folder

    all_runs = []
    for agent_key, versions in agent_runs.items():
        if not versions: continue
        
        if args.version == "all":
            # Add ALL versions for this agent for aggregation
            for v_num in versions:
                all_runs.append(load_run_data(versions[v_num], args))
        elif args.version is not None:
            # Add only the SPECIFIC version
            try:
                v_requested = int(args.version)
                if v_requested in versions:
                    all_runs.append(load_run_data(versions[v_requested], args))
                elif v_requested == 0 and -1 in versions:
                    all_runs.append(load_run_data(versions[-1], args))
            except ValueError:
                print(f"Warning: Invalid version '{args.version}' requested for {agent_key[1]}")
        else:
            # Default: Add only the LATEST version
            latest_v = max(versions.keys())
            all_runs.append(load_run_data(versions[latest_v], args))

    if not all_runs:
        if args.version is not None:
            print(f"No data found for experiment '{args.experiment}' with version {args.version}")
        else:
            print(f"No data found for experiment: {args.experiment}")
        return
    
    found_exp_ids = set(r["exp_id"] for r in all_runs)
    use_simple_labels = len(found_exp_ids) <= 1

    save_dir = Path("results/plots") / args.experiment
    if save_dir.exists():
        import shutil
        print(f"Clearing existing plots in {save_dir}")
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load style config if provided, else use defaults
    plot_config = None
    style_to_load = args.style or "default"
    
    style_path = Path(style_to_load)
    # Check standard locations
    potential_paths = [
        style_path,
        Path("conf/plot_styles") / style_path,
        Path("conf/plot_styles") / f"{style_to_load}.yaml"
    ]
    
    actual_path = None
    for p in potential_paths:
        if p.exists() and p.is_file():
            actual_path = p
            break
    
    if actual_path:
        with open(actual_path, "r") as f:
            plot_config = yaml.safe_load(f)
    elif args.style:
        print(f"Warning: Style config {args.style} not found. Using hardcoded defaults.")
    
    if not plot_config:
        plot_config = {
            "plots": [
                {"metric": "eval/reward", "title": f"Evaluation Reward: {args.experiment}", "ylabel": "Avg Reward", "filename": "eval_reward.png"},
                {"metric": "train/reward", "title": f"Training Reward: {args.experiment}", "ylabel": "Reward", "filename": "train_reward.png", "window": 20},
                {"metric": "train/length", "title": f"Episode Length: {args.experiment}", "ylabel": "Steps", "filename": "train_length.png", "window": 20},
                {"metric": "losses/actor_loss", "title": f"Actor Loss: {args.experiment}", "ylabel": "Loss", "filename": "offline_actor_loss.png", "window": 20, "x_axis": "step", "xlabel": "Training Steps"},
                {"metric": "losses/q_loss", "title": f"Q Loss: {args.experiment}", "ylabel": "Loss", "filename": "offline_q_loss.png", "window": 20, "x_axis": "step", "xlabel": "Training Steps"},
                {"metric": "losses/value_loss", "title": f"Value Loss: {args.experiment}", "ylabel": "Loss", "filename": "offline_value_loss.png", "window": 20, "x_axis": "step", "xlabel": "Training Steps"},
            ]
        }

    for p_def in plot_config.get("plots", []):
        metric = p_def.get("metric")
        title_tmpl = p_def.get("title", f"{metric}: {args.experiment}")
        ylabel = p_def.get("ylabel", metric)
        filename_tmpl = p_def.get("filename", f"{metric.replace('/', '_')}.png")
        window = p_def.get("window", 1)
        x_axis = p_def.get("x_axis", "transitions")
        xlabel = p_def.get("xlabel", "Transitions (Dataset Size)" if x_axis == "transitions" else "Steps")
        split_by = p_def.get("split_by")

        if split_by:
            # Group all_runs by the split_by attribute
            split_groups = defaultdict(list)
            for r in all_runs:
                val = r.get(split_by, "unknown")
                split_groups[val].append(r)
            
            for split_val, runs_in_split in split_groups.items():
                # For each split, create the (exp_id, label) groups create_plot expects
                current_groups = defaultdict(list)
                for r in runs_in_split:
                    current_groups[(r["exp_id"], r["label"])].append(r)
                
                # Format title and filename
                title = title_tmpl.replace("{split_value}", str(split_val)).replace("{experiment}", args.experiment)
                filename = filename_tmpl.replace("{split_value}", str(split_val)).replace("{experiment}", args.experiment)
                
                create_plot(current_groups, metric, title, ylabel, save_dir / filename, 
                           window=window, use_simple_labels=use_simple_labels, 
                           x_axis_col=x_axis, xlabel=xlabel)
        else:
            # Standard combined plot
            current_groups = defaultdict(list)
            for r in all_runs:
                current_groups[(r["exp_id"], r["label"])].append(r)
            
            title = title_tmpl.replace("{experiment}", args.experiment)
            filename = filename_tmpl.replace("{experiment}", args.experiment)
            
            create_plot(current_groups, metric, title, ylabel, save_dir / filename, 
                       window=window, use_simple_labels=use_simple_labels, 
                       x_axis_col=x_axis, xlabel=xlabel)

    generate_time_table(all_runs, save_dir)

def generate_time_table(all_runs, save_dir):
    table_data = []
    for run in all_runs:
        label = run["label"]
        source = run["source"]
        data = run["data"]
        
        # Check if we have time metrics
        if "time/train" in data and "time/total" in data:
            # Final cumulative values
            total_time = data["time/total"]["values"][-1]
            train_time = data["time/train"]["values"][-1]
            eval_time = total_time - train_time
            
            table_data.append({
                "Method": label,
                "Source": source,
                "Total (m)": total_time / 60,
                "Train (m)": train_time / 60,
                "Eval (m)": eval_time / 60,
                "Train %": (train_time / total_time * 100) if total_time > 0 else 0,
                "Eval %": (eval_time / total_time * 100) if total_time > 0 else 0,
                "Total (s)": total_time,
                "Train (s)": train_time,
                "Eval (s)": eval_time,
            })
    
    if table_data:
        df = pd.DataFrame(table_data)
        # Round numerical columns for better display
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].round(2)
        
        df.to_csv(save_dir / "timing_report.csv", index=False)
        
        print("\nTiming Report:")
        print(df.to_string(index=False))
        
        try:
            with open(save_dir / "timing_report.md", "w") as f:
                f.write(df.to_markdown(index=False))
        except Exception as e:
            print(f"Could not save markdown table: {e}")

if __name__ == "__main__":
    main()
