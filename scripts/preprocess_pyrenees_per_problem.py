"""
preprocess_pyrenees_per_problem.py — Multi-Problem Preprocessing & Competency Clustering for Pyrenees ITS.

Processes each of the 11 Pyrenees datasets individually:
  1. problem.csv: Problem-level policy (130 features, 3 discrete actions: 0=PS, 1=WE, 2=FWE)
  2. 10 exercise datasets (ex132(w), etc.): Step-level policy (123 features, 2 discrete actions: 0=PS/Elicit, 1=WE/Tell)

For each problem type, this script produces:
  - in/datasets/pyrenees/per_problem/{problem_id}/cql/*.pkl (chunked dataset for DatasetReader)
  - in/datasets/pyrenees/per_problem/{problem_id}/clean.npz (trajectories for eval)
  - in/datasets/pyrenees/per_problem/{problem_id}/scaler.npz (feature normalization)
  - in/datasets/pyrenees/per_problem/{problem_id}/gmm_scaler.npz (calibrated 3-tier GMM competency parameters)
  - results/plots/pyrenees/clusters/{problem_id}_clusters.png (cluster visualization)

Usage:
  python scripts/preprocess_pyrenees_per_problem.py [--problem PROBLEM_ID]
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.dataset_utils import DatasetWriter

META_COLS = [
    "feature_recordID", "answerID", "time", "userID", "problem",
    "decisionID", "decisionPoint", "decisionOrdering", "substepMode",
    "KC", "session", "substepOrdering", "action", "reward",
]

CLUSTER_FEATS = [
    "pctCorrect",
    "pctCorrectKC",
    "pctCorrectSession",
    "nStepSinceLastWrong",
    "nTotalHintSession",
    "avgTimeOnStep",
]


def fit_gmm_for_problem(norm_states, feat_cols, p_low=35, p_high=92, sample_size=100000):
    """
    Fit calibrated 3-tier Gaussian competency model on multi-feature space.
    Dynamically finds cluster feature indices in feat_cols.
    """
    feat_cols = list(feat_cols)
    feat_indices = [feat_cols.index(f) for f in CLUSTER_FEATS if f in feat_cols]
    used_names = [feat_cols[i] for i in feat_indices]

    rng = np.random.default_rng(42)
    n_samples = len(norm_states)
    idx = rng.choice(n_samples, size=min(sample_size, n_samples), replace=False)
    X_feat = norm_states[idx][:, feat_indices]

    # Composite competency metric: accuracy + error recovery - hint reliance - step time
    # Weights aligned with feature positions
    w_acc = 1.0
    w_rec = 0.5
    w_hint = -0.5
    w_time = -0.3

    score_parts = []
    for i, name in enumerate(used_names):
        if "pctCorrect" in name:
            score_parts.append(w_acc * X_feat[:, i])
        elif "nStepSinceLastWrong" in name:
            score_parts.append(w_rec * X_feat[:, i])
        elif "nTotalHint" in name or "Hint" in name:
            score_parts.append(w_hint * X_feat[:, i])
        elif "avgTimeOnStep" in name or "Time" in name:
            score_parts.append(w_time * X_feat[:, i])
        else:
            score_parts.append(0.5 * X_feat[:, i])

    scores = np.sum(score_parts, axis=0)
    p_l = np.percentile(scores, p_low)
    p_h = np.percentile(scores, p_high)

    labels = np.zeros(len(scores), dtype=int)
    labels[scores > p_l] = 1
    labels[scores > p_h] = 2

    n_feats = len(feat_indices)
    means = np.zeros((3, n_feats))
    covariances = np.zeros((3, n_feats, n_feats))
    weights = np.zeros(3)

    for k in range(3):
        X_k = X_feat[labels == k]
        if len(X_k) == 0:
            means[k] = np.zeros(n_feats)
            covariances[k] = np.eye(n_feats)
            weights[k] = 1.0 / 3.0
        else:
            means[k] = X_k.mean(axis=0)
            covariances[k] = np.cov(X_k, rowvar=False) + 1e-4 * np.eye(n_feats)
            weights[k] = len(X_k) / len(X_feat)

    precisions = np.array([np.linalg.inv(c) for c in covariances])
    log_dets = np.array([np.linalg.slogdet(c)[1] for c in covariances])
    log_weights = np.log(weights + 1e-12)

    return {
        "means": means,
        "covariances": covariances,
        "precisions": precisions,
        "log_dets": log_dets,
        "log_weights": log_weights,
        "weights": weights,
        "feature_indices": np.array(feat_indices),
        "feature_names": np.array(used_names),
        "X_sub": X_feat,
        "labels": labels,
    }


def compute_gmm_posteriors_np(X_feat, means, precisions, log_dets, log_weights):
    d = X_feat.shape[-1]
    const = 0.5 * d * np.log(2.0 * np.pi)
    log_probs = []
    for k in range(len(means)):
        diff = X_feat - means[k]
        maha = np.sum(diff * (diff @ precisions[k]), axis=-1)
        log_p = log_weights[k] - 0.5 * log_dets[k] - 0.5 * maha - const
        log_probs.append(log_p)
    log_probs = np.stack(log_probs, axis=-1)
    max_log = np.max(log_probs, axis=-1, keepdims=True)
    exp_p = np.exp(log_probs - max_log)
    return exp_p / np.sum(exp_p, axis=-1, keepdims=True)


def plot_competency_clusters(gmm_res, problem_id, out_plot_path):
    Path(out_plot_path).parent.mkdir(parents=True, exist_ok=True)
    X_sub = gmm_res["X_sub"]
    means = gmm_res["means"]
    precisions = gmm_res["precisions"]
    log_dets = gmm_res["log_dets"]
    log_weights = gmm_res["log_weights"]
    weights = gmm_res["weights"]

    posteriors = compute_gmm_posteriors_np(X_sub, means, precisions, log_dets, log_weights)
    hard_labels = posteriors.argmax(axis=1)

    fig = plt.figure(figsize=(13, 5), facecolor="#fafafa")
    gs = GridSpec(1, 2, width_ratios=[1.6, 1], figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#f5f5f5")

    colors = ["#e74c3c", "#f39c12", "#27ae60"]
    tier_labels = ["Low Competency", "Medium Competency", "High Competency (Mastery)"]

    n_plot = min(15000, len(X_sub))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_sub), n_plot, replace=False)

    for tier in [0, 1, 2]:
        m = hard_labels[idx] == tier
        ax1.scatter(
            X_sub[idx][m, 0], X_sub[idx][m, 1],
            c=colors[tier], label=tier_labels[tier],
            alpha=0.35, s=10, rasterized=True
        )

    # Plot cluster centroids
    for tier in [0, 1, 2]:
        ax1.scatter(
            means[tier, 0], means[tier, 1],
            c=colors[tier], s=260, marker="*",
            edgecolors="black", linewidths=1.5, zorder=5
        )

    ax1.set_xlabel("pctCorrect (z-score)", fontsize=11)
    ax1.set_ylabel("pctCorrectKC (z-score)", fontsize=11)
    ax1.set_title(f"Competency Clusters: {problem_id}\n(GMM Soft Posterior Distribution)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9.5, framealpha=0.9)
    ax1.spines[["top", "right"]].set_visible(False)

    # Proportion Bar Chart
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#f5f5f5")
    counts = np.bincount(hard_labels, minlength=3)
    total = len(hard_labels)
    bars = ax2.bar(range(3), counts, color=colors, edgecolor="white", width=0.55)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["Low", "Medium", "High"], fontsize=11)
    ax2.set_ylabel("Number of Steps", fontsize=11)
    ax2.set_title(f"Student Tier Distribution\n(Total: {total:,})", fontsize=12, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)

    for bar, n in zip(bars, counts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{n:,}\n({100*n/total:.1f}%)",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold"
        )

    plt.suptitle(f"Pyrenees Competency Clustering — {problem_id}", fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(out_plot_path, dpi=150, bbox_inches="tight")
    plt.close()


def process_single_csv(csv_path: Path, output_base_dir: Path, plot_base_dir: Path):
    problem_id = csv_path.stem  # e.g., 'problem' or 'ex132(w)'
    is_problem_level = (problem_id == "problem")
    expected_actions = 3 if is_problem_level else 2

    print("\n" + "=" * 70)
    print(f"  PROCESSING: {problem_id} ({'Problem-Level Policy' if is_problem_level else 'Step-Level Policy'})")
    print(f"  Source:     {csv_path}")
    print("=" * 70)

    out_prob_dir = output_base_dir / "per_problem" / problem_id
    out_cql_dir = out_prob_dir / "cql"
    out_cql_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"  Rows loaded: {len(df):,}")

    feat_cols = [c for c in df.columns if c not in META_COLS]
    print(f"  Feature dimensions: {len(feat_cols)}")

    # 1. Fit feature scaler
    feat_matrix = df[feat_cols].values.astype(np.float32)
    mean = np.mean(feat_matrix, axis=0)
    std = np.std(feat_matrix, axis=0)
    std[std == 0.0] = 1.0
    data_min = np.min(feat_matrix, axis=0)
    data_max = np.max(feat_matrix, axis=0)

    norm_feats = (feat_matrix - mean) / std
    actions = df["action"].values.astype(np.int64)
    rewards = df["reward"].values.astype(np.float32)

    # Validate action space
    unique_actions = np.unique(actions)
    print(f"  Actions found: {unique_actions.tolist()} (Expected {expected_actions} actions)")

    # 2. Build trajectories
    writer = DatasetWriter(save_dir=out_cql_dir, chunk_size=100_000, env_name="pyrenees")

    npz_states = []
    npz_actions = []
    npz_rewards = []
    npz_dones = []

    # Group by student session / problem trajectory
    group_cols = ["userID", "problem"] if "problem" in df.columns else ["userID"]
    grouped = df.groupby(group_cols, sort=False).indices

    total_transitions = 0
    total_trajectories = 0
    action_counts = {}

    for group_key, indices in grouped.items():
        if len(indices) == 0:
            continue

        traj_states = norm_feats[indices]
        traj_actions = actions[indices]
        traj_rewards = rewards[indices]
        traj_len = len(indices)

        traj_dones = np.zeros(traj_len, dtype=np.float32)
        traj_dones[-1] = 1.0

        traj_next_states = np.empty_like(traj_states)
        traj_next_states[:-1] = traj_states[1:]
        traj_next_states[-1] = traj_states[-1]

        for a in traj_actions:
            action_counts[int(a)] = action_counts.get(int(a), 0) + 1

        writer.batch_add(
            obs=traj_states,
            logic_obs=None,
            action=traj_actions,
            reward=traj_rewards,
            next_obs=traj_next_states,
            next_logic_obs=None,
            done=traj_dones,
        )

        total_transitions += traj_len
        npz_states.append(traj_states)
        npz_actions.append(traj_actions)
        npz_rewards.append(traj_rewards)
        npz_dones.append(traj_dones)
        total_trajectories += 1

    writer.close()
    print(f"  DatasetWriter chunks saved -> {out_cql_dir}")
    print(f"  Trajectories: {total_trajectories:,}, Transitions: {total_transitions:,}")
    print(f"  Action Distribution: {action_counts}")

    # 3. Fit dedicated GMM competency model
    print(f"  Fitting dedicated 3-tier GMM competency model for {problem_id}...")
    all_norm_steps = np.vstack(npz_states)
    gmm_res = fit_gmm_for_problem(all_norm_steps, feat_cols)

    # 4. Save GMM parameters
    gmm_out_path = out_prob_dir / "gmm_scaler.npz"
    np.savez(
        gmm_out_path,
        means=gmm_res["means"],
        covariances=gmm_res["covariances"],
        precisions=gmm_res["precisions"],
        log_dets=gmm_res["log_dets"],
        log_weights=gmm_res["log_weights"],
        cluster_weights=gmm_res["weights"],
        feature_indices=gmm_res["feature_indices"],
        feature_names=gmm_res["feature_names"],
    )
    print(f"  GMM competency parameters saved -> {gmm_out_path}")

    # 5. Save standard scaler
    scaler_out_path = out_prob_dir / "scaler.npz"
    np.savez(
        scaler_out_path,
        mean=mean,
        std=std,
        data_min=data_min,
        data_max=data_max,
        feat_cols=np.array(feat_cols),
        n_features=len(feat_cols),
        n_actions=expected_actions,
        problem_id=problem_id,
        is_problem_level=is_problem_level,
    )
    print(f"  Scaler parameters saved -> {scaler_out_path}")

    # 6. Save clean NPZ for evaluation
    clean_npz_path = out_prob_dir / "clean.npz"
    np.savez_compressed(
        clean_npz_path,
        states=np.array(npz_states, dtype=object),
        actions=np.array(npz_actions, dtype=object),
        rewards=np.array(npz_rewards, dtype=object),
        dones=np.array(npz_dones, dtype=object),
        n_features=len(feat_cols),
        n_actions=expected_actions,
    )
    print(f"  Clean trajectories saved -> {clean_npz_path}")

    # 7. Generate cluster diagnostic plot
    plot_out_path = plot_base_dir / "clusters" / f"{problem_id}_clusters.png"
    plot_competency_clusters(gmm_res, problem_id, plot_out_path)
    print(f"  Cluster diagnostic plot saved -> {plot_out_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Pyrenees per-problem datasets & clusters.")
    parser.add_argument("--problem", type=str, default=None, help="Specific problem ID to preprocess (e.g. 'problem', 'ex132(w)'). Default: all 11.")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "in" / "datasets" / "pyrenees" / "Pyrenees data clean"
    out_base_dir = PROJECT_ROOT / "in" / "datasets" / "pyrenees"
    plot_base_dir = PROJECT_ROOT / "results" / "plots" / "pyrenees"

    csv_files = sorted(glob.glob(str(data_dir / "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    if args.problem:
        matched = [f for f in csv_files if Path(f).stem == args.problem or Path(f).name == args.problem]
        if not matched:
            raise ValueError(f"Could not find CSV for problem '{args.problem}' in {data_dir}")
        csv_files = matched

    print(f"Starting per-problem preprocessing for {len(csv_files)} problem datasets...")
    for csv_file in csv_files:
        process_single_csv(Path(csv_file), out_base_dir, plot_base_dir)

    print("\n" + "=" * 70)
    print("  ALL PYRENEES PROBLEM DATASETS & GMM CLUSTERS PREPROCESSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
