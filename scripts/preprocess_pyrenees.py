"""
preprocess_pyrenees.py — Preprocess raw Pyrenees CSV data for BlendRL.

Changes from original:
  - Preserves all 3 raw actions (PS=0, WE=1, FWE=2).
  - Adds cluster visualization (results/plots/pyrenees/competency_clusters.png).
  - Saves KMeans cluster thresholds alongside the scaler.

Actions:
  0 = PS  (Problem Solving)
  1 = WE  (Worked Example)
  2 = FWE (Faded Worked Example)

Run:
    python scripts/preprocess_pyrenees.py
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from src.dataset_utils import DatasetWriter

# ─── Performance feature indices for clustering ───────────────────────────────
# After standardisation, these z-scored columns are used for KMeans(k=3).
CLUSTER_FEATS = ["pctCorrect", "pctCorrectKC"]

# ─── KMeans thresholds (overwritten at runtime; also saved to disk) ───────────
# Low  | Med  boundary: pctCorrect z-score ≈ -1.74
# Med  | High boundary: pctCorrect z-score ≈ +0.02
LOW_MED_PCT_CORRECT  = -1.74
MED_HIGH_PCT_CORRECT =  0.02


def run_clustering(all_states_by_traj, feat_cols, mean, std):
    """
    Cluster students into 3 competency tiers using per-student mean performance.

    Returns
    -------
    thresh_low_med  : np.ndarray  shape (2,) in z-score space
    thresh_med_high : np.ndarray  shape (2,) in z-score space
    student_labels  : np.ndarray  shape (N,)  0=low, 1=med, 2=high
    student_profiles: np.ndarray  shape (N, 2)
    """
    feat_cols = list(feat_cols)
    ci = [feat_cols.index(f) for f in CLUSTER_FEATS]

    # Per-student mean over all steps
    profiles = np.array([traj[:, ci].mean(axis=0) for traj in all_states_by_traj])

    # Filter out degenerate outliers (>3σ on pctCorrect)
    pc_mean, pc_std = profiles[:, 0].mean(), profiles[:, 0].std()
    mask = np.abs(profiles[:, 0] - pc_mean) <= 3 * pc_std
    print(f"  Clustering on {mask.sum()} / {len(profiles)} students "
          f"(excluded {(~mask).sum()} outliers).")

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    labels_filtered = kmeans.fit_predict(profiles[mask])
    centers = kmeans.cluster_centers_

    # Sort clusters low → high by summed center
    sorted_idx = np.argsort(centers.sum(axis=1))
    label_map = {orig: new for new, orig in enumerate(sorted_idx)}
    labels_filtered_remap = np.array([label_map[l] for l in labels_filtered])
    sorted_centers = centers[sorted_idx]

    # Assign labels to all trajectories (outliers default to nearest center)
    all_labels = np.zeros(len(profiles), dtype=np.int32)
    full_labels = kmeans.predict(profiles)
    full_labels_remap = np.array([label_map[l] for l in full_labels])
    all_labels = full_labels_remap

    # Midpoint thresholds
    thresh_low_med  = (sorted_centers[0] + sorted_centers[1]) / 2
    thresh_med_high = (sorted_centers[1] + sorted_centers[2]) / 2

    print(f"  Low  center : pctCorrect={sorted_centers[0,0]:.3f}, pctCorrectKC={sorted_centers[0,1]:.3f}")
    print(f"  Med  center : pctCorrect={sorted_centers[1,0]:.3f}, pctCorrectKC={sorted_centers[1,1]:.3f}")
    print(f"  High center : pctCorrect={sorted_centers[2,0]:.3f}, pctCorrectKC={sorted_centers[2,1]:.3f}")
    print(f"  Threshold low|med  : pctCorrect={thresh_low_med[0]:.4f}, pctCorrectKC={thresh_low_med[1]:.4f}")
    print(f"  Threshold med|high : pctCorrect={thresh_med_high[0]:.4f}, pctCorrectKC={thresh_med_high[1]:.4f}")
    n = np.bincount(all_labels)
    print(f"  Cluster sizes — Low:{n[0]:,}  Med:{n[1]:,}  High:{n[2]:,}")

    return thresh_low_med, thresh_med_high, all_labels, profiles


def save_cluster_plot(profiles, labels, thresh_low_med, thresh_med_high, out_path):
    """Save a publication-quality cluster visualization."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("  [warn] matplotlib not available; skipping cluster plot.")
        return

    rng = np.random.default_rng(42)
    n_plot = min(15000, len(profiles))
    idx = rng.choice(len(profiles), n_plot, replace=False)
    X_plot = profiles[idx]
    y_plot = labels[idx]

    colors      = ["#e74c3c", "#f39c12", "#27ae60"]
    tier_labels = ["Low Competency", "Medium Competency", "High Competency"]
    markers     = ["o", "s", "^"]

    fig = plt.figure(figsize=(13, 6), facecolor="#fafafa")
    gs  = GridSpec(1, 2, width_ratios=[1.65, 1], figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax1.set_facecolor("#f5f5f5")
    ax2.set_facecolor("#f5f5f5")

    # ── Scatter ───────────────────────────────────────────────────────────────
    for tier in [0, 1, 2]:
        m = y_plot == tier
        ax1.scatter(
            X_plot[m, 0], X_plot[m, 1],
            c=colors[tier], label=tier_labels[tier],
            alpha=0.30, s=9, marker=markers[tier],
            linewidths=0, rasterized=True,
        )

    # Cluster centers (stars)
    n_all = np.bincount(labels)
    sorted_centers = np.array([
        profiles[labels == t].mean(axis=0) for t in [0, 1, 2]
    ])
    for tier in [0, 1, 2]:
        ax1.scatter(
            sorted_centers[tier, 0], sorted_centers[tier, 1],
            c=colors[tier], s=260, marker="*",
            edgecolors="black", linewidths=1.5, zorder=5,
        )

    # Decision boundaries
    y_lo, y_hi = ax1.get_ylim() or (-1, 1)
    for thresh, lbl in [
        (thresh_low_med[0],  f"Low|Med\n({thresh_low_med[0]:.2f})"),
        (thresh_med_high[0], f"Med|High\n({thresh_med_high[0]:.2f})"),
    ]:
        ax1.axvline(thresh, color="#666", linestyle="--", linewidth=1.3, alpha=0.85, zorder=2)

    ax1.set_xlabel("pctCorrect (z-score)", fontsize=12, labelpad=6)
    ax1.set_ylabel("pctCorrectKC (z-score)", fontsize=12, labelpad=6)
    ax1.set_title("Student Competency Clusters\n(KMeans, k=3)", fontsize=13, fontweight="bold", pad=10)
    ax1.legend(fontsize=10, framealpha=0.9, markerscale=1.6)
    ax1.spines[["top", "right"]].set_visible(False)

    # Annotate boundaries inside the plot
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    for thresh, lbl in [
        (thresh_low_med[0],  f"Low|Med\n({thresh_low_med[0]:.2f})"),
        (thresh_med_high[0], f"Med|High\n({thresh_med_high[0]:.2f})"),
    ]:
        ax1.text(thresh + 0.08, ylim[1] - 0.05 * (ylim[1] - ylim[0]),
                 lbl, fontsize=8.5, color="#444", va="top", ha="left")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    bars = ax2.bar(range(3), n_all, color=colors, edgecolor="white", width=0.55)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["Low", "Medium", "High"], fontsize=11)
    ax2.set_ylabel("Number of Students", fontsize=11, labelpad=6)
    ax2.set_title("Students per Tier", fontsize=12, fontweight="bold", pad=10)
    ax2.spines[["top", "right"]].set_visible(False)

    total = n_all.sum()
    for bar, n in zip(bars, n_all):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{n:,}\n({100*n/total:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    fig.suptitle(
        "Pyrenees Student Competency Clustering",
        fontsize=15, fontweight="bold", y=1.02,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Cluster plot saved → {out_path}")


def preprocess_pyrenees():
    data_dir    = Path("in/datasets/pyrenees/Pyrenees data clean")
    out_cql_dir = Path("in/datasets/pyrenees/cql")
    out_cql_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(
        [f for f in glob.glob(str(data_dir / "*.csv")) if "problem.csv" not in f]
    )
    print(f"Found {len(csv_files)} exercise CSV files.")

    meta_cols = [
        "feature_recordID", "answerID", "time", "userID", "problem",
        "decisionID", "decisionPoint", "decisionOrdering", "substepMode",
        "KC", "session", "substepOrdering", "action", "reward",
    ]

    df_sample = pd.read_csv(csv_files[0], nrows=5)
    feat_cols = [c for c in df_sample.columns if c not in meta_cols]
    print(f"Identified {len(feat_cols)} feature columns.")

    # ── Pass 1: fit scaler ─────────────────────────────────────────────────────
    print("Pass 1: Reading CSV files for scaler fit…")
    all_dfs = [pd.read_csv(f) for f in csv_files]
    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows: {len(full_df):,}")

    feat_matrix = full_df[feat_cols].values.astype(np.float32)
    mean = np.mean(feat_matrix, axis=0)
    std  = np.std(feat_matrix,  axis=0)
    std[std == 0.0] = 1.0

    # ── Pass 2: build trajectories ────────────────────────────────────────────
    print("Pass 2: Building trajectories…")
    writer = DatasetWriter(save_dir=out_cql_dir, chunk_size=100_000, env_name="pyrenees")

    npz_states  = []
    npz_actions = []
    npz_rewards = []
    npz_dones   = []

    total_transitions   = 0
    total_trajectories  = 0
    action_counts       = {0: 0, 1: 0, 2: 0}

    for df in all_dfs:
        norm_feats = (df[feat_cols].values.astype(np.float32) - mean) / std
        df_actions = df["action"].values.astype(np.int64)
        df_rewards = df["reward"].values.astype(np.float32)

        grouped = df.groupby(["userID", "problem"], sort=False).indices

        for (user, prob), indices in grouped.items():
            if len(indices) == 0:
                continue

            traj_states  = norm_feats[indices]
            traj_actions = df_actions[indices]
            traj_rewards = df_rewards[indices]
            traj_len     = len(indices)

            traj_dones = np.zeros(traj_len, dtype=np.float32)
            traj_dones[-1] = 1.0

            traj_next_states = np.empty_like(traj_states)
            traj_next_states[:-1] = traj_states[1:]
            traj_next_states[-1]  = traj_states[-1]

            # Accumulate action counts
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
            total_transitions  += traj_len

            npz_states.append(traj_states)
            npz_actions.append(traj_actions)
            npz_rewards.append(traj_rewards)
            npz_dones.append(traj_dones)
            total_trajectories += 1

    writer.close()
    print(f"  DatasetWriter: {total_transitions:,} transitions → {out_cql_dir}")
    total_a = sum(action_counts.values())
    print(f"  Action distribution:")
    for a, name in [(0, "PS"), (1, "WE"), (2, "FWE")]:
        print(f"    {name}(={a}): {action_counts.get(a,0):,} ({100*action_counts.get(a,0)/total_a:.1f}%)")

    # ── KMeans clustering ──────────────────────────────────────────────────────
    print("Running competency clustering (KMeans k=3)…")
    thresh_low_med, thresh_med_high, cluster_labels, profiles = run_clustering(
        npz_states, feat_cols, mean, std
    )

    # ── Save cluster plot ──────────────────────────────────────────────────────
    print("Generating cluster visualization…")
    save_cluster_plot(
        profiles, cluster_labels, thresh_low_med, thresh_med_high,
        out_path="results/plots/pyrenees/competency_clusters.png",
    )

    # ── Save scaler + cluster info ─────────────────────────────────────────────
    np.savez(
        "in/datasets/pyrenees/pyrenees_scaler.npz",
        mean=mean, std=std, feat_cols=np.array(feat_cols),
        thresh_low_med=thresh_low_med,
        thresh_med_high=thresh_med_high,
        cluster_feat_names=np.array(CLUSTER_FEATS),
    )
    print("Scaler + cluster thresholds saved → in/datasets/pyrenees/pyrenees_scaler.npz")

    # ── Save compressed NPZ for evaluation ────────────────────────────────────
    npz_path = Path("in/datasets/pyrenees/pyrenees_clean.npz")
    np.savez_compressed(
        npz_path,
        states=np.array(npz_states,  dtype=object),
        actions=np.array(npz_actions, dtype=object),
        rewards=np.array(npz_rewards, dtype=object),
        dones=np.array(npz_dones,    dtype=object),
    )
    print(f"  Eval dataset ({total_trajectories:,} trajectories) → {npz_path}")
    print("Preprocessing complete!")


if __name__ == "__main__":
    preprocess_pyrenees()
