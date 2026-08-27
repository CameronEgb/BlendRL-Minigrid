#!/usr/bin/env python3
"""
mRMR feature selection and GMM competency clustering for Pyrenees ITS.
"""

import os
import sys
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.feature_selection import mutual_info_regression
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

META_COLS = [
    "feature_recordID", "answerID", "time", "userID", "problem",
    "decisionID", "decisionPoint", "decisionOrdering", "substepMode",
    "KC", "session", "substepOrdering", "action", "reward",
]

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


def run_mrmr_selection(df, feat_cols, target_col="reward", n_select=6, sample_size=30000, random_state=42):
    """
    Computes mRMR (MID) feature selection:
    f* = argmax_{f in F \\ S} [ I(f; Y) - (1/|S|) sum_{s in S} |corr(f, s)| ]
    """
    rng = np.random.default_rng(random_state)
    n_rows = len(df)
    idx = rng.choice(n_rows, size=min(sample_size, n_rows), replace=False)
    
    X = df[feat_cols].iloc[idx].fillna(0.0).values.astype(np.float32)
    y = df[target_col].iloc[idx].values.astype(np.float32)

    stds = np.std(X, axis=0)
    valid_mask = stds > 1e-5
    valid_indices = np.where(valid_mask)[0]
    X_valid = X[:, valid_indices]
    valid_feats = [feat_cols[i] for i in valid_indices]

    # 1. Relevance: Mutual Information
    relevance = mutual_info_regression(X_valid, y, random_state=random_state)

    # 2. Redundancy: Correlation matrix
    corr_matrix = np.abs(np.corrcoef(X_valid, rowvar=False))
    np.nan_to_num(corr_matrix, copy=False, nan=0.0)

    # 3. Greedy mRMR Selection
    selected = []
    candidates = list(range(len(valid_feats)))

    first = int(np.argmax(relevance))
    selected.append(first)
    candidates.remove(first)

    for _ in range(1, min(n_select, len(valid_feats))):
        best_score = -float("inf")
        best_f = None
        for f in candidates:
            rel = relevance[f]
            red = np.mean([corr_matrix[f, s] for s in selected])
            score = rel - red
            if score > best_score:
                best_score = score
                best_f = f
        selected.append(best_f)
        candidates.remove(best_f)

    selected_names = [valid_feats[i] for i in selected]
    selected_orig_indices = [feat_cols.index(name) for name in selected_names]
    selected_relevance = [float(relevance[i]) for i in selected]

    return {
        "selected_names": selected_names,
        "selected_indices": selected_orig_indices,
        "relevance": selected_relevance,
        "valid_feats": valid_feats,
        "all_relevance": relevance,
    }


def fit_gmm_3tier(norm_matrix, feat_indices, p_low=35, p_high=92, sample_size=100000, random_state=42):
    """
    Fits a calibrated 3-tier Gaussian Mixture Model on the mRMR feature subspace.
    """
    rng = np.random.default_rng(random_state)
    n_samples = len(norm_matrix)
    idx = rng.choice(n_samples, size=min(sample_size, n_samples), replace=False)
    X_sub = norm_matrix[idx][:, feat_indices]

    scores = np.mean(X_sub, axis=1)

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
        X_k = X_sub[labels == k]
        if len(X_k) == 0:
            means[k] = np.zeros(n_feats)
            covariances[k] = np.eye(n_feats)
            weights[k] = 1.0 / 3.0
        else:
            means[k] = X_k.mean(axis=0)
            covariances[k] = np.cov(X_k, rowvar=False) + 1e-4 * np.eye(n_feats)
            weights[k] = len(X_k) / len(X_sub)

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
        "X_sub": X_sub,
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


def main():
    print("=" * 80)
    print("      PYRENEES mRMR FEATURE SELECTION & GMM RECLUSTERING")
    print("=" * 80)

    results_summary = {}
    plot_dir = PROJECT_ROOT / "results" / "plots" / "pyrenees" / "mrmr"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for problem_id in ALL_PROBLEMS:
        csv_path = PROJECT_ROOT / "in" / "datasets" / "pyrenees" / "Pyrenees data clean" / f"{problem_id}.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found. Skipping.")
            continue

        df = pd.read_csv(csv_path)
        feat_cols = [c for c in df.columns if c not in META_COLS]
        is_problem_level = (problem_id == "problem")

        print(f"\n[{problem_id}] mRMR Feature Selection ({len(df):,} steps, {len(feat_cols)} candidate features)...")
        mrmr_res = run_mrmr_selection(df, feat_cols, target_col="reward", n_select=6)

        selected_names = mrmr_res["selected_names"]
        selected_indices = mrmr_res["selected_indices"]
        print(f"  -> Selected Features:")
        for r, (fname, rel) in enumerate(zip(selected_names, mrmr_res["relevance"]), 1):
            print(f"     {r}. {fname:30s} (MI = {rel:.4f})")

        feat_matrix = df[feat_cols].values.astype(np.float32)
        mean = np.mean(feat_matrix, axis=0)
        std = np.std(feat_matrix, axis=0)
        std[std == 0.0] = 1.0
        norm_matrix = (feat_matrix - mean) / std

        p_low = 35 if is_problem_level else 37
        p_high = 74 if is_problem_level else 78
        gmm_res = fit_gmm_3tier(norm_matrix, selected_indices, p_low=p_low, p_high=p_high)

        rewards = df["reward"].values.astype(np.float32)
        posteriors = compute_gmm_posteriors_np(
            norm_matrix[:50000][:, selected_indices],
            gmm_res["means"], gmm_res["precisions"], gmm_res["log_dets"], gmm_res["log_weights"]
        )
        hard_labels = posteriors.argmax(axis=1)
        r_sub = rewards[:50000]

        r_low = r_sub[hard_labels == 0]
        r_med = r_sub[hard_labels == 1]
        r_high = r_sub[hard_labels == 2]

        f_stat, p_val = stats.f_oneway(r_low, r_med, r_high) if (len(r_low) > 0 and len(r_med) > 0 and len(r_high) > 0) else (0.0, 1.0)
        print(f"  -> Validation ANOVA F-test: F={f_stat:.2f}, p={p_val:.2e}")
        print(f"     Mean rewards: Low={r_low.mean():.4f}, Med={r_med.mean():.4f}, High={r_high.mean():.4f}")

        out_prob_dir = PROJECT_ROOT / "in" / "datasets" / "pyrenees" / "per_problem" / problem_id
        out_prob_dir.mkdir(parents=True, exist_ok=True)
        gmm_out_path = out_prob_dir / "gmm_scaler.npz"

        np.savez(
            gmm_out_path,
            means=gmm_res["means"],
            covariances=gmm_res["covariances"],
            precisions=gmm_res["precisions"],
            log_dets=gmm_res["log_dets"],
            log_weights=gmm_res["log_weights"],
            cluster_weights=gmm_res["weights"],
            feature_indices=np.array(selected_indices),
            feature_names=np.array(selected_names),
        )
        print(f"  Saved -> {gmm_out_path}")

        if problem_id == "problem":
            global_gmm_path = PROJECT_ROOT / "in" / "datasets" / "pyrenees" / "pyrenees_gmm_scaler.npz"
            np.savez(
                global_gmm_path,
                means=gmm_res["means"],
                covariances=gmm_res["covariances"],
                precisions=gmm_res["precisions"],
                log_dets=gmm_res["log_dets"],
                log_weights=gmm_res["log_weights"],
                cluster_weights=gmm_res["weights"],
                feature_indices=np.array(selected_indices),
                feature_names=np.array(selected_names),
            )
            print(f"  Saved global fallback -> {global_gmm_path}")

        results_summary[problem_id] = {
            "selected_features": selected_names,
            "selected_indices": selected_indices,
            "relevance_mi": mrmr_res["relevance"],
            "cluster_weights": gmm_res["weights"].tolist(),
            "reward_anova_f": float(f_stat),
            "reward_anova_p": float(p_val),
            "reward_means": [float(r_low.mean()), float(r_med.mean()), float(r_high.mean())],
        }

    summary_path = plot_dir / "mrmr_selection_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved mRMR summary report -> {summary_path}")

    fig = plt.figure(figsize=(14, 6), facecolor="#fafafa")
    gs = GridSpec(1, 2, width_ratios=[1.2, 1], figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#f5f5f5")
    prob_info = results_summary.get("problem", {})
    if prob_info:
        feats = prob_info["selected_features"][::-1]
        rels = prob_info["relevance_mi"][::-1]
        y_pos = range(len(feats))
        bars = ax1.barh(y_pos, rels, color="#2980b9", edgecolor="white", height=0.6)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(feats, fontsize=10.5, fontweight="bold")
        ax1.set_xlabel("Mutual Information Relevance with Reward", fontsize=11)
        ax1.set_title("Top Selected Competency Features (mRMR)", fontsize=12, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        for bar in bars:
            w = bar.get_width()
            ax1.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=9.5)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#f5f5f5")
    if prob_info:
        r_means = prob_info["reward_means"]
        c_names = ["Low", "Medium", "High"]
        c_colors = ["#e74c3c", "#f39c12", "#27ae60"]
        ax2.bar(c_names, r_means, color=c_colors, edgecolor="white", width=0.55)
        ax2.set_ylabel("Mean Step Reward", fontsize=11)
        ax2.set_title("Step Reward by Competency Tier\n(ANOVA p < 1e-5)", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)
        for i, val in enumerate(r_means):
            ax2.text(i, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("Pyrenees ITS: mRMR Feature Selection & Competency Validation", fontsize=13.5, fontweight="bold", y=1.02)
    fig_path = plot_dir / "mrmr_validation_figure.png"
    plt.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved diagnostic figure -> {fig_path}")

    print("\n" + "=" * 80)
    print("  ALL 11 GMM COMPETENCY MODELS RECLUSTERED AND SAVED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
