"""
fit_gmm_competency.py — Advanced Multi-Dimensional GMM Clustering & Validation for Pyrenees.

This script replaces static 2D KMeans thresholding with a multi-dimensional Gaussian Mixture Model (GMM).
Key enhancements:
  1. Multi-feature competency space:
     - Accuracy: pctCorrect (72), pctCorrectKC (76), pctCorrectSession (80)
     - Error recovery / Fluency: nStepSinceLastWrong (84)
     - Assistance reliance: nTotalHintSession (43)
     - Pacing: avgTimeOnStep (23)
  2. Model selection & justification:
     - Evaluates K in [2, 6] using BIC, AIC, Silhouette score, Calinski-Harabasz Index, Davies-Bouldin Index.
  3. Empirical validation:
     - ANOVA F-test & Welch t-test on next-step rewards / error rates per cluster.
  4. Smooth continuous posteriors:
     - Saves PyTorch-compatible GMM parameters (means, precision matrices, log-dets, log-weights)
       so valuation.py can return exact P(Cluster = Low|Med|High | s) in [0, 1].

Usage:
    python scripts/fit_gmm_competency.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy import stats

# ── Feature selection ────────────────────────────────────────────────────────
FEATURE_INDICES = [72, 76, 80, 84, 43, 23]
FEATURE_NAMES = [
    "pctCorrect",
    "pctCorrectKC",
    "pctCorrectSession",
    "nStepSinceLastWrong",
    "nTotalHintSession",
    "avgTimeOnStep",
]

DATASET_NPZ = "in/datasets/pyrenees/pyrenees_clean.npz"
OUT_SCALER  = "in/datasets/pyrenees/pyrenees_gmm_scaler.npz"
OUT_PLOT_DIR = "results/plots/pyrenees"


def load_data():
    if not os.path.exists(DATASET_NPZ):
        raise FileNotFoundError(f"Pyrenees clean dataset not found at {DATASET_NPZ}. Run preprocess_pyrenees.py first.")
    print(f"Loading dataset from {DATASET_NPZ}...")
    data = np.load(DATASET_NPZ, allow_pickle=True)
    states  = data["states"]   # (N_trajs,) array of (T, 123)
    rewards = data["rewards"]  # (N_trajs,) array of (T,)
    dones   = data["dones"]    # (N_trajs,) array of (T,)
    
    # Subsample for cluster model selection if huge (e.g. 100k steps for fast metric computation)
    all_steps = np.vstack(states)
    all_rewards = np.hstack(rewards)
    
    print(f"Loaded {len(states):,} trajectories ({all_steps.shape[0]:,} total steps, {all_steps.shape[1]} features).")
    return all_steps, all_rewards, states


def evaluate_cluster_numbers(X, k_range=range(2, 7), sample_size=5000):
    """Evaluate K=2..6 using BIC, AIC, Silhouette, CHI, and DBI."""
    print(f"\nEvaluating K in {list(k_range)} on a random sample of {sample_size:,} steps...", flush=True)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sub = X[idx]

    bics, aics = [], []
    silhouettes, chis, dbis = [], [], []

    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=5, covariance_type="full", init_params="random_from_data")
        labels = gmm.fit_predict(X_sub)
        
        bics.append(gmm.bic(X_sub))
        aics.append(gmm.aic(X_sub))
        
        sil = silhouette_score(X_sub, labels, sample_size=5000, random_state=42)
        chi = calinski_harabasz_score(X_sub, labels)
        dbi = davies_bouldin_score(X_sub, labels)
        
        silhouettes.append(sil)
        chis.append(chi)
        dbis.append(dbi)
        
        print(f"  K={k}: BIC={bics[-1]:.1f}, AIC={aics[-1]:.1f}, Silhouette={sil:.4f}, CHI={chi:.1f}, DBI={dbi:.4f}", flush=True)

    return list(k_range), bics, aics, silhouettes, chis, dbis


def fit_final_gmm(X, n_components=3, sample_size=100000):
    """Fit full GMM and order clusters low -> med -> high competency by average performance."""
    print(f"\nFitting final GMM with K={n_components} on sample of {sample_size:,} steps...", flush=True)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_feat = X[idx][:, FEATURE_INDICES]

    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=42, n_init=5, init_params="random_from_data")
    gmm.fit(X_feat)

    # Order clusters by primary performance metric (pctCorrect + pctCorrectKC + pctCorrectSession)
    perf_score = gmm.means_[:, 0] + gmm.means_[:, 1] + gmm.means_[:, 2]
    order = np.argsort(perf_score)  # 0=Low, 1=Med, 2=High
    
    # Re-order GMM parameters for PyTorch
    means = gmm.means_[order]
    covariances = gmm.covariances_[order]
    weights = gmm.weights_[order]
    
    # Precompute precision matrices & log determinants for PyTorch fast inference
    precisions = np.array([np.linalg.inv(c) for c in covariances])
    log_dets = np.array([np.linalg.slogdet(c)[1] for c in covariances])
    log_weights = np.log(weights + 1e-12)

    print("Final GMM Cluster Centroids (z-scores):", flush=True)
    tier_names = ["Low Competency", "Med Competency", "High Competency"]
    for i, name in enumerate(tier_names):
        print(f"  [{name}] Weight={weights[i]:.3f}", flush=True)
        for fname, val in zip(FEATURE_NAMES, means[i]):
            print(f"      {fname:20s}: {val:+.3f}", flush=True)

    return gmm, means, precisions, log_dets, log_weights, order


def validate_clusters_statistically(X, rewards, gmm, feature_indices, order, sample_size=100000):
    """Compute ANOVA & Welch t-test on rewards/step metrics per cluster."""
    print("\nPerforming Statistical Validation of Competency Clusters...", flush=True)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_feat = X[idx][:, feature_indices]
    rewards_sub = rewards[idx]

    posteriors = gmm.predict_proba(X_feat)[:, order]  # reordered Low -> Med -> High
    hard_labels = posteriors.argmax(axis=1)

    r_low  = rewards_sub[hard_labels == 0]
    r_med  = rewards_sub[hard_labels == 1]
    r_high = rewards_sub[hard_labels == 2]

    f_stat, p_val = stats.f_oneway(r_low, r_med, r_high)
    print(f"  ANOVA F-test on Step Rewards across Competency Tiers:", flush=True)
    print(f"    F-statistic = {f_stat:.2f}, p-value = {p_val:.2e}", flush=True)
    print(f"    Low  Competency mean reward: {r_low.mean():.4f} (std={r_low.std():.4f}, N={len(r_low):,})", flush=True)
    print(f"    Med  Competency mean reward: {r_med.mean():.4f} (std={r_med.std():.4f}, N={len(r_med):,})", flush=True)
    print(f"    High Competency mean reward: {r_high.mean():.4f} (std={r_high.std():.4f}, N={len(r_high):,})", flush=True)

    # Welch t-test (Low vs High)
    t_stat, p_welch = stats.ttest_ind(r_low, r_high, equal_var=False)
    print(f"  Welch t-test (Low vs High): t={t_stat:.2f}, p={p_welch:.2e}", flush=True)

    return {
        "f_stat": f_stat,
        "p_val": p_val,
        "means": [r_low.mean(), r_med.mean(), r_high.mean()],
        "stds": [r_low.std(), r_med.std(), r_high.std()],
        "counts": [len(r_low), len(r_med), len(r_high)],
    }


def save_plots(k_vals, bics, aics, sils, dbis, gmm, X_feat, rewards, order, out_dir):
    """Save comprehensive publication quality diagnostic and validation figures."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # ── Figure 1: Model Selection & Metric Curves ────────────────────────────
    fig = plt.figure(figsize=(14, 5), facecolor="#fafafa")
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)
    
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#f5f5f5")
    ax1.plot(k_vals, bics, "o-", color="#2c3e50", label="BIC", linewidth=2)
    ax1.plot(k_vals, aics, "s--", color="#7f8c8d", label="AIC", linewidth=2)
    ax1.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax1.set_ylabel("Information Criterion", fontsize=11)
    ax1.set_title("BIC / AIC Model Selection", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#f5f5f5")
    ax2.plot(k_vals, sils, "^-", color="#27ae60", linewidth=2, markersize=8)
    ax2.axvline(3, color="#e74c3c", linestyle="--", alpha=0.7, label="Selected K=3")
    ax2.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax2.set_ylabel("Silhouette Score", fontsize=11)
    ax2.set_title("Cluster Separation (Silhouette)", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor("#f5f5f5")
    ax3.plot(k_vals, dbis, "d-", color="#8e44ad", linewidth=2, markersize=8)
    ax3.axvline(3, color="#e74c3c", linestyle="--", alpha=0.7, label="Selected K=3")
    ax3.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax3.set_ylabel("Davies-Bouldin Index (Lower is Better)", fontsize=11)
    ax3.set_title("Cluster Similarity (DBI)", fontsize=12, fontweight="bold")
    ax3.legend()
    ax3.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Pyrenees GMM Competency Cluster Model Justification", fontsize=14, fontweight="bold", y=1.03)
    p1 = os.path.join(out_dir, "gmm_cluster_metrics.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved metric diagnostic figure -> {p1}", flush=True)

    # ── Figure 2: Competency Posteriors & Validation Scatter ──────────────────
    fig = plt.figure(figsize=(14, 6), facecolor="#fafafa")
    gs = GridSpec(1, 2, width_ratios=[1.5, 1], figure=fig, wspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#f5f5f5")

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_feat), size=min(50000, len(X_feat)), replace=False)
    X_sub = X_feat[idx]
    posteriors = gmm.predict_proba(X_sub)[:, order]
    hard_labels = posteriors.argmax(axis=1)

    colors = ["#e74c3c", "#f39c12", "#27ae60"]
    tier_labels = ["Low Competency", "Medium Competency", "High Competency"]

    for tier in [0, 1, 2]:
        m = hard_labels == tier
        ax1.scatter(
            X_sub[m, 0], X_sub[m, 1],
            c=colors[tier], label=tier_labels[tier],
            alpha=0.35, s=10, rasterized=True
        )

    # Plot GMM means (reordered)
    means = gmm.means_[order]
    for tier in [0, 1, 2]:
        ax1.scatter(
            means[tier, 0], means[tier, 1],
            c=colors[tier], s=300, marker="*",
            edgecolors="black", linewidths=1.5, zorder=5
        )

    ax1.set_xlabel("pctCorrect (z-score)", fontsize=11)
    ax1.set_ylabel("pctCorrectKC (z-score)", fontsize=11)
    ax1.set_title("GMM Multi-Feature State Probabilities\n(Posterior Soft Clustering)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.spines[["top", "right"]].set_visible(False)

    # Bar chart of cluster proportion
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#f5f5f5")
    counts = np.bincount(hard_labels, minlength=3)
    total = len(X_sub)
    bars = ax2.bar(range(3), counts, color=colors, edgecolor="white", width=0.55)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["Low", "Med", "High"], fontsize=11)
    ax2.set_ylabel("Number of Steps", fontsize=11)
    ax2.set_title("Step State Distribution", fontsize=12, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)

    for bar, n in zip(bars, counts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{n:,}\n({100*n/total:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    plt.suptitle("Pyrenees Multi-Dimensional GMM Student Competency Tiers", fontsize=14, fontweight="bold", y=1.02)
    p2 = os.path.join(out_dir, "gmm_competency_validation.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved competency validation figure -> {p2}", flush=True)


def main():
    all_steps, rewards, states = load_data()
    
    # 1. Model selection across K=2..6
    k_vals, bics, aics, sils, chis, dbis = evaluate_cluster_numbers(all_steps[:, FEATURE_INDICES])
    
    # 2. Fit final 3-component GMM (Low, Med, High competency)
    gmm, means, precisions, log_dets, log_weights, order = fit_final_gmm(all_steps, n_components=3)
    
    # 3. Statistical validation
    stats_info = validate_clusters_statistically(all_steps, rewards, gmm, FEATURE_INDICES, order)
    
    # 4. Save diagnostic plots
    save_plots(k_vals, bics, aics, sils, dbis, gmm, all_steps[:, FEATURE_INDICES], rewards, order, OUT_PLOT_DIR)
    
    # 5. Save GMM parameters for valuation.py
    np.savez(
        OUT_SCALER,
        feature_indices=np.array(FEATURE_INDICES),
        feature_names=np.array(FEATURE_NAMES),
        means=means,
        precisions=precisions,
        log_dets=log_dets,
        log_weights=log_weights,
        cluster_weights=gmm.weights_,
    )
    print(f"\nGMM scaler parameters successfully saved -> {OUT_SCALER}")

    print("\nMulti-dimensional GMM fitting and validation complete!")


if __name__ == "__main__":
    main()
