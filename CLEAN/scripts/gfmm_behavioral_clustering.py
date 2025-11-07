import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# PATHS & CONTSTANTS 
# ------------------------------------------------------------
ROOT_DIR = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN"
INPUT_FILE = ROOT_DIR + "/Outputs/Preprocessed/Q1K_CHU_BC_DATA_NOV_05_2025.csv"
OUTPUT_FILE = ROOT_DIR + "/Outputs/Clustered/clustered_GMM_Q1K_CHU_BC_DATA_NOV_05_2025.csv"
SILHOUETTE_PLOTS_FILE = ROOT_DIR + "/Outputs/Plots/GMM_Q1K_CHU_BC_DATA_NOV_05_2025_silhouette_scores.png"
RADAR_PLOTS_FILE = ROOT_DIR + "/Outputs/Plots/GMM_Q1K_CHU_BC_DATA_NOV_05_2025_cluster_radars.png"

BEHAVIORAL_VARS = [
    'SRS_social_cognition_tscore',
    'SRS_social_communication_tscore',
    'SRS_restrictive_repetitive_tscore',
    # 'attention_deficit_hyperactivity_tscore',
    # 'oppositional_defiant_tscore',
    'nonverbal_iq',
    'verbal_iq'
    # 'ghf_sleeping'
]
pretty_labels = {
    'SRS_social_cognition_tscore': 'Social Cognition',
    'SRS_social_communication_tscore': 'Social Communication',
    'SRS_restrictive_repetitive_tscore': 'Repetitive behavior',
    # 'attention_deficit_hyperactivity_tscore': 'ADHD',
    # 'oppositional_defiant_tscore': 'Oppositional',
    'nonverbal_iq': 'NVIQ',
    'verbal_iq': 'VIQ'
    # 'ghf_sleeping': 'Sleeping'
}

# Load the data
df = pd.read_csv(INPUT_FILE)

# Keep participants with age between 5-18 inclusively
df = df[(df['age_at_test'] >= 5) & (df['age_at_test'] < 19)]

# # Filter participants with autism, ADHD, or ASD diagnoses (handle NaNs safely)
# diagnosis_mask = (~df['diagnosis'].isna()) & (df['diagnosis'].str.strip().str.lower() != 'none') & (df['diagnosis'].str.strip() != '')
# df = df[diagnosis_mask]

# Ensure behavioral variables are numeric
for col in BEHAVIORAL_VARS:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with any NaNs in the selected behavioral variables to avoid GMM errors
complete_case_mask = df[BEHAVIORAL_VARS].notna().all(axis=1)
df = df.loc[complete_case_mask].copy()

# ----------------------------------------------
# Standardize the data for Gaussian Mixture Model
# ----------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(df[BEHAVIORAL_VARS])

print(f"Data shape after preprocessing: {X.shape}")
print(f"Any NaN values remaining? {np.isnan(X).any()}")

# -------------------------
# Gaussian Mixture Model clustering
# -------------------------
# Model selection metrics
sil_scores = []
aic_scores = []
bic_scores = []
K_range = range(2, 11)  # Test 2-10 clusters

print("\nEvaluating different numbers of components:")
print("K\tSilhouette\tAIC\t\tBIC")
print("-" * 50)

for k in K_range:
    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=k, random_state=42, max_iter=200)
    gmm.fit(X)
    
    # Get cluster assignments
    labels = gmm.predict(X)
    
    # Calculate metrics
    sil = silhouette_score(X, labels)
    aic = gmm.aic(X)
    bic = gmm.bic(X)
    
    sil_scores.append(sil)
    aic_scores.append(aic)
    bic_scores.append(bic)
    
    print(f"{k}\t{sil:.4f}\t\t{aic:.2f}\t\t{bic:.2f}")

# -------------------------
# Plots for model selection
# -------------------------

# Create subplots for different metrics
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot silhouette scores
axes[0, 0].plot(K_range, sil_scores, marker='o', color='blue')
axes[0, 0].set_xlabel("Number of components")
axes[0, 0].set_ylabel("Silhouette score")
axes[0, 0].set_title("Silhouette Score vs Number of Components")
axes[0, 0].grid(True, alpha=0.3)

# Plot AIC scores
axes[0, 1].plot(K_range, aic_scores, marker='s', color='red')
axes[0, 1].set_xlabel("Number of components")
axes[0, 1].set_ylabel("AIC")
axes[0, 1].set_title("AIC vs Number of Components")
axes[0, 1].grid(True, alpha=0.3)

# Plot BIC scores
axes[1, 0].plot(K_range, bic_scores, marker='^', color='green')
axes[1, 0].set_xlabel("Number of components")
axes[1, 0].set_ylabel("BIC")
axes[1, 0].set_title("BIC vs Number of Components")
axes[1, 0].grid(True, alpha=0.3)

# Combined plot
axes[1, 1].plot(K_range, sil_scores, marker='o', label='Silhouette', color='blue')
axes[1, 1].plot(K_range, np.array(aic_scores)/max(aic_scores), marker='s', label='AIC (normalized)', color='red')
axes[1, 1].plot(K_range, np.array(bic_scores)/max(bic_scores), marker='^', label='BIC (normalized)', color='green')
axes[1, 1].set_xlabel("Number of components")
axes[1, 1].set_ylabel("Normalized scores")
axes[1, 1].set_title("Combined Model Selection Metrics")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle(f"Gaussian Mixture Model: Model Selection (n={len(df)}, Complete Cases)", y=1.02)
# Save silhouette plots
plt.savefig(SILHOUETTE_PLOTS_FILE, dpi=300)
print(f"Saved silhouette plots: {SILHOUETTE_PLOTS_FILE}")

plt.show()

# -------------------------
# Select best model based on different criteria
# -------------------------
best_k_sil = K_range[np.argmax(sil_scores)]
best_k_aic = K_range[np.argmin(aic_scores)]
best_k_bic = K_range[np.argmin(bic_scores)]

print(f"\nBest K based on different criteria:")
print(f"  Silhouette score: {best_k_sil}")
print(f"  AIC: {best_k_aic}")
print(f"  BIC: {best_k_bic}")

# Use BIC as the primary criterion (commonly used for GMM)
best_k = 4 # best_k_bic
print(f"\nUsing K = {best_k} (based on BIC)")

# -------------------------------------------------
# Final clustering with best K
# -------------------------------------------------
gmm_final = GaussianMixture(n_components=best_k, random_state=42, max_iter=200)
gmm_final.fit(X)
cluster_labels = gmm_final.predict(X)

# Get cluster probabilities
cluster_probs = gmm_final.predict_proba(X)

# Add cluster labels back to the processed dataframe
df_processed_with_clusters = df.copy()
df_processed_with_clusters['cluster'] = cluster_labels

# Add cluster assignment column to the original dataframe
df_with_clusters = df.copy()
df_with_clusters['cluster'] = np.nan  # Initialize with NaN
df_with_clusters.loc[df.index, 'cluster'] = cluster_labels

# Add cluster probabilities
for i in range(best_k):
    df_with_clusters[f'cluster_{i}_prob'] = np.nan
    df_with_clusters.loc[df.index, f'cluster_{i}_prob'] = cluster_probs[:, i]

# Show cluster sizes
unique, counts = np.unique(cluster_labels, return_counts=True)
print(f"\nNumber of participants per cluster:")
for cl, cnt in zip(unique, counts):
    print(f"  Cluster {cl}: {cnt} participants")

# Show cluster means in original scale
print(f"\nCluster means (original scale):")
cluster_means = scaler.inverse_transform(gmm_final.means_)
for i, mean in enumerate(cluster_means):
    print(f"  Cluster {i}:")
    for j, var in enumerate(BEHAVIORAL_VARS):
        print(f"    {var}: {mean[j]:.2f}")

# Show model parameters
print(f"\nModel parameters:")
print(f"  Converged: {gmm_final.converged_}")
print(f"  Number of iterations: {gmm_final.n_iter_}")
print(f"  Log-likelihood: {gmm_final.score(X):.2f}")
print(f"  AIC: {gmm_final.aic(X):.2f}")
print(f"  BIC: {gmm_final.bic(X):.2f}")

# =========================
# Radar plots per cluster
# =========================
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

# Nice labels (order matches your vars)
pretty_labels = {
    'SRS_social_cognition_tscore': 'Social Cognition',
    'SRS_social_communication_tscore': 'Social Communication',
    'SRS_restrictive_repetitive_tscore': 'Repetitive behavior',
    # 'attention_deficit_hyperactivity_tscore': 'ADHD',
    # 'oppositional_defiant_tscore': 'Oppositional'
    'nonverbal_iq': 'NVIQ',
    'verbal_iq': 'VIQ'
}
labels = [pretty_labels[v] for v in BEHAVIORAL_VARS]
p = len(labels)

# Option: normalize across clusters so all radars share a 0–1 scale
# Comment this block if you prefer true T-score values
vals = cluster_means.copy()                         # K x p (original scale from earlier)
vmin = vals.min(axis=0); vmax = vals.max(axis=0)
rng = np.where((vmax - vmin) == 0, 1, (vmax - vmin))
vals_norm = (vals - vmin) / rng                # 0..1 for plotting
radar_vals = vals                         # change to vals_norm use normal scale

# Angles for the polygon
angles = np.linspace(0, 2*np.pi, p, endpoint=False)
angles = np.concatenate([angles, angles[:1]])       # close the loop

# Layout
K = vals.shape[0]
rows, cols = ceil(K/2), 2 if K > 1 else 1
fig = plt.figure(figsize=(5*cols, 5*rows))

# Map cluster id -> size
sizes = dict(zip(unique, counts))

for k in range(K):
    ax = plt.subplot(rows, cols, k+1, projection='polar')
    # values for this cluster, closed polygon
    v = radar_vals[k]
    data = np.concatenate([v, v[:1]])

    # draw outer reference polygon (pentagon outline)
    ref = np.ones(p)
    ref = np.concatenate([ref, ref[:1]])
    # ax.plot(angles, ref, color='black', linewidth=2)
    
    # cluster polygon
    ax.plot(angles, data, linewidth=2, color='purple')
    ax.fill(angles, data, alpha=0.15, color='purple')

    # formatting
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks([])                   # cleaner look
    # ax.set_ylim(0, 1)                   # because we normalized 0..1
    min_T = vals.min()
    max_T = vals.max()
    ax.set_ylim(min_T, max_T)
    ax.set_title(f"Cluster {k}  (n={sizes.get(k,0)})", fontsize=13, pad=14)

plt.tight_layout()
out_png = RADAR_PLOTS_FILE
plt.savefig(out_png, dpi=300)
plt.show()
print(f"Saved radar plots: {out_png}")

# Save df with clusters
df_with_clusters.to_csv(OUTPUT_FILE, index=False)

print(f"\nResults saved to: {OUTPUT_FILE}")
