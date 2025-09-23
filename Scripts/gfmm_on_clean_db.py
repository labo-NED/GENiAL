import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/final_db.csv")

# Keep only participants with ASD, ASD_behavior, or ADHD
# Use parentheses to ensure correct operator precedence with bitwise OR
df = df[(df['ASD'] == 1) | (df['ADHD'] == 1)]

# Save database as new CSV
df.to_csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/ASD_ADHD_GFMM_db.csv", index=False)

# Define the same behavioral variables as in the k-means script
behavioral_vars = [
    # 'IQ',
    'SRS_social_cognition_tscore', # Social Cognition
    'SRS_social_communication_tscore', # Social Communication
    'SRS_restrictive_repetitive_tscore', # Restrictive/repetitive behaviors
    'attention_deficit_hyperactivity_tscore' # Attention problems
]

# ----------------------------------------------
# Standardize the data for Gaussian Mixture Model
# ----------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(df[behavioral_vars])

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
best_k = best_k_sil
print(f"\nUsing K = {best_k} (based on SILHOUETTE)")

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
    for j, var in enumerate(behavioral_vars):
        print(f"    {var}: {mean[j]:.2f}")

# Show model parameters
print(f"\nModel parameters:")
print(f"  Converged: {gmm_final.converged_}")
print(f"  Number of iterations: {gmm_final.n_iter_}")
print(f"  Log-likelihood: {gmm_final.score(X):.2f}")
print(f"  AIC: {gmm_final.aic(X):.2f}")
print(f"  BIC: {gmm_final.bic(X):.2f}")

# Save df with clusters
df_with_clusters.to_csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/final_ASD_ADHD_gfmm_cluster_db.csv", index=False)

print(f"\nResults saved to: /Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/final_ASD_ADHD_gfmm_cluster_db.csv")
