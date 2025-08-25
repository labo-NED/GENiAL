import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# 1. Load and preprocess
# -------------------------
behavioral_vars = [
    # 'IQ',
    'SRS_social_cognition_tscore',
    'SRS_social_communication_tscore',
    'SRS_restrictive_repetitive_tscore',
    'ASEBA_internalizing_problems_tscore',
    'ASEBA_externalizing_problems_tscore',
    'ASEBA_aggressive_behavior_tscore',
    'ASEBA_attention_problems_tscore',
    'ASEBA_anxious_depressed_tscore',
    'SCQ_score' # Social Communication Questionnaire
]

df = pd.read_csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/Q1KDatabase-ECNEEGIQGENCHUSJ_DATA_for_cluster_analysis.csv")   # Replace with your dataset

# Filter for probands only (participants ending with _P)
print("Original dataset size:", len(df))
df = df[df['participant_id'].str.endswith('_P', na=False)]
print("After filtering for probands:", len(df))

# Check for missing values
print("Missing values per column:")
print(df[behavioral_vars].isnull().sum())
print(f"\nTotal rows: {len(df)}")
print(f"Rows with complete data: {df[behavioral_vars].dropna().shape[0]}")

# Prepare data for Gower distance (keep missing values as NaN)
X = df[behavioral_vars].values
print(f"\nData shape: {X.shape}")
print(f"Total missing values: {np.isnan(X).sum()}")
print(f"Percentage missing: {np.isnan(X).sum() / X.size * 100:.1f}%")

# -------------------------
# 2. Gower Distance + Hierarchical Clustering
# -------------------------

# Calculate Gower distance matrix
print("\nCalculating Gower distance matrix...")
# Note: We'll use a simplified approach since scipy doesn't have built-in Gower
# For now, we'll use a robust distance metric that handles missing values

# Use pairwise complete observations for distance calculation
from sklearn.metrics.pairwise import nan_euclidean_distances

# Calculate distance matrix using nan_euclidean (handles missing values)
print("Computing distance matrix with missing value handling...")
distance_matrix = nan_euclidean_distances(X, X)

# Hierarchical clustering
print("Performing hierarchical clustering...")
linkage_matrix = linkage(squareform(distance_matrix), method='ward')

# Test different numbers of clusters
sil_scores = []
K_range = range(2, 11)  # Test 2-10 clusters

for k in K_range:
    # Get cluster labels
    cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust') - 1  # Convert to 0-based indexing
    
    # Calculate silhouette score (using the distance matrix)
    try:
        sil = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
        sil_scores.append(sil)
    except:
        # If silhouette fails, use a different approach
        sil_scores.append(0)

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(K_range, sil_scores, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
plt.title(f"Hierarchical Clustering: choosing K (n={len(df)} probands)")
plt.grid(True, alpha=0.3)
plt.show()

# Best k
best_k = K_range[np.argmax(sil_scores)]
print(f"\nBest K: {best_k}")

# Final clustering with best K
final_cluster_labels = fcluster(linkage_matrix, best_k, criterion='maxclust') - 1

# Add cluster labels to dataframe
df['cluster'] = final_cluster_labels

# Show cluster sizes
unique, counts = np.unique(final_cluster_labels, return_counts=True)
print(f"\nNumber of participants per cluster:")
for cl, cnt in zip(unique, counts):
    print(f"  Cluster {cl}: {cnt} participants")

# -------------------------
# 3. ANALYZE CLUSTER CHARACTERISTICS
# -------------------------
print("\n" + "="*60)
print("CLUSTER ANALYSIS")
print("="*60)

# Calculate cluster means for each variable
cluster_means = df.groupby('cluster')[behavioral_vars].mean()
print("\nCluster means by variable:")
print(cluster_means.round(2))

# Calculate cluster standard deviations
cluster_stds = df.groupby('cluster')[behavioral_vars].std()
print("\nCluster standard deviations:")
print(cluster_stds.round(2))

# -------------------------
# 4. VISUALIZATIONS
# -------------------------
print("\nCreating visualizations...")

# Dendrogram
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
dendrogram(linkage_matrix, labels=df['participant_id'].values, leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Participant ID')
plt.ylabel('Distance')

# Heatmap of cluster means
plt.subplot(2, 2, 2)
sns.heatmap(cluster_means.T, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
plt.title('Cluster Means by Variable')
plt.ylabel('Variables')
plt.xlabel('Cluster')

# Box plots for key variables
plt.subplot(2, 2, 3)
key_vars = ['SRS_social_cognition_tscore', 'SCQ_score']
for i, var in enumerate(key_vars):
    plt.subplot(2, 2, 3 + i)
    df.boxplot(column=var, by='cluster', ax=plt.gca())
    plt.title(f'{var} by Cluster')
    plt.suptitle('')  # Remove automatic suptitle

plt.tight_layout()
plt.show()

# -------------------------
# 5. SAVE RESULTS
# -------------------------
# Save the clustered data
output_file = "proband_clusters_gower.csv"
df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
