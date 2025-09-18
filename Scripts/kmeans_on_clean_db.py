import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/final_db.csv")

# Keep only participants with ASD, ASD_behavior, or ADHD
# Use parentheses to ensure correct operator precedence with bitwise OR
df = df[(df['ASD'] == 1) | (df['ADHD'] == 1)]

# Save database as new CSV
df.to_csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/ASD_ADHD_db.csv", index=False)

# Now, do kmeans clustering on the clean database
behavioral_vars = [
    # 'IQ',
    'SRS_social_cognition_tscore', # Social Cognition
    'SRS_social_communication_tscore', # Social Communication
    'SRS_restrictive_repetitive_tscore', # Restrictive/repetitive behaviors
    'attention_deficit_hyperactivity_tscore' # Attention problems
]

# ----------------------------------------------
# Standardize the data for k-means clustering
# ----------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(df[behavioral_vars])

print(f"Data shape after preprocessing: {X.shape}")
print(f"Any NaN values remaining? {np.isnan(X).any()}")

# -------------------------
# K-means clustering
# -------------------------
sil_scores = []
K_range = range(2, 6)  # Test 2-5 clusters

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    sil = silhouette_score(X, labels)
    sil_scores.append(sil)

# -------------------------
# Plots
# -------------------------

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(K_range, sil_scores, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
method_name = "Complete Cases"
plt.title(f"K-means: choosing K (n={len(df)}, {method_name})")
plt.grid(True, alpha=0.3)
plt.show()

# Best k
best_k = K_range[np.argmax(sil_scores)]
print(f"\nBest K: {best_k}")

# -------------------------------------------------
# 8. Final clustering with best K
# -------------------------------------------------
kmeans_final = KMeans(n_clusters=best_k, random_state=42)
cluster_labels = kmeans_final.fit_predict(X)

# Add cluster labels back to the processed dataframe
df_processed_with_clusters = df.copy()
df_processed_with_clusters['cluster'] = cluster_labels

# Add cluster assignment column to the original dataframe
df_with_clusters = df.copy()
df_with_clusters['cluster'] = np.nan  # Initialize with NaN
df_with_clusters.loc[df.index, 'cluster'] = cluster_labels

# Show cluster sizes
unique, counts = np.unique(cluster_labels, return_counts=True)
print(f"\nNumber of participants per cluster:")
for cl, cnt in zip(unique, counts):
    print(f"  Cluster {cl}: {cnt} participants")

# Save df with clusters
df_with_clusters.to_csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/final_ASD_ADHD_cluster_db.csv", index=False)