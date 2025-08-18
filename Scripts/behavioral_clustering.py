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
    'SCQ_score'
]

df = pd.read_csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/Q1KDatabase-ECNEEGIQGENCHUSJ_DATA_for_cluster_analysis.csv")   # Replace with your dataset

# Check for missing values
print("Missing values per column:")
print(df[behavioral_vars].isnull().sum())
print(f"\nTotal rows: {len(df)}")
print(f"Rows with complete data: {df[behavioral_vars].dropna().shape[0]}")

# Handle missing values using SimpleImputer (mean imputation)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(df[behavioral_vars])

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X_imputed)

print(f"\nData shape after preprocessing: {X.shape}")
print(f"Any NaN values remaining? {np.isnan(X).any()}")

# -------------------------
# 2. K-means clustering
# -------------------------
sil_scores = []
K_range = range(2, 20)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    sil = silhouette_score(X, labels)
    sil_scores.append(sil)

n_participants = X.shape[0]

plt.figure(figsize=(10, 6))
plt.plot(K_range, sil_scores, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")

# n_participants here is the number of rows after imputation (i.e., all participants, even those with missing data imputed)
# For a more accurate count of participants with complete (non-missing) data before imputation:
n_complete = df[behavioral_vars].dropna().shape[0]
plt.title(f"K-means: choosing K (n={n_participants} after imputation, {n_complete} with complete data)")
plt.show()

# Best k
best_k = K_range[np.argmax(sil_scores)]
print("Best K:", best_k)

# Final model with best K
kmeans_final = KMeans(n_clusters=best_k, random_state=42)
cluster_labels = kmeans_final.fit_predict(X)
unique, counts = np.unique(cluster_labels, return_counts=True)
print("\nNumber of participants per cluster (K-means, best K):")
for cl, cnt in zip(unique, counts):
    print(f"  Cluster {cl}: {cnt}")

# -------------------------
# 3. ANALYZE VARIABLE IMPORTANCE
# -------------------------
print("\n" + "="*60)
print("VARIABLE IMPORTANCE ANALYSIS")
print("="*60)

# Method 1: Cluster center analysis (standardized differences)
print("\n1. CLUSTER CENTER ANALYSIS:")
print("-" * 40)
cluster_centers = kmeans_final.cluster_centers_
center_diff = np.std(cluster_centers, axis=0)  # Standard deviation across cluster centers
variable_importance_centers = pd.DataFrame({
    'Variable': behavioral_vars,
    'Center_StdDev': center_diff
}).sort_values('Center_StdDev', ascending=False)

print("Variables ranked by standard deviation of cluster centers:")
for idx, row in variable_importance_centers.iterrows():
    print(f"  {row['Variable']:<40} {row['Center_StdDev']:.3f}")

# Method 2: ANOVA F-scores
print("\n2. ANOVA F-SCORES:")
print("-" * 40)
f_scores, p_values = f_classif(X, cluster_labels)
variable_importance_anova = pd.DataFrame({
    'Variable': behavioral_vars,
    'F_Score': f_scores,
    'P_Value': p_values
}).sort_values('F_Score', ascending=False)

print("Variables ranked by ANOVA F-scores:")
for idx, row in variable_importance_anova.iterrows():
    significance = "***" if row['P_Value'] < 0.001 else "**" if row['P_Value'] < 0.01 else "*" if row['P_Value'] < 0.05 else ""
    print(f"  {row['Variable']:<40} F={row['F_Score']:.3f}, p={row['P_Value']:.4f} {significance}")

# Method 3: Random Forest feature importance
print("\n3. RANDOM FOREST FEATURE IMPORTANCE:")
print("-" * 40)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, cluster_labels)
variable_importance_rf = pd.DataFrame({
    'Variable': behavioral_vars,
    'RF_Importance': rf.feature_importances_
}).sort_values('RF_Importance', ascending=False)

print("Variables ranked by Random Forest importance:")
for idx, row in variable_importance_rf.iterrows():
    print(f"  {row['Variable']:<40} {row['RF_Importance']:.3f}")

# Method 4: Cluster means by variable (non-standardized for interpretation)
print("\n4. CLUSTER MEANS BY VARIABLE:")
print("-" * 40)
# Create dataframe with original (imputed but not standardized) data
df_analysis = pd.DataFrame(X_imputed, columns=behavioral_vars)
df_analysis['Cluster'] = cluster_labels

cluster_means = df_analysis.groupby('Cluster')[behavioral_vars].mean()
print("Mean values by cluster (original scale):")
print(cluster_means.round(2))

# Calculate range (max - min) across clusters for each variable
cluster_ranges = cluster_means.max() - cluster_means.min()
variable_importance_range = pd.DataFrame({
    'Variable': behavioral_vars,
    'Range_Across_Clusters': cluster_ranges
}).sort_values('Range_Across_Clusters', ascending=False)

print(f"\nVariables ranked by range across clusters:")
for idx, row in variable_importance_range.iterrows():
    print(f"  {row['Variable']:<40} {row['Range_Across_Clusters']:.2f}")

# -------------------------
# 4. VISUALIZATIONS
# -------------------------
print("\n5. CREATING VISUALIZATIONS...")

# Heatmap of cluster centers
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.heatmap(cluster_centers.T, 
            xticklabels=[f'Cluster {i}' for i in range(best_k)],
            yticklabels=behavioral_vars,
            cmap='RdBu_r', center=0, annot=True, fmt='.2f')
plt.title('Cluster Centers (Standardized Values)')
plt.tight_layout()

# Variable importance comparison
plt.subplot(2, 2, 2)
importance_comparison = pd.DataFrame({
    'ANOVA_F': variable_importance_anova.set_index('Variable')['F_Score'],
    'RF_Importance': variable_importance_rf.set_index('Variable')['RF_Importance'],
    'Center_StdDev': variable_importance_centers.set_index('Variable')['Center_StdDev']
})
# Normalize each method to 0-1 scale for comparison
importance_comparison_norm = importance_comparison.div(importance_comparison.max())
importance_comparison_norm.plot(kind='bar')
plt.title('Variable Importance (Normalized)')
plt.ylabel('Normalized Importance')
plt.xticks(rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# Summary ranking
print("\n" + "="*60)
print("SUMMARY: TOP PREDICTIVE VARIABLES")
print("="*60)

# Create ranking for each method
anova_ranking = {var: idx for idx, var in enumerate(variable_importance_anova['Variable'])}
rf_ranking = {var: idx for idx, var in enumerate(variable_importance_rf['Variable'])}
center_ranking = {var: idx for idx, var in enumerate(variable_importance_centers['Variable'])}

ranks = pd.DataFrame({
    'Variable': behavioral_vars,
    'ANOVA_Rank': [anova_ranking[var] for var in behavioral_vars],
    'RF_Rank': [rf_ranking[var] for var in behavioral_vars],
    'Center_Rank': [center_ranking[var] for var in behavioral_vars]
})

ranks['Average_Rank'] = ranks[['ANOVA_Rank', 'RF_Rank', 'Center_Rank']].mean(axis=1)
final_ranking = ranks.sort_values('Average_Rank')

print("Variables ranked by average across all methods:")
for idx, row in final_ranking.iterrows():
    print(f"  {idx+1:2d}. {row['Variable']:<35} (avg rank: {row['Average_Rank']:.1f})")
