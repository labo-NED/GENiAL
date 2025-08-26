import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# CONFIGURATION - Change this to switch between methods
# -------------------------
USE_MEDIAN_IMPUTATION = True  # Set to True for median imputation, False for dropping NAs

# -------------------------
# 1. Load and preprocess
# -------------------------
behavioral_vars = [
    'IQ',
    'SRS_social_cognition_tscore',
    'SRS_social_communication_tscore',
    'SRS_restrictive_repetitive_tscore',
    'ASEBA_attention_problems_tscore',
    'SCQ_score' # Social Communication Questionnaire
]

df = pd.read_csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/Q1KDatabase-ECNEEGIQGENCHUSJ_DATA_for_cluster_analysis.csv")   # Replace with your dataset

# Filter for probands and siblings only (participants ending with _P or _S)
print("Original dataset size:", len(df))
df = df[df['participant_id'].str.endswith('_P', na=False) | df['participant_id'].str.contains(r'_S\d+$', na=False)]
print("After filtering for probands and siblings:", len(df))

# Keep only participants aged 0 to 20
age_col = 'eeg_age_years_testdate'
df = df[(df[age_col] >= 0) & (df[age_col] <= 18)]
print(f"After filtering for age 0-20: {len(df)} participants")


# Ensure participants have at least one diagnosis (at least one 1 in any diagnosis column)
diagnosis_cols = [
    'ASD', 'ASD_behavior', 'ADHD', 'OCD', 'motor_disorder','anxiety', 'neurological_conditions', 'genetic_disorder', 'other'
]
print("Original dataset size after proband/sibling filter:", len(df))
df = df[df[diagnosis_cols].sum(axis=1) >= 1]
print("After filtering for probands/siblings with at least one diagnosis (sum of diagnosis columns >= 1):", len(df))

# Handle missing values based on configuration
if USE_MEDIAN_IMPUTATION:
    # Median imputation if participant has at most 2/6 missing, otherwise drop
    max_missing_allowed = 2
    missing_counts = df[behavioral_vars].isnull().sum(axis=1)
    eligible_for_imputation = missing_counts <= max_missing_allowed

    print(f"\nUsing MEDIAN IMPUTATION for participants with at most {max_missing_allowed} missing variables (out of {len(behavioral_vars)})")
    print(f"Participants eligible for imputation: {eligible_for_imputation.sum()} / {len(df)}")
    print(f"Dropping participants with >{max_missing_allowed} missing variables: {(~eligible_for_imputation).sum()}")

    df_eligible = df.loc[eligible_for_imputation, behavioral_vars].copy()

    # Impute missing values with median for eligible participants
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(df_eligible)
    df_processed = df_eligible.copy()
    df_processed.iloc[:, :] = X_imputed

    print(f"Imputed {df_eligible.isnull().sum().sum()} missing values")
    
    # Get the sex distribution for the participants in df_processed
    sex_counts = df.loc[df_processed.index, 'sex'].value_counts(dropna=False)
    total = len(df_processed)
    n_female = sex_counts.get('F', 0)
    n_male = sex_counts.get('M', 0)
    n_other = sex_counts.sum() - n_female - n_male

    print("\nSex distribution in the data used for clustering:")
    print(f"  Female (F): {n_female} ({n_female/total:.1%})")
    print(f"  Male (M):   {n_male} ({n_male/total:.1%})")
    if n_other > 0:
        print(f"  Other/Unknown: {n_other} ({n_other/total:.1%})")
else:
    print(f"\nUsing COMPLETE CASES ONLY (dropping participants with missing data)")
    # Use only complete cases
    df_processed = df[behavioral_vars].dropna()
    print(f"Using {len(df_processed)} participants with complete data")

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(df_processed)

print(f"Data shape after preprocessing: {X.shape}")
print(f"Any NaN values remaining? {np.isnan(X).any()}")

# -------------------------
# 2. K-means clustering
# -------------------------
sil_scores = []
K_range = range(2, 5)  # Test 2-5 clusters

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    sil = silhouette_score(X, labels)
    sil_scores.append(sil)

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(K_range, sil_scores, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
method_name = "Median Imputation" if USE_MEDIAN_IMPUTATION else "Complete Cases"
plt.title(f"K-means: choosing K (n={len(df_processed)} probands, {method_name})")
plt.grid(True, alpha=0.3)
plt.show()

# Best k
best_k = K_range[np.argmax(sil_scores)]
print(f"\nBest K: {best_k}")

# Final clustering with best K
kmeans_final = KMeans(n_clusters=best_k, random_state=42)
cluster_labels = kmeans_final.fit_predict(X)

# Add cluster labels back to the processed dataframe
df_processed_with_clusters = df_processed.copy()
df_processed_with_clusters['cluster'] = cluster_labels

# Show cluster sizes
unique, counts = np.unique(cluster_labels, return_counts=True)
print(f"\nNumber of participants per cluster:")
for cl, cnt in zip(unique, counts):
    print(f"  Cluster {cl}: {cnt} participants")

# -------------------------
# 3. ANALYZE CLUSTER CHARACTERISTICS
# -------------------------
print("\n" + "="*60)
print("CLUSTER ANALYSIS")
print("="*60)

# Calculate cluster means for each variable (original scale)
cluster_means = df_processed_with_clusters.groupby('cluster')[behavioral_vars].mean()
print("\nCluster means by variable (original scale):")
print(cluster_means.round(2))

# Calculate cluster standard deviations
cluster_stds = df_processed_with_clusters.groupby('cluster')[behavioral_vars].std()
print("\nCluster standard deviations:")
print(cluster_stds.round(2))

# -------------------------
# 4. VISUALIZATIONS
# -------------------------
print("\nCreating visualizations...")

# Heatmap of cluster centers (standardized)
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
cluster_centers = kmeans_final.cluster_centers_
sns.heatmap(cluster_centers.T, 
            xticklabels=[f'Cluster {i}' for i in range(best_k)],
            yticklabels=behavioral_vars,
            cmap='RdBu_r', center=0, annot=True, fmt='.2f')
plt.title('Cluster Centers (Standardized Values)')

# Heatmap of cluster means (original scale)
plt.subplot(2, 2, 2)
sns.heatmap(cluster_means.T, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
plt.title('Cluster Means (Original Scale)')
plt.ylabel('Variables')
plt.xlabel('Cluster')

# Box plots for key variables
key_vars = ['SRS_social_communication_tscore', 'SCQ_score']
for i, var in enumerate(key_vars):
    plt.subplot(2, 2, 3 + i)
    df_processed_with_clusters.boxplot(column=var, by='cluster', ax=plt.gca())
    plt.title(f'{var} by Cluster')
    plt.suptitle('')  # Remove automatic suptitle

plt.tight_layout()
plt.show()

# -------------------------
# 5. SAVE RESULTS
# -------------------------
# Save the clustered data
method_suffix = "median_imputed" if USE_MEDIAN_IMPUTATION else "complete_cases"
output_file = f"proband_clusters_kmeans_{method_suffix}.csv"
df_processed_with_clusters.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)