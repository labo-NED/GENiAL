from numpy._core.numeric import False_, True_
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
USE_MEDIAN_IMPUTATION = False  # Set to True for median imputation, False for dropping NAs
# df = pd.read_csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/Q1KDatabase-ECNEEGIQGENCHUSJ_DATA_for_cluster_analysis.csv")   # Replace with your dataset
# # Load additional info from the cleaned/flattened file
# df_complete = pd.read_csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/Q1KDatabase-ECNEEGIQGENCHUSJ_DATA_flattened_cleaned_cnv_renamedcols_IQ_groups_demog_behavioral_scores.csv")

df = pd.read_csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/Combined_Q1K_BC_DATA.csv")



# -------------------------
# 1. Load and preprocess
# -------------------------
behavioral_vars = [
    # 'IQ',
    'SRS_social_cognition_tscore', # Social Cognition
    'SRS_social_communication_tscore', # Social Communication
    'SRS_restrictive_repetitive_tscore', # Restrictive/repetitive behaviors
    'attention_deficit_hyperactivity_tscore' # Attention problems
]

# -------------------------
# 2. Filtering
# -------------------------
print("Original dataset size:", len(df))

# Filter for probands and siblings only (participants ending with _P or _S)
# df = df[df['participant_id'].str.endswith('_P', na=False) | df['participant_id'].str.contains(r'_S\d+$', na=False)]
# print("After filtering for probands and siblings:", len(df))

# Keep only participants aged 5 to 18
age_col = 'eeg_age_years_testdate'
df = df[(df[age_col] >= 5) & (df[age_col] <= 18)]
print(f"After filtering for age 5-18: {len(df)} participants")

# Save the dataframe after age filtering
df.to_csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/filtered_after_age.csv", index=False)
print("Saved filtered dataframe after age filtering to 'filtered_after_age.csv'")

# # Ensure participants have at least one diagnosis (at least one 1 in any diagnosis column)
# diagnosis_cols = ['ASD', 'ASD_behavior', 'ADHD'] # 'OCD', 'motor_disorder','anxiety', 'neurological_conditions', 'genetic_disorder', 'other'
# df = df[df[diagnosis_cols].sum(axis=1) >= 1]
# print("After filtering for at least one diagnosis (sum of diagnosis columns >= 1):", len(df))

# ------------------------------------------------------------
# 3. Handle missing values based on configuration
# ------------------------------------------------------------
if USE_MEDIAN_IMPUTATION:
    # Use median imputation if participant has at most 1/4 missing, otherwise drop
    max_missing_allowed = 1
    missing_counts = df[behavioral_vars].isnull().sum(axis=1)
    eligible_for_imputation = missing_counts <= max_missing_allowed

    print(f"\nUsing MEDIAN IMPUTATION for participants with at most {max_missing_allowed} missing variables (out of {len(behavioral_vars)})")
    print(f"Participants eligible for imputation: {eligible_for_imputation.sum()} / {len(df)}")
    print(f"Dropping participants with >{max_missing_allowed} missing variables: {(~eligible_for_imputation).sum()}")

    # Keep participant_id for eligible participants
    df_eligible = df.loc[eligible_for_imputation, ['participant_id'] + behavioral_vars].copy()

    # Impute missing values with median for eligible participants (only behavioral_vars)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(df_eligible[behavioral_vars])
    df_processed = df_eligible.copy()
    df_processed[behavioral_vars] = X_imputed

    print(f"Imputed {df_eligible[behavioral_vars].isnull().sum().sum()} missing values")
    
else:
    print(f"\nUsing COMPLETE CASES ONLY (dropping participants with missing data)")
    
    # Use only complete cases, but keep participant_id
    df_complete_cases_mask = df[behavioral_vars].notnull().all(axis=1)
    df_processed = df.loc[df_complete_cases_mask, ['participant_id'] + behavioral_vars].copy()
    
    # Print IDs of participants that were dropped due to missing data
    dropped_participants = df.index.difference(df_processed.index)
    if len(dropped_participants) > 0:
        print("\nParticipants dropped due to missing data (complete case analysis):")
        print(df.loc[dropped_participants, 'participant_id'].tolist())
    else:
        print("\nNo participants were dropped due to missing data (complete case analysis).")
    
    print(f"Using {len(df_processed)} participants with complete data")

# Save the dataframe with only full data participants (complete cases) to CSV
df_complete_cases = df[behavioral_vars].dropna()
df_complete_cases_full = df.loc[df_complete_cases.index]
df_complete_cases_full.to_csv("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/complete_cases_only.csv", index=False)
print("Saved dataframe with only full data participants (complete cases) to 'complete_cases_only.csv'")

# ----------------------------------
# 4. Quick check on sex distribution
# ----------------------------------
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

# ----------------------------------------------
# 5. Standardize the data for k-means clustering
# ----------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(df_processed[behavioral_vars])

print(f"Data shape after preprocessing: {X.shape}")
print(f"Any NaN values remaining? {np.isnan(X).any()}")

# -------------------------
# 6. K-means clustering
# -------------------------
sil_scores = []
K_range = range(2, 6)  # Test 2-5 clusters

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    sil = silhouette_score(X, labels)
    sil_scores.append(sil)

# -------------------------
# 7. Plots
# -------------------------

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(K_range, sil_scores, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
method_name = "Median Imputation" if USE_MEDIAN_IMPUTATION else "Complete Cases"
plt.title(f"K-means: choosing K (n={len(df_processed)}, {method_name})")
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
df_processed_with_clusters = df_processed.copy()
df_processed_with_clusters['cluster'] = cluster_labels

# Show cluster sizes
unique, counts = np.unique(cluster_labels, return_counts=True)
print(f"\nNumber of participants per cluster:")
for cl, cnt in zip(unique, counts):
    print(f"  Cluster {cl}: {cnt} participants")

# -------------------------------------------------
# 9. Analyze cluster characteristics
# -------------------------------------------------
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

# Age distribution per cluster
if 'eeg_age_years_testdate' in df.columns:
    # Try to get the original age column from the original dataframe using the index of the processed df
    ages = df.loc[df_processed_with_clusters.index, 'eeg_age_years_testdate']
    df_processed_with_clusters['age'] = ages

    print("\nAge distribution per cluster:")
    for cl in sorted(df_processed_with_clusters['cluster'].unique()):
        cluster_ages = df_processed_with_clusters[df_processed_with_clusters['cluster'] == cl]['age'].dropna()
        print(f"  Cluster {cl}: n={len(cluster_ages)}, mean={cluster_ages.mean():.2f}, std={cluster_ages.std():.2f}, min={cluster_ages.min():.2f}, max={cluster_ages.max():.2f}")

    # Plot age distribution per cluster
    plt.figure(figsize=(8, 5))
    for cl in sorted(df_processed_with_clusters['cluster'].unique()):
        cluster_ages = df_processed_with_clusters[df_processed_with_clusters['cluster'] == cl]['age'].dropna()
        plt.hist(cluster_ages, bins=15, alpha=0.5, label=f'Cluster {cl}')
    plt.xlabel('Age (years)')
    plt.ylabel('Count')
    plt.title('Age Distribution per Cluster')
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No 'eeg_age_years_testdate' column found in the original dataframe for age distribution analysis.")

# Sex distribution per cluster
if 'sex' in df.columns:
    # Try to get the original sex column from the original dataframe using the index of the processed df
    sexes = df.loc[df_processed_with_clusters.index, 'sex']
    df_processed_with_clusters['sex'] = sexes

    print("\nSex distribution per cluster:")
    for cl in sorted(df_processed_with_clusters['cluster'].unique()):
        cluster_sexes = df_processed_with_clusters[df_processed_with_clusters['cluster'] == cl]['sex'].dropna()
        value_counts = cluster_sexes.value_counts()
        total = value_counts.sum()
        print(f"  Cluster {cl}:")
        for sex_value, count in value_counts.items():
            percent = 100 * count / total if total > 0 else 0
            print(f"    {sex_value}: {count} ({percent:.1f}%)")
else:
    print("No 'sex' column found in the original dataframe for sex distribution analysis.")

# Diagnosis distribution per cluster
diagnosis_cols = ['ASD', 'ASD_behavior', 'ADHD']
for diag_col in diagnosis_cols:
    if diag_col in df.columns:
        # Get the original diagnosis column from the original dataframe using the index of the processed df
        diagnoses = df.loc[df_processed_with_clusters.index, diag_col]
        df_processed_with_clusters[diag_col] = diagnoses

# Print diagnosis distribution per cluster for each diagnosis column
for diag_col in diagnosis_cols:
    if diag_col in df_processed_with_clusters.columns:
        print(f"\n{diag_col} distribution per cluster:")
        for cl in sorted(df_processed_with_clusters['cluster'].unique()):
            cluster_diag = df_processed_with_clusters[df_processed_with_clusters['cluster'] == cl][diag_col].dropna()
            value_counts = cluster_diag.value_counts()
            total = value_counts.sum()
            print(f"  Cluster {cl}:")
            for diag_value, count in value_counts.items():
                percent = 100 * count / total if total > 0 else 0
                print(f"    {diag_value}: {count} ({percent:.1f}%)")

# Analyze IQ distribution per cluster
if 'IQ' in df.columns:
    # Add IQ column to df_processed_with_clusters if not already present
    if 'IQ' not in df_processed_with_clusters.columns:
        iq_values = df.loc[df_processed_with_clusters.index, 'IQ']
        df_processed_with_clusters['IQ'] = iq_values
    print("\nIQ distribution per cluster:")
    for cl in sorted(df_processed_with_clusters['cluster'].unique()):
        cluster_iq = df_processed_with_clusters[df_processed_with_clusters['cluster'] == cl]['IQ'].dropna()
        if len(cluster_iq) > 0:
            print(f"  Cluster {cl}: n={len(cluster_iq)}, mean={cluster_iq.mean():.2f}, std={cluster_iq.std():.2f}, min={cluster_iq.min():.2f}, max={cluster_iq.max():.2f}")
        else:
            print(f"  Cluster {cl}: No IQ data available.")
else:
    print("No 'IQ' column found in the original dataframe for IQ distribution analysis.")

# Analyze household_income, highest_education_level, family_ethnicity per cluster

# Define the columns to analyze
demographic_cols = ['household_income', 'highest_education_level', 'family_ethnicity']

for col in demographic_cols:
    if col in df.columns:
        # Add the column to df_processed_with_clusters if not already present
        if col not in df_processed_with_clusters.columns:
            values = df.loc[df_processed_with_clusters.index, col]
            df_processed_with_clusters[col] = values

for col in demographic_cols:
    if col in df_processed_with_clusters.columns:
        print(f"\n{col} distribution per cluster:")
        cluster_data = []
        for cl in sorted(df_processed_with_clusters['cluster'].unique()):
            cluster_vals = df_processed_with_clusters[df_processed_with_clusters['cluster'] == cl][col].dropna()
            value_counts = cluster_vals.value_counts()
            total = value_counts.sum()
            print(f"  Cluster {cl}:")
            for val, count in value_counts.items():
                percent = 100 * count / total if total > 0 else 0
                print(f"    {val}: {count} ({percent:.1f}%)")
                # For plotting
                cluster_data.append({'Cluster': cl, col: val, 'Count': count})
        # Plotting
        if cluster_data:
            plot_df = pd.DataFrame(cluster_data)
            plt.figure(figsize=(10, 5))
            sns.barplot(
                data=plot_df,
                x='Cluster',
                y='Count',
                hue=col,
                palette='tab10'
            )
            plt.title(f"{col.replace('_', ' ').title()} Distribution per Cluster")
            plt.ylabel("Count")
            plt.xlabel("Cluster")
            plt.legend(title=col.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
    else:
        print(f"No '{col}' column found in the original dataframe for analysis.")

# # Analyze if have "hg_version" or "single_gene_test = yes"

# # Use columns from df_complete, and match participants based on participant_id

# # Check if either 'hg_version' or 'single_gene_test' columns exist in df_complete
# has_hg_version = 'hg_version' in df_complete.columns
# has_single_gene_test = 'single_gene_test' in df_complete.columns

# if has_hg_version or has_single_gene_test:
#     print("\nAnalysis of CNV and single gene testing per cluster (using df_complete, matched by participant_id):")

#     # Ensure participant_id is present in both dataframes
#     if 'participant_id' in df_processed_with_clusters.columns and 'participant_id' in df_complete.columns:
#         # Set index for fast lookup
#         df_complete_indexed = df_complete.set_index('participant_id', drop=False)

#         if has_hg_version:
#             # Add 'hg_version' to df_processed_with_clusters if not already present
#             if 'hg_version' not in df_processed_with_clusters.columns:
#                 hg_values = df_processed_with_clusters['participant_id'].map(
#                     df_complete_indexed['hg_version']
#                 )
#                 df_processed_with_clusters['hg_version'] = hg_values

#             print("\nCNV testing (hg_version not empty) per cluster:")
#             for cl in sorted(df_processed_with_clusters['cluster'].unique()):
#                 cluster_hg = df_processed_with_clusters[df_processed_with_clusters['cluster'] == cl]['hg_version']
#                 total = cluster_hg.notnull().sum()
#                 cnv_tested = cluster_hg.notnull() & (cluster_hg.astype(str).str.strip() != "")
#                 cnv_count = cnv_tested.sum()
#                 percent = 100 * cnv_count / total if total > 0 else 0
#                 print(f"  Cluster {cl}: {cnv_count} participants had CNV testing ({percent:.1f}%) out of {total} with data")
#                 if total == 0:
#                     print("    No 'hg_version' data available.")

#         if has_single_gene_test:
#             # Add 'single_gene_test' to df_processed_with_clusters if not already present
#             if 'single_gene_test' not in df_processed_with_clusters.columns:
#                 sgt_values = df_processed_with_clusters['participant_id'].map(
#                     df_complete_indexed['single_gene_test']
#                 )
#                 df_processed_with_clusters['single_gene_test'] = sgt_values

#             print("\nSingle gene testing ('single_gene_test = yes') per cluster:")
#             for cl in sorted(df_processed_with_clusters['cluster'].unique()):
#                 cluster_sgt = df_processed_with_clusters[df_processed_with_clusters['cluster'] == cl]['single_gene_test'].dropna()
#                 yes_count = (cluster_sgt.astype(str).str.lower().str.strip() == 'yes').sum()
#                 total = cluster_sgt.notnull().sum()
#                 percent = 100 * yes_count / total if total > 0 else 0
#                 print(f"  Cluster {cl}: {yes_count} participants had single gene testing ('yes') ({percent:.1f}%) out of {total} with data")
#                 if total == 0:
#                     print("    No 'single_gene_test' data available.")
#     else:
#         print("participant_id column missing in one of the dataframes; cannot match participants for CNV/single gene test analysis.")
# else:
#     print("\nNo 'hg_version' or 'single_gene_test' columns found in df_complete for analysis.")

# # Analyze the "Genes" column for overlap per cluster using df_complete, matched by participant_id

# if 'Genes' in df_complete.columns:
#     print("\nAnalysis of gene overlap per cluster (using df_complete, matched by participant_id):")

#     import matplotlib.pyplot as plt

#     if 'participant_id' in df_processed_with_clusters.columns and 'participant_id' in df_complete.columns:
#         # Set index for fast lookup
#         df_complete_indexed = df_complete.set_index('participant_id', drop=False)

#         # Add 'Genes' to df_processed_with_clusters if not already present
#         if 'Genes' not in df_processed_with_clusters.columns:
#             genes_values = df_processed_with_clusters['participant_id'].map(
#                 df_complete_indexed['Genes']
#             )
#             df_processed_with_clusters['Genes'] = genes_values

#         # Prepare a dictionary to store sets of genes per cluster
#         cluster_genes = {}
#         for cl in sorted(df_processed_with_clusters['cluster'].unique()):
#             # Get the 'Genes' column for this cluster, drop missing
#             genes_series = df_processed_with_clusters[df_processed_with_clusters['cluster'] == cl]['Genes'].dropna()
#             genes_set = set()
#             for genes_str in genes_series:
#                 # Split by comma, semicolon, or whitespace, strip spaces
#                 if isinstance(genes_str, str):
#                     for part in genes_str.replace(';', ',').split(','):
#                         gene = part.strip()
#                         if gene:
#                             genes_set.add(gene)
#             cluster_genes[cl] = genes_set
#             print(f"  Cluster {cl}: {len(genes_set)} unique genes")

#         # For each cluster, find genes that are shared between at least two participants in that cluster
#         print("\nGenes shared between participants within each cluster:")
#         from collections import Counter
#         import pandas as pd

#         clusters = sorted(cluster_genes.keys())
#         # For table/graph: collect (cluster, gene, count) for genes with count >= 2
#         shared_genes_records = []

#         for cl in clusters:
#             # Get all genes for each participant in this cluster
#             cluster_df = df_processed_with_clusters[df_processed_with_clusters['cluster'] == cl]
#             # Build a list of sets, each set is the genes for one participant
#             participant_genes = []
#             for genes_str in cluster_df['Genes'].dropna():
#                 if isinstance(genes_str, str):
#                     genes = set(g.strip() for g in genes_str.replace(';', ',').split(',') if g.strip())
#                     if genes:
#                         participant_genes.append(genes)
#             # Count gene occurrences across participants
#             gene_counter = Counter()
#             for genes in participant_genes:
#                 gene_counter.update(genes)
#             # Find genes that are present in at least two participants in this cluster
#             shared_genes = [(gene, count) for gene, count in gene_counter.items() if count >= 2]
#             print(f"  Cluster {cl}: {len(shared_genes)} genes shared by at least two participants")
#             if shared_genes:
#                 print(f"    Shared genes: {[gene for gene, count in shared_genes]}")
#                 for gene, count in shared_genes:
#                     shared_genes_records.append({'Cluster': cl, 'Gene': gene, 'Count': count})
#             else:
#                 print("    No genes shared by more than one participant in this cluster.")

#         # Create a DataFrame for shared genes (count >= 2)
#         if shared_genes_records:
#             shared_genes_df = pd.DataFrame(shared_genes_records)
#             print("\nTable of genes shared by at least two participants (per cluster):")
#             print(shared_genes_df.sort_values(['Cluster', 'Count', 'Gene'], ascending=[True, False, True]).to_string(index=False))

#             # Only show genes with count >= 3 in the plot (as per instruction: "those over 2 with same genes only")
#             plot_df = shared_genes_df[shared_genes_df['Count'] > 2]
#             if not plot_df.empty:
#                 plt.figure(figsize=(12, 6))
#                 # Sort for better visualization
#                 plot_df = plot_df.sort_values(['Count', 'Gene'], ascending=[False, True])
#                 # Create a bar plot: x = gene (with cluster), y = count
#                 plot_df['Gene (Cluster)'] = plot_df['Gene'] + " (C" + plot_df['Cluster'].astype(str) + ")"
#                 plt.bar(plot_df['Gene (Cluster)'], plot_df['Count'], color='skyblue')
#                 plt.xticks(rotation=90)
#                 plt.ylabel('Number of Participants')
#                 plt.title('Genes shared by more than 2 participants (per cluster)')
#                 plt.tight_layout()
#                 plt.show()
#             else:
#                 print("\nNo genes are shared by more than 2 participants in any cluster (no bar plot shown).")
#         else:
#             print("\nNo genes are shared by at least two participants in any cluster (no table or plot to show).")
#     else:
#         print("participant_id column missing in one of the dataframes; cannot match participants for gene overlap analysis.")
# else:
#     print("\nNo 'Genes' column found in df_complete for analysis.")


# -------------------------
# 10. VISUALIZATIONS
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
key_vars = ['SRS_social_communication_tscore', 'attention_deficit_hyperactivity_tscore']
for i, var in enumerate(key_vars):
    plt.subplot(2, 2, 3 + i)
    df_processed_with_clusters.boxplot(column=var, by='cluster', ax=plt.gca())
    plt.title(f'{var} by Cluster')
    plt.suptitle('')  # Remove automatic suptitle

plt.tight_layout()
plt.show()

# -------------------------
# 11. SAVE RESULTS
# -------------------------
# Save the clustered data
method_suffix = "median_imputed" if USE_MEDIAN_IMPUTATION else "complete_cases"
output_file = f"/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/Cluster Analysis/proband_clusters_kmeans_{method_suffix}.csv"
df_processed_with_clusters.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)