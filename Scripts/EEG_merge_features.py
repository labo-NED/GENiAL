import pandas as pd
import numpy as np

# File paths
eeg_features_path = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/EEG/GENIAL/RS-2s/features_global_summary.csv"
behavioral_data_path = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/final_ASD_ADHD_gfmm_cluster_db.csv"
output_path = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/merged_EEG_behavioral_data.csv"
# eeg_features_path = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/EEG/GENIAL/RS-2s/alpha_peaks.csv"
# behavioral_data_path = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/merged_EEG_behavioral_data.csv"
# output_path = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/merged_EEG_behavioral_data_V2.csv"

print("Loading data files...")

# Load EEG features
eeg_df = pd.read_csv(eeg_features_path)
print(f"EEG features loaded: {len(eeg_df)} participants")
print(f"EEG columns: {list(eeg_df.columns)}")

# Load behavioral data
behavioral_df = pd.read_csv(behavioral_data_path)
print(f"Behavioral data loaded: {len(behavioral_df)} participants")
print(f"Behavioral columns: {list(behavioral_df.columns)}")

# Display unique SubjectIDs to understand the matching
print(f"\nUnique EEG SubjectIDs (first 10): {eeg_df['SubjectID'].unique()[:10]}")
print(f"Unique Behavioral participant_ids (first 10): {behavioral_df['participant_id'].unique()[:10]}")

# Check for exact matches
eeg_subjects = set(eeg_df['SubjectID'].unique())
behavioral_subjects = set(behavioral_df['participant_id'].unique())

exact_matches = eeg_subjects.intersection(behavioral_subjects)
print(f"\nExact matches found: {len(exact_matches)}")
print(f"Exact matches: {list(exact_matches)[:10]}")

# # Check for partial matches (in case there are slight differences)
# print(f"\nChecking for partial matches...")
# partial_matches = []
# for eeg_id in eeg_subjects:
#     for behavioral_id in behavioral_subjects:
#         if eeg_id in behavioral_id or behavioral_id in eeg_id:
#             partial_matches.append((eeg_id, behavioral_id))

# print(f"Partial matches found: {len(partial_matches)}")
# if partial_matches:
#     print("Partial matches (first 10):")
#     for match in partial_matches[:10]:
#         print(f"  EEG: {match[0]} <-> Behavioral: {match[1]}")

# Perform the merge
print(f"\nPerforming merge...")
merged_df = pd.merge(
    behavioral_df, 
    eeg_df, 
    left_on='participant_id', 
    right_on='SubjectID', 
    how='left'  # Keep all behavioral participants, add EEG data where available
)

print(f"Merged dataset: {len(merged_df)} participants")
print(f"Participants with EEG data: {merged_df['SubjectID'].notna().sum()}")
print(f"Participants without EEG data: {merged_df['SubjectID'].isna().sum()}")

# Display summary of merged data
print(f"\nMerged dataset columns: {len(merged_df.columns)}")
print("Column names:")
for i, col in enumerate(merged_df.columns):
    print(f"  {i+1:2d}. {col}")

# Check for missing EEG data by cluster
if 'cluster' in merged_df.columns:
    print(f"\nMissing EEG data by cluster:")
    missing_by_cluster = merged_df.groupby('cluster')['SubjectID'].apply(lambda x: x.isna().sum())
    total_by_cluster = merged_df.groupby('cluster').size()
    print("Cluster | Missing EEG | Total | % Missing")
    print("-" * 40)
    for cluster in missing_by_cluster.index:
        missing = missing_by_cluster[cluster]
        total = total_by_cluster[cluster]
        pct = (missing / total) * 100 if total > 0 else 0
        print(f"   {cluster}   |     {missing:2d}     |  {total:2d}  |   {pct:5.1f}%")

# Save merged data
print(f"\nSaving merged data to: {output_path}")
merged_df.to_csv(output_path, index=False)

# Create a summary of the merge
summary_stats = {
    'Total_participants': len(merged_df),
    'Participants_with_EEG': merged_df['SubjectID'].notna().sum(),
    'Participants_without_EEG': merged_df['SubjectID'].isna().sum(),
    'EEG_coverage_percent': (merged_df['SubjectID'].notna().sum() / len(merged_df)) * 100,
    'Total_columns': len(merged_df.columns),
    'Behavioral_columns': len(behavioral_df.columns),
    'EEG_columns': len(eeg_df.columns) - 2  # Subtract SubjectID and Task
}

print(f"\n=== MERGE SUMMARY ===")
for key, value in summary_stats.items():
    if isinstance(value, float):
        print(f"{key}: {value:.1f}")
    else:
        print(f"{key}: {value}")

print(f"\n✅ Merge completed successfully!")
print(f"📁 Output saved to: {output_path}")
