#!/usr/bin/env python3
"""
Merge EEG features with clustered behavioral data.
Filters EEG features for RSRio condition only and merges on participant_id.
"""

import pandas as pd
import os
from datetime import datetime

# Input file paths
TIMESTAMP = "FEB_18_2026"
EEG_FEATURES_FILE = "/Volumes/LaCie/Q1K-EMMA/Q1K_BC_HAPPEv3_ICA/2s_epochs/features/Q1K_BC_aggregated_EEG_features_global.csv" # 2s file
# FOOOF_FEATURES_FILE = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/DATA/OUTPUTS/eeg_features/Q1K_BC_aggregated_FOOOF_features_global.csv"
ORIGNIAL_DATA_FILE = "/Volumes/LaCie/Q1K-EMMA/Database/clustered_EEG_features_global_RSRio_FEB_18_2026.csv"

# Output directory
OUTPUT_DIR = "/Volumes/LaCie/Q1K-EMMA/Database"

def main():
    print("=" * 80)
    print("EEG Features and Behavioral Data Merger")
    print("=" * 80)
    
    # --- Merge Behavioral Data with EEG features ----
    # Read the EEG features file
    print(f"\n1. Reading EEG features from:\n   {EEG_FEATURES_FILE}")
    eeg_df = pd.read_csv(EEG_FEATURES_FILE)
    print(f"   - Total rows: {len(eeg_df)}")
    print(f"   - Columns: {len(eeg_df.columns)}")
    print(f"   - Conditions found: {eeg_df['condition'].unique()}")
    
    # Filter for RSRio condition only
    print(f"\n2. Filtering for condition = 'RSRio'")
    eeg_rsrio = eeg_df[eeg_df['condition'] == 'RSRio'].copy()
    print(f"   - Rows after filtering: {len(eeg_rsrio)}")
    print(f"   - Unique participants: {eeg_rsrio['participant_id'].nunique()}")
    
    # Drop the condition column as it's now all RSRio
    eeg_rsrio = eeg_rsrio.drop(columns=['condition'])
    print(f"   - Dropped 'condition' column (all RSRio)")
    
    # Read the behavioral data file
    print(f"\n3. Reading clustered data from:\n   {ORIGNIAL_DATA_FILE}")
    original_df = pd.read_csv(ORIGNIAL_DATA_FILE)
    print(f"   - Total rows: {len(original_df)}")
    print(f"   - Columns: {len(original_df.columns)}")
    print(f"   - Unique participants: {original_df['participant_id'].nunique()}")
    
    # Merge the dataframes on participant_id
    print(f"\n4. Merging datasets on 'participant_id'")
    merged_df = original_df.merge(eeg_rsrio, on='participant_id', how='left')
    print(f"   - Merged rows: {len(merged_df)}")
    print(f"   - Total columns: {len(merged_df.columns)}")

    # # ---- Merge fooof features ----
    # print(f"\n5. Merging fooof features on 'participant_id'")
    # fooof_df = pd.read_csv(FOOOF_FEATURES_FILE)

    # # Filter for RSRio condition only
    # fooof_eeg_rsrio = fooof_df[fooof_df['condition'] == 'RSRio'].copy()
    # print(f"   - Rows after filtering fooof features: {len(fooof_eeg_rsrio)}")
    # print(f"   - Unique participants with fooof features: {fooof_eeg_rsrio['participant_id'].nunique()}")
    
    # # Keep only fooof columns
    # fooof_eeg_rsrio = fooof_eeg_rsrio.drop(columns=['condition', 'source_file', 'n_channels','n_epochs'])

    # # Merge fooof features with merged data (behavioral & eeg)
    # merged_df = merged_df.merge(fooof_eeg_rsrio, on='participant_id', how='left')
    # print(f"   - Merged rows with fooof features: {len(merged_df)}")
    # print(f"   - Total columns: {len(merged_df.columns)}")
    
    # Check how many participants have EEG data
    eeg_columns = [col for col in eeg_rsrio.columns if col != 'participant_id']
    eeg_columns = eeg_columns + [col for col in eeg_rsrio.columns if col != 'participant_id']
    participants_with_eeg = merged_df[eeg_columns[0]].notna().sum()
    print(f"   - Participants with EEG features: {participants_with_eeg}/{len(merged_df)}")
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%b_%d_%Y").upper()
    output_filename = f"merged_clustered_behavioral_EEG_features_global_RSRio_{timestamp}.csv"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Save the merged dataframe
    print(f"\n6. Saving merged data to:\n   {output_path}")
    merged_df.to_csv(output_path, index=False)
    print(f"   ✓ File saved successfully!")
    
    # Print summary statistics
    print(f"\n" + "=" * 80)
    print("MERGE SUMMARY")
    print("=" * 80)
    print(f"Input files:")
    print(f"  - EEG features (RSRio only): {len(eeg_rsrio)} rows, {len(eeg_rsrio.columns)} columns")
    print(f"  - Original data: {len(original_df)} rows, {len(original_df.columns)} columns")
    print(f"\nOutput file:")
    print(f"  - Merged data: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    print(f"  - Location: {output_path}")
    print(f"\nData completeness:")
    print(f"  - Participants with EEG data: {participants_with_eeg} ({participants_with_eeg/len(merged_df)*100:.1f}%)")
    print(f"  - Participants without EEG data: {len(merged_df) - participants_with_eeg} ({(len(merged_df) - participants_with_eeg)/len(merged_df)*100:.1f}%)")
    print("=" * 80)

if __name__ == "__main__":
    main()

