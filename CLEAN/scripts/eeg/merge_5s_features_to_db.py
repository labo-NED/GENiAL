#!/usr/bin/env python3
"""
Merge 5s EEG features into the main merged file.
Merges 5s features (higuchi_fd_5s, katz_fd_5s, samp_entropy_5s, CI_5s, CI_lowscale_5s, CI_highscale_5s)
from two source files into the main merged file.
"""

import pandas as pd
import numpy as np
import os

# Input file paths
TIMESTAMP = "DEC_16_2025"
MAIN_FILE = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/merged_clustered_behavioral_EEG_features_global_RSRio_DEC_12_2025.csv"
EEG_5S_FILE_1 = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/eeg_features/Q1K_BC_aggregated_EEG_5s_features_global_DEC_16_2025.csv"
EEG_5S_FILE_2 = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/eeg_features/Q1K_BC_aggregated_EEG_features_global.csv"

# 5s features to merge
FEATURES_5S = ['higuchi_fd_5s', 'katz_fd_5s', 'samp_entropy_5s', 'CI_5s', 'CI_lowscale_5s', 'CI_highscale_5s']

# Output directory
OUTPUT_DIR = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs"

def main():
    print("=" * 80)
    print("Merge 5s EEG Features into Main File")
    print("=" * 80)
    
    # Read the main file (contains all 2s features and behavioral data)
    print(f"\n1. Reading main file from:\n   {MAIN_FILE}")
    main_df = pd.read_csv(MAIN_FILE)
    print(f"   - Total rows: {len(main_df)}")
    print(f"   - Total columns: {len(main_df.columns)}")
    print(f"   - Unique participants: {main_df['participant_id'].nunique()}")
    
    # Check if 5s features already exist
    existing_5s_features = [f for f in FEATURES_5S if f in main_df.columns]
    if existing_5s_features:
        print(f"   - Warning: Some 5s features already exist: {existing_5s_features}")
        print(f"   - These will be overwritten with new data")
    
    # Read first 5s features file
    print(f"\n2. Reading 5s features from file 1:\n   {EEG_5S_FILE_1}")
    eeg_5s_df_1 = pd.read_csv(EEG_5S_FILE_1)
    print(f"   - Total rows: {len(eeg_5s_df_1)}")
    print(f"   - Columns: {list(eeg_5s_df_1.columns)}")
    
    # Filter for RSRio if condition column exists
    if 'condition' in eeg_5s_df_1.columns:
        print(f"   - Conditions found: {eeg_5s_df_1['condition'].unique()}")
        eeg_5s_df_1 = eeg_5s_df_1[eeg_5s_df_1['condition'] == 'RSRio'].copy()
        eeg_5s_df_1 = eeg_5s_df_1.drop(columns=['condition'])
        print(f"   - Rows after filtering for RSRio: {len(eeg_5s_df_1)}")
    
    # Extract only participant_id and 5s features
    available_features_1 = [f for f in FEATURES_5S if f in eeg_5s_df_1.columns]
    eeg_5s_1 = eeg_5s_df_1[['participant_id'] + available_features_1].copy()
    print(f"   - Available 5s features: {available_features_1}")
    
    # Read second 5s features file
    print(f"\n3. Reading 5s features from file 2:\n   {EEG_5S_FILE_2}")
    eeg_5s_df_2 = pd.read_csv(EEG_5S_FILE_2)
    print(f"   - Total rows: {len(eeg_5s_df_2)}")
    print(f"   - Columns: {list(eeg_5s_df_2.columns)}")
    
    # Filter for RSRio if condition column exists
    if 'condition' in eeg_5s_df_2.columns:
        print(f"   - Conditions found: {eeg_5s_df_2['condition'].unique()}")
        eeg_5s_df_2 = eeg_5s_df_2[eeg_5s_df_2['condition'] == 'RSRio'].copy()
        eeg_5s_df_2 = eeg_5s_df_2.drop(columns=['condition'])
        print(f"   - Rows after filtering for RSRio: {len(eeg_5s_df_2)}")
    
    # Extract only participant_id and 5s features
    available_features_2 = [f for f in FEATURES_5S if f in eeg_5s_df_2.columns]
    eeg_5s_2 = eeg_5s_df_2[['participant_id'] + available_features_2].copy()
    print(f"   - Available 5s features: {available_features_2}")
    
    # Combine both 5s feature files (file 1 takes priority for overlapping participants)
    print(f"\n4. Combining 5s features from both files")
    # Start with file 1
    eeg_5s_combined = eeg_5s_1.copy()
    participants_in_1 = set(eeg_5s_1['participant_id'])
    
    # Get all features that should be in the combined dataset
    all_features = list(set(available_features_1 + available_features_2))
    
    # For participants in both files, file 1 takes priority
    # For participants only in file 2, add them
    eeg_5s_2_new = eeg_5s_2[~eeg_5s_2['participant_id'].isin(participants_in_1)].copy()
    
    if len(eeg_5s_2_new) > 0:
        print(f"   - Adding {len(eeg_5s_2_new)} participants from file 2")
        # Add new participants from file 2
        eeg_5s_combined = pd.concat([eeg_5s_combined, eeg_5s_2_new], ignore_index=True)
    
    # For participants in both files, fill missing features from file 2
    participants_in_both = participants_in_1.intersection(set(eeg_5s_2['participant_id']))
    if len(participants_in_both) > 0:
        print(f"   - {len(participants_in_both)} participants in both files (file 1 takes priority)")
        # Fill missing features from file 2 for participants in both
        for feature in all_features:
            if feature in eeg_5s_2.columns:
                # Get values from file 2 for participants in both
                file2_values = eeg_5s_2.set_index('participant_id')[feature]
                # Fill only where the value is missing in combined
                if feature in eeg_5s_combined.columns:
                    mask = (eeg_5s_combined['participant_id'].isin(participants_in_both)) & \
                           (eeg_5s_combined[feature].isna())
                else:
                    # Feature doesn't exist yet, add it for participants in both
                    eeg_5s_combined[feature] = np.nan
                    mask = eeg_5s_combined['participant_id'].isin(participants_in_both)
                
                if mask.any():
                    eeg_5s_combined.loc[mask, feature] = eeg_5s_combined.loc[mask, 'participant_id'].map(file2_values)
    
    # Ensure all expected features are present (as NaN if missing)
    for feature in FEATURES_5S:
        if feature not in eeg_5s_combined.columns:
            eeg_5s_combined[feature] = np.nan
    
    # Keep only participant_id and the 5s features
    eeg_5s_combined = eeg_5s_combined[['participant_id'] + FEATURES_5S].copy()
    
    print(f"   - Combined 5s features: {len(eeg_5s_combined)} participants")
    print(f"   - Features available: {[f for f in FEATURES_5S if f in eeg_5s_combined.columns]}")
    
    # Merge 5s features into main file
    print(f"\n5. Merging 5s features into main file on 'participant_id'")
    merged_df = main_df.merge(eeg_5s_combined, on='participant_id', how='left')
    print(f"   - Merged rows: {len(merged_df)}")
    print(f"   - Total columns: {len(merged_df.columns)}")
    
    # Check how many participants have 5s features
    participants_with_5s = merged_df[FEATURES_5S[0]].notna().sum() if FEATURES_5S[0] in merged_df.columns else 0
    print(f"   - Participants with 5s features: {participants_with_5s}/{len(merged_df)}")
    
    # Generate output filename with DEC_16 timestamp
    output_filename = f"merged_clustered_behavioral_EEG_features_global_RSRio_{TIMESTAMP}.csv"
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
    print(f"  - Main file: {len(main_df)} rows, {len(main_df.columns)} columns")
    print(f"  - 5s features file 1: {len(eeg_5s_1)} rows")
    print(f"  - 5s features file 2: {len(eeg_5s_2)} rows")
    print(f"\nOutput file:")
    print(f"  - Merged data: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    print(f"  - Location: {output_path}")
    print(f"\nData completeness:")
    print(f"  - Participants with 5s features: {participants_with_5s} ({participants_with_5s/len(merged_df)*100:.1f}%)")
    print(f"  - Participants without 5s features: {len(merged_df) - participants_with_5s} ({(len(merged_df) - participants_with_5s)/len(merged_df)*100:.1f}%)")
    print(f"\n5s features merged:")
    for feature in FEATURES_5S:
        status = "✓" if feature in merged_df.columns else "✗"
        non_null = merged_df[feature].notna().sum() if feature in merged_df.columns else 0
        print(f"  {status} {feature}: {non_null} participants")
    print("=" * 80)

if __name__ == "__main__":
    main()

