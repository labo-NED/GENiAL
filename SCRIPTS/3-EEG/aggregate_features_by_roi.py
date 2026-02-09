### AGGREGATE EEG FEATURES BY ROI (GENiAL PROJECT)
# This script reads individual feature CSV files and aggregates them by ROI
# Output: Single CSV with one row per participant-condition, columns for each feature x ROI combination
# Note: 2s and 5s epoch data are merged horizontally (complementary features combined)
#       Features are suffixed with _2s or _5s to distinguish them
#
# Emmanuelle Coutu-Nadeau (Nov 2025)

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# # ------------ Paths ------------
# root_dir = '/home/emmacona/projects/def-lippes/emmacona'
# features_dir = os.path.join('/scratch/emmacona/Q1K_BC_EEG_features')
# output_file = os.path.join(root_dir, 'Q1K_BC_EEG_features', 'Q1K_BC_aggregated_EEG_features_by_roi.csv')

# Locally
root_dir = '/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/DATA/'
features_dir = os.path.join(root_dir, 'OUTPUTS/eeg_features')
output_file = os.path.join(root_dir, 'OUTPUTS/eeg_features/Q1K_BC_aggregated_EEG_features_by_roi.csv')


# ------------ ROI Mapping ------------
# Map ROI names to electrode numbers as strings
# Channels in CSV files are converted from float to int to string (e.g., 1.0 -> '1', 124.0 -> '124')
rois = {
    'F': ['3', '4', '9', '11', '16', '19', '22', '23', '124', '24'],  # Frontal
    'Cz': ['6', '13', '30', '37', '87', '105', '112'],  # Central
    'TR': ['103', '104', '111', '116', '117', '122', '123'],  # Temporal Right
    'TL': ['27', '28', '29', '33', '34', '36', '41'],  # Temporal Left
    'PR': ['92', '96', '97', '98', '102', '108', '100', '95'],  # Parietal Right
    'PL': ['45', '46', '47', '51', '52', '58', '57', '64'],  # Parietal Left
    'Oz': ['60', '67', '72', '77', '85', '70', '75', '83']  # Occipital
}

# ------------ Helper Functions ------------

def extract_participant_id(filename):
    """
    Extract participant ID, condition, and epoch type from filename.
    
    Handles two formats:
    - Q1K: Q1K_HSJ_1525_1192_P_RSRio_20250307_011113_5s.csv 
           -> (Q1K_HSJ_1525-1192_P, RSRio, 5s)
    - Brain Canada: BC_2017_82437_889488_S1_V1_2s.csv 
                    -> (BC_889488_S1, None, 2s)
    
    Returns: (participant_id, condition, epoch_type)
    """
    # Remove path and extensions
    base_name = os.path.basename(filename)
    base_name = base_name.replace('features_avg_', '').replace('.csv', '')
    
    # Extract epoch type (2s or 5s)
    epoch_type = None
    if base_name.endswith('_2s'):
        epoch_type = '2s'
        base_name = base_name[:-3]  # Remove _2s
    elif base_name.endswith('_5s'):
        epoch_type = '5s'
        base_name = base_name[:-3]  # Remove _5s
    
    # Remove _processed suffix if present
    base_name = base_name.replace('_processed', '')
    
    participant_id = None
    condition = None
    
    # Q1K format: Q1K_HSJ_1525_1192_P_RSRio_YYYYMMDD_HHMMSS
    if base_name.startswith('Q1K_'):
        parts = base_name.split('_')
        
        # Find RS/RSRio/RSRIO in the parts
        condition_idx = None
        for i, part in enumerate(parts):
            if part.upper() in ['RS', 'RSRIO']:
                condition = 'RSRio' if part.upper() == 'RSRIO' else 'RS'
                condition_idx = i
                break
        
        # Build participant ID (everything before condition and timestamp)
        if condition_idx:
            # Get parts before condition (skip timestamp parts)
            id_parts = parts[:condition_idx]
        else:
            # No condition found, use the whole name (minus timestamps)
            # Assume last 2 parts are timestamp if they look like numbers
            if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
                id_parts = parts[:-2]
            else:
                id_parts = parts
        
        # Now build the participant_id from id_parts
        # Three formats to handle:
        # 1. Q1K_HSJ_1525-1045_S1 (already has dash)
        # 2. Q1K_HSJ_1525_1052_S1 (needs dash: two 4-digit IDs)
        # 3. Q1K_HSJ_100114_S2 (single ID, no dash)
        
        if len(id_parts) >= 3:
            # Check if part at index 2 already contains a dash
            if '-' in id_parts[2]:
                # Format 1: already has dash
                participant_id = '_'.join(id_parts)
            elif len(id_parts) >= 4 and id_parts[2].isdigit() and id_parts[3].isdigit():
                # Check if we have two separate numeric IDs (format 2)
                # Both should be 4 digits (like 1525 and 1052)
                if len(id_parts[2]) == 4 and len(id_parts[3]) == 4:
                    # Format 2: Q1K_HSJ_1525_1052 -> Q1K_HSJ_1525-1052
                    participant_id = f"{id_parts[0]}_{id_parts[1]}_{id_parts[2]}-{id_parts[3]}"
                    if len(id_parts) > 4:
                        participant_id += '_' + '_'.join(id_parts[4:])
                else:
                    # Format 3: single ID like Q1K_HSJ_100114_S2
                    participant_id = '_'.join(id_parts)
            else:
                # Format 3 or other: just join all parts
                participant_id = '_'.join(id_parts)
        else:
            participant_id = '_'.join(id_parts)
    
    # Brain Canada format: BC_2017_82437_889488_S1_V1
    elif base_name.startswith('BC_'):
        parts = base_name.split('_')
        if len(parts) >= 4:
            # BC_YYYY_XXXXX_ID_[SUFFIX...]
            # Use ID and any following parts (S1, S2, V1, etc.) as participant ID
            participant_id = f"BC_{parts[3]}"
            if len(parts) > 4:
                # Add suffixes like S1, V1, P, etc.
                participant_id += '_' + '_'.join(parts[4:])
            condition = 'RSRio'  # Brain Canada files are all RSRio
    # Fallback: use the whole base_name
    if participant_id is None:
        participant_id = base_name
    
    return participant_id, condition, epoch_type


def load_and_aggregate_features(csv_file, rois):
    """
    Load a feature CSV file and aggregate by ROI.
    
    Args:
        csv_file: Path to features_avg_*.csv file
        rois: Dictionary mapping ROI names to lists of channel IDs
    
    Returns:
        Dictionary with aggregated features by ROI
    """
    # Load CSV
    df = pd.read_csv(csv_file)
    
    # Make sure 'channel' column exists
    if 'channel' not in df.columns:
        print(f"Warning: No 'channel' column in {csv_file}")
        return None
    
    # Convert channel column: float -> int -> string
    # e.g., 1.0 -> 1 -> '1', 124.0 -> 124 -> '124'
    # Also handle 'E3' format if present
    try:
        # Try converting to float first (in case it's numeric), then to int, then to string
        df['channel'] = df['channel'].astype(float).astype(int).astype(str)
    except (ValueError, TypeError):
        # If that fails, assume it's already string format and strip 'E' prefix
        df['channel'] = df['channel'].astype(str).str.replace('^E', '', regex=True)
    
    # Get feature columns (all except 'channel')
    feature_cols = [col for col in df.columns if col != 'channel']
    
    aggregated = {}
    
    # For each ROI, average features across channels in that ROI
    for roi_name, channel_list in rois.items():
        # Filter rows for channels in this ROI
        roi_data = df[df['channel'].isin(channel_list)]
        
        if len(roi_data) == 0:
            print(f"Warning: No channels found for ROI {roi_name} in {csv_file}")
            continue
        
        # Average each feature across channels in this ROI
        for feature in feature_cols:
            col_name = f"{feature}_{roi_name}"
            aggregated[col_name] = roi_data[feature].mean()
    
    return aggregated


# ------------ Main Processing ------------

def main():
    print("="*60)
    print("EEG FEATURE AGGREGATION BY ROI")
    print("="*60)
    print(f"\nFeatures directory: {features_dir}")
    print(f"Output file: {output_file}\n")
    
    # Find all feature CSV files
    csv_pattern = os.path.join(features_dir, 'features_avg_*.csv')
    csv_files = glob.glob(csv_pattern)
    
    if len(csv_files) == 0:
        print(f"Error: No CSV files found matching pattern: {csv_pattern}")
        return
    
    print(f"Found {len(csv_files)} feature CSV files\n")
    
    # Process each file
    all_data = []
    
    for idx, csv_file in enumerate(csv_files, 1):
        filename = os.path.basename(csv_file)
        print(f"Processing {idx}/{len(csv_files)}: {filename}")
        
        # Extract participant ID, condition, and epoch type
        participant_id, condition, epoch_type = extract_participant_id(filename)
        
        if participant_id is None:
            print(f"  Warning: Could not extract participant ID, skipping")
            continue
        
        print(f"  Participant: {participant_id}, Condition: {condition}, Epoch type: {epoch_type}")
        
        # Load and aggregate features
        aggregated = load_and_aggregate_features(csv_file, rois)
        
        if aggregated is None or len(aggregated) == 0:
            print(f"  Warning: No features aggregated, skipping")
            continue
        
        # Add metadata
        row_data = {
            'participant_id': participant_id,
            'condition': condition,
            'epoch_type': epoch_type,
            'source_file': filename
        }
        row_data.update(aggregated)
        
        all_data.append(row_data)
        print(f"  Aggregated {len(aggregated)} feature x ROI combinations")
        print()
    
    # Convert to DataFrame
    if len(all_data) == 0:
        print("Error: No data to aggregate")
        return
    
    df_all = pd.DataFrame(all_data)
    
    # Sort by participant_id, condition, epoch_type
    sort_cols = ['participant_id']
    if 'condition' in df_all.columns:
        sort_cols.append('condition')
    sort_cols.append('epoch_type')
    
    df_all = df_all.sort_values(sort_cols)
    
    # Merge 2s and 5s data: combine features from both into one row per participant-condition
    # Since 2s and 5s files contain DIFFERENT complementary features, we merge them horizontally
    print("\nMerging 2s and 5s data (combining complementary features)...")
    
    # Get feature columns (exclude metadata)
    metadata_cols = ['participant_id', 'condition', 'epoch_type', 'source_file']
    feature_cols = [col for col in df_all.columns if col not in metadata_cols]
    
    # Add epoch_type suffix to feature columns to distinguish 2s vs 5s features
    # This creates columns like: alpha_power_F_2s, alpha_power_F_5s
    rows_with_suffix = []
    for _, row in df_all.iterrows():
        new_row = {
            'participant_id': row['participant_id'],
            'condition': row['condition'],
        }
        epoch_type = row['epoch_type']
        for col in feature_cols:
            new_row[f"{col}_{epoch_type}"] = row[col]
        rows_with_suffix.append(new_row)
    
    df_with_suffix = pd.DataFrame(rows_with_suffix)
    
    # Group by participant_id and condition, combining rows from different epoch types
    groupby_cols = ['participant_id', 'condition'] if 'condition' in df_with_suffix.columns else ['participant_id']
    
    # Aggregate: for each group, take the first non-null value for each column
    # (since 2s and 5s have different column names due to suffix, there's no overlap)
    df_merged = df_with_suffix.groupby(groupby_cols, dropna=False).first().reset_index()
    
    print(f"  Before merge: {len(df_all)} rows")
    print(f"  After merge: {len(df_merged)} rows")
    
    # Save to CSV (wide format: one row per participant-condition combination)
    df_merged.to_csv(output_file, index=False)
    
    print("="*60)
    print(f"✅ SUCCESS!")
    print(f"Aggregated data for {len(df_merged)} rows")
    print(f"Total columns: {len(df_merged.columns)}")
    print(f"\nColumn breakdown:")
    print(f"  - Metadata: participant_id, condition")
    print(f"  - Feature columns (includes _2s and _5s variants): {len(df_merged.columns) - 2}")
    print(f"\nOutput saved to: {output_file}")
    print("="*60)
    
    # Show sample of the data
    print("\nFirst few rows (metadata only):")
    metadata_cols = ['participant_id', 'condition']
    print(df_merged[metadata_cols].head(10))
    
    # Show summary statistics
    print(f"\nSummary Statistics:")
    print(f"  - Total rows: {len(df_merged)}")
    print(f"  - Unique participants: {df_merged['participant_id'].nunique()}")
    
    # Count how many have both 2s and 5s data
    cols_2s = [col for col in df_merged.columns if col.endswith('_2s')]
    cols_5s = [col for col in df_merged.columns if col.endswith('_5s')]
    has_2s = df_merged[cols_2s].notna().any(axis=1).sum() if cols_2s else 0
    has_5s = df_merged[cols_5s].notna().any(axis=1).sum() if cols_5s else 0
    print(f"  - Rows with 2s features: {has_2s}")
    print(f"  - Rows with 5s features: {has_5s}")
    
    if 'condition' in df_merged.columns and df_merged['condition'].notna().any():
        print(f"\nCondition breakdown:")
        print(df_merged['condition'].value_counts(dropna=False))


if __name__ == "__main__":
    main()

