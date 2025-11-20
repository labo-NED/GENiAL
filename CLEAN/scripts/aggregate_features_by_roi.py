### AGGREGATE EEG FEATURES BY ROI (GENiAL PROJECT)
# This script reads individual feature CSV files and aggregates them by ROI
# Output: Single CSV with one row per participant, columns for each feature x ROI combination
#
# Emmanuelle Coutu-Nadeau (Nov 2025)

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# ------------ Paths ------------
root_dir = '/home/emmacona/projects/def-lippes/emmacona'
features_dir = os.path.join(root_dir, 'Q1K_BC_EEG_features')
output_file = os.path.join(root_dir, 'Q1K_BC_EEG_features', 'Q1K_BC_aggregated_EEG_features_by_roi.csv')

# ------------ ROI Mapping ------------
# Map ROI names to electrode numbers (converting E3 -> 3, E124 -> 124, etc.)
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
    Extract participant ID from filename.
    
    Handles two formats:
    - Q1K: Q1K_HSJ_1525_1009_M1_2s.csv -> Q1K_HSJ_1525-1009_M1
    - Brain Canada: BC_2019_12345_A_processed_2s.csv -> BC_12345_A
    
    Returns: (participant_id, epoch_type)
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
    
    # Q1K format: handle:
    #   - Q1K_HSJ_1525_1009_M1  --> Q1K_HSJ_1525-1009_M1
    #   - Q1K_HSJ_1525-1009_M1  (already correct, just use as is)
    #   - Q1K_HSJ_1525_M1       --> Q1K_HSJ_1525_M1
    if base_name.startswith('Q1K_'):
        if '-' in base_name:
            # Already "Q1K_HSJ_1525-1009_M1" format, use as is
            participant_id = base_name
        else:
            parts = base_name.split('_')
            if len(parts) >= 4 and parts[2].isdigit() and parts[3].startswith('M') is False and parts[3].startswith('F') is False:
                # Q1K_HSJ_1525_1009_M1 --> Q1K_HSJ_1525-1009_M1
                participant_id = f"{parts[0]}_{parts[1]}_{parts[2]}-{parts[3]}"
                if len(parts) > 4:
                    participant_id += '_' + '_'.join(parts[4:])
            else:
                # e.g., Q1K_HSJ_1525_M1, just use as is
                participant_id = base_name
    
    # Brain Canada format: BC_YYYY_XXXX_ID_LETTER_... -> BC_ID_LETTER
    elif base_name.startswith('BC_'):
        parts = base_name.split('_')
        if len(parts) >= 4:
            # Extract: BC_YYYY_XXXX_[ID]_[LETTER]
            participant_id = f"BC_{parts[3]}_{parts[4]}"
            # Handle additional suffixes like S1, S2
            if len(parts) > 4 and parts[4]:
                participant_id += '_' + parts[4]
    
    # Fallback: use the whole base_name
    if participant_id is None:
        participant_id = base_name
    
    return participant_id, epoch_type


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
    
    # Convert channel column to string for matching
    df['channel'] = df['channel'].astype(str)
    
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
        
        # Extract participant ID and epoch type
        participant_id, epoch_type = extract_participant_id(filename)
        
        if participant_id is None:
            print(f"  Warning: Could not extract participant ID, skipping")
            continue
        
        print(f"  Participant: {participant_id}, Epoch type: {epoch_type}")
        
        # Load and aggregate features
        aggregated = load_and_aggregate_features(csv_file, rois)
        
        if aggregated is None or len(aggregated) == 0:
            print(f"  Warning: No features aggregated, skipping")
            continue
        
        # Add metadata
        row_data = {
            'participant_id': participant_id,
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
    
    # Sort by participant_id and epoch_type
    df_all = df_all.sort_values(['participant_id', 'epoch_type'])
    
    # Pivot to create one row per participant
    # This creates columns like: feature_2s_F, feature_5s_F, etc.
    
    # Pivot the data to have one row per participant
    # First, add epoch_type suffix to all feature columns
    feature_cols = [col for col in df_all.columns if col not in ['participant_id', 'epoch_type', 'source_file']]
    
    # Create a new dataframe with renamed columns including epoch type
    df_pivot = df_all.copy()
    for col in feature_cols:
        df_pivot[col] = df_pivot.apply(lambda row: (f"{col}_{row['epoch_type']}", row[col]), axis=1)
    
    # Pivot to wide format
    df_wide = df_pivot.groupby('participant_id').first().reset_index()
    
    # Flatten any remaining structures
    df_all = df_wide
    
    # Save to CSV
    df_all.to_csv(output_file, index=False)
    
    print("="*60)
    print(f"✅ SUCCESS!")
    print(f"Aggregated data for {len(df_all)} participant-epoch combinations")
    print(f"Total columns: {len(df_all.columns)}")
    print(f"\nColumn breakdown:")
    print(f"  - Metadata: participant_id, epoch_type, source_file")
    print(f"  - Features x ROIs: {len(df_all.columns) - 3}")
    print(f"\nOutput saved to: {output_file}")
    print("="*60)
    
    # Show sample of the data
    print("\nFirst few rows:")
    print(df_all.head())
    
    # Show summary statistics
    print(f"\nParticipants with 2s data: {len(df_all[df_all['epoch_type'] == '2s'])}")
    print(f"Participants with 5s data: {len(df_all[df_all['epoch_type'] == '5s'])}")
    print(f"\nUnique participants: {df_all['participant_id'].nunique()}")


if __name__ == "__main__":
    main()

