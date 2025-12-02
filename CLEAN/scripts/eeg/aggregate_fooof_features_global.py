### AGGREGATE FOOOF APERIODIC FEATURES GLOBALLY (GENiAL PROJECT)
# This script reads individual FOOOF feature PKL files and aggregates them by averaging across all channels
# Output: Single CSV with one row per participant-condition, columns for fooof_offset and fooof_exponent (averaged globally)
#
# Based on aggregate_features_global.py
# Emmanuelle Coutu-Nadeau (Dec 2025)

import os
import glob
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ------------ Paths ------------
# Toggle between local and cluster
IS_LOCAL = False  # Set to True for local runs, False for cluster runs

if IS_LOCAL:
    root_dir = '/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/'
    features_dir = os.path.join(root_dir, 'Outputs/eeg_fooof_aperiodic')
    output_file = os.path.join(root_dir, 'Outputs/eeg_fooof_aperiodic/Q1K_BC_aggregated_FOOOF_features_global.csv')
else:
    root_dir = '/home/emmacona/links/projects/def-lippes/emmacona'
    features_dir = os.path.join(root_dir, 'Q1K_BC_HAPPEv3_ICA/Features')
    output_file = os.path.join(features_dir, 'Q1K_BC_aggregated_FOOOF_features_global.csv')


# ------------ Helper Functions ------------

def extract_participant_id(filename):
    """
    Extract participant ID, condition, and epoch type from filename.
    
    Handles two formats:
    - Q1K: Q1K_HSJ_1525_1192_P_RSRio_20250307_011113.csv 
           -> (Q1K_HSJ_1525-1192_P, RSRio)
    - Brain Canada: BC_2017_82437_889488_S1_V1.csv 
                    -> (BC_889488_S1_V1, RSRio)
    
    Returns: (participant_id, condition)
    """
    # Remove path and extensions
    base_name = os.path.basename(filename)
    base_name = base_name.replace('fooof_aperiodic_', '').replace('.pkl', '').replace('.csv', '')
    
    # Remove _2s or _5s suffix if present (FOOOF files are typically from 2s epochs)
    if base_name.endswith('_2s'):
        base_name = base_name[:-3]
    elif base_name.endswith('_5s'):
        base_name = base_name[:-3]
    
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
            condition = 'RSRio'  # All Brain Canada files are RSRio
    
    # Fallback: use the whole base_name
    if participant_id is None:
        participant_id = base_name
    
    return participant_id, condition


def load_and_aggregate_fooof_features(pkl_file):
    """
    Load a FOOOF feature PKL file and aggregate by averaging across all channels.
    
    Args:
        pkl_file: Path to fooof_aperiodic_*.pkl file
    
    Returns:
        Dictionary with aggregated FOOOF features (offset and exponent, averaged globally)
    """
    # Load PKL
    try:
        with open(pkl_file, 'rb') as f:
            results = pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load {pkl_file}: {e}")
        return None
    
    # Make sure required keys exist
    if 'fooof_offset' not in results or 'fooof_exponent' not in results:
        print(f"Warning: Missing fooof_offset or fooof_exponent in {pkl_file}")
        return None
    
    # Extract offset and exponent arrays
    offset = results['fooof_offset']
    exponent = results['fooof_exponent']
    
    # Convert to numpy arrays if not already
    offset = np.array(offset)
    exponent = np.array(exponent)
    
    # Aggregate FOOOF features by averaging across all channels
    aggregated = {
        'fooof_offset_mean': offset.mean(),
        'fooof_exponent_mean': exponent.mean(),
        'fooof_offset_std': offset.std(),
        'fooof_exponent_std': exponent.std(),
        'fooof_offset_median': np.median(offset),
        'fooof_exponent_median': np.median(exponent),
        'n_channels': len(offset),
        'n_epochs': results.get('n_epochs', np.nan)  # Store number of epochs too
    }
    
    return aggregated


# ------------ Main Processing ------------

def main():
    print("="*60)
    print("FOOOF APERIODIC FEATURE AGGREGATION (GLOBAL AVERAGE)")
    print("="*60)
    print(f"\nFeatures directory: {features_dir}")
    print(f"Output file: {output_file}\n")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Find all FOOOF feature PKL files
    pkl_pattern = os.path.join(features_dir, 'fooof_aperiodic_*.pkl')
    pkl_files = glob.glob(pkl_pattern)
    
    if len(pkl_files) == 0:
        print(f"Error: No PKL files found matching pattern: {pkl_pattern}")
        return
    
    print(f"Found {len(pkl_files)} FOOOF feature PKL files\n")
    
    # Process each file
    all_data = []
    skipped_count = 0
    
    for idx, pkl_file in enumerate(pkl_files, 1):
        filename = os.path.basename(pkl_file)
        print(f"Processing {idx}/{len(pkl_files)}: {filename}")
        
        # Extract participant ID and condition
        participant_id, condition = extract_participant_id(filename)
        
        if participant_id is None:
            print(f"  Warning: Could not extract participant ID, skipping")
            skipped_count += 1
            continue
        
        print(f"  Participant: {participant_id}, Condition: {condition}")
        
        # Load and aggregate FOOOF features
        aggregated = load_and_aggregate_fooof_features(pkl_file)
        
        if aggregated is None or len(aggregated) == 0:
            print(f"  Warning: No features aggregated, skipping")
            skipped_count += 1
            continue
        
        # Add metadata
        row_data = {
            'participant_id': participant_id,
            'condition': condition,
            'source_file': filename
        }
        row_data.update(aggregated)
        
        all_data.append(row_data)
        print(f"  Aggregated: offset_mean={aggregated['fooof_offset_mean']:.4f}, "
              f"exponent_mean={aggregated['fooof_exponent_mean']:.4f}")
        print()
    
    # Convert to DataFrame
    if len(all_data) == 0:
        print("Error: No data to aggregate")
        return
    
    df_all = pd.DataFrame(all_data)
    
    # Sort by participant_id and condition
    sort_cols = ['participant_id']
    if 'condition' in df_all.columns:
        sort_cols.append('condition')
    
    df_all = df_all.sort_values(sort_cols)
    
    # Save to CSV
    df_all.to_csv(output_file, index=False)
    
    print("="*60)
    print(f"✅ SUCCESS!")
    print(f"Aggregated data for {len(df_all)} participants")
    print(f"Skipped: {skipped_count} files")
    print(f"Total columns: {len(df_all.columns)}")
    print(f"\nOutput saved to: {output_file}")
    print("="*60)
    
    # Show sample of the data
    print("\nFirst 10 rows:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df_all.head(10).to_string())
    
    # Show summary statistics
    print(f"\n\nSummary Statistics:")
    print(f"  - Total rows: {len(df_all)}")
    print(f"  - Unique participants: {df_all['participant_id'].nunique()}")
    
    if 'condition' in df_all.columns and df_all['condition'].notna().any():
        print(f"\nCondition breakdown:")
        print(df_all['condition'].value_counts(dropna=False))
    
    print("\nFOOOF Parameter Ranges:")
    print(f"  Offset (mean):   [{df_all['fooof_offset_mean'].min():.4f}, {df_all['fooof_offset_mean'].max():.4f}]")
    print(f"  Exponent (mean): [{df_all['fooof_exponent_mean'].min():.4f}, {df_all['fooof_exponent_mean'].max():.4f}]")
    
    print("\nFOOOF Parameter Statistics:")
    print(df_all[['fooof_offset_mean', 'fooof_exponent_mean', 
                   'fooof_offset_std', 'fooof_exponent_std']].describe())


if __name__ == "__main__":
    main()

