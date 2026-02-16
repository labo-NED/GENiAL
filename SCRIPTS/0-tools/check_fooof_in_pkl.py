### CHECK FOOOF FEATURES IN PKL FILES
# This script checks if FOOOF features (offset and exponent) are stored in the pkl files
# and compares them with CSV files to diagnose the NaN issue
#
# Usage: python check_fooof_in_pkl.py [pkl_file_path]

import pickle
import numpy as np
import sys
import os
import glob

def check_pkl_file(pkl_path):
    """Check if FOOOF features exist in a pkl file"""
    print(f"\n{'='*60}")
    print(f"Checking: {os.path.basename(pkl_path)}")
    print(f"{'='*60}")
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\nKeys in pkl file: {list(data.keys())}")
        
        # Check for FOOOF features
        fooof_features = ['fooof_offset', 'fooof_exponent']
        found_features = {}
        
        for feature in fooof_features:
            if feature in data:
                feature_data = data[feature]
                print(f"\n{feature}:")
                print(f"  Type: {type(feature_data)}")
                
                if isinstance(feature_data, np.ndarray):
                    print(f"  Shape: {feature_data.shape}")
                    print(f"  Dtype: {feature_data.dtype}")
                    print(f"  Has NaN: {np.isnan(feature_data).any()}")
                    print(f"  All NaN: {np.isnan(feature_data).all()}")
                    if feature_data.size > 0:
                        print(f"  Min: {np.nanmin(feature_data):.6e}")
                        print(f"  Max: {np.nanmax(feature_data):.6e}")
                        print(f"  Mean: {np.nanmean(feature_data):.6e}")
                    found_features[feature] = feature_data
                elif isinstance(feature_data, (list, tuple)):
                    print(f"  Length: {len(feature_data)}")
                    arr = np.array(feature_data)
                    print(f"  Converted to array shape: {arr.shape}")
                    print(f"  Has NaN: {np.isnan(arr).any()}")
                    if arr.size > 0:
                        print(f"  Min: {np.nanmin(arr):.6e}")
                        print(f"  Max: {np.nanmax(arr):.6e}")
                        print(f"  Mean: {np.nanmean(arr):.6e}")
                    found_features[feature] = arr
                else:
                    print(f"  Value: {feature_data}")
                    found_features[feature] = feature_data
            else:
                print(f"\n{feature}: NOT FOUND in pkl file")
        
        # Check other metadata
        if 'channel_list' in data:
            print(f"\nchannel_list: {len(data['channel_list'])} channels")
        if 'n_epochs' in data:
            print(f"n_epochs: {data['n_epochs']}")
        if 'n_channels' in data:
            print(f"n_channels: {data['n_channels']}")
        
        # Summary
        print(f"\n{'='*60}")
        if len(found_features) == 2:
            print("✅ Both FOOOF features found in pkl file")
            if all(isinstance(v, np.ndarray) for v in found_features.values()):
                if all(not np.isnan(v).all() for v in found_features.values()):
                    print("✅ FOOOF features contain valid (non-NaN) data")
                    return True
                else:
                    print("⚠️  FOOOF features contain some NaN values")
                    return False
            else:
                print("⚠️  FOOOF features are not numpy arrays")
                return False
        else:
            print("❌ FOOOF features missing or incomplete in pkl file")
            return False
            
    except Exception as e:
        print(f"❌ Error loading pkl file: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) > 1:
        # Check specific file
        pkl_path = sys.argv[1]
        if os.path.exists(pkl_path):
            check_pkl_file(pkl_path)
        else:
            print(f"Error: File not found: {pkl_path}")
    else:
        # Check all pkl files in default directory
        print("No file specified. Searching for pkl files...")
        
        # Try to find pkl files in common locations
        search_dirs = [
            '/Volumes/LaCie/Q1K-EMMA/Q1K_BC_HAPPEv3_ICA/2s_epochs/features',
            '/home/emmacona/links/projects/def-lippes/emmacona/Q1K_BC_HAPPEv3_ICA/Features',
        ]
        
        pkl_files = []
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                pattern = os.path.join(search_dir, 'features_*.pkl')
                found = glob.glob(pattern)
                pkl_files.extend(found)
                print(f"Found {len(found)} pkl files in {search_dir}")
        
        if len(pkl_files) == 0:
            print("No pkl files found. Please specify a pkl file path as argument.")
            print("\nUsage: python check_fooof_in_pkl.py <pkl_file_path>")
            return
        
        # Check first few files
        print(f"\nChecking first 5 pkl files...")
        for pkl_file in pkl_files[:5]:
            check_pkl_file(pkl_file)


if __name__ == "__main__":
    main()
