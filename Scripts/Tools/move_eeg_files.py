#!/usr/bin/env python3
"""
Script to identify and move EEG files based on participant IDs from CSV.

This script:
1. Reads participant IDs from FINAL_DATABASE_USED_IN_R.csv
2. Scans EEG files in /Volumes/NED_Backup3/COMBINED_Q1K_BC_2s
3. Matches files to participant IDs based on naming conventions:
   - Q1K files: everything before _RSRio (case insensitive)
   - BC files: BC_ + number after BC_2017_
4. Moves matching .set files to GENIAL subfolder
"""

import os
import shutil
import pandas as pd
import re
from pathlib import Path


def load_participant_ids(csv_path):
    """Load participant IDs from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        participant_ids = set(df['participant_id'].tolist())
        print(f"Loaded {len(participant_ids)} participant IDs from CSV")
        return participant_ids
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return set()


def extract_participant_id_from_filename(filename):
    """
    Extract participant ID from EEG filename.
    
    For Q1K files: everything before _RSRio (case insensitive)
    For BC files: BC_ + number after BC_2017_
    """
    # Handle Q1K files
    if filename.lower().startswith('q1k_'):
        # Find _rsrio pattern (case insensitive)
        rsrio_match = re.search(r'_rsrio', filename.lower())
        if rsrio_match:
            # Get everything before _rsrio, preserving original case
            rsrio_pos = rsrio_match.start()
            participant_part = filename[:rsrio_pos]
            return participant_part
    
    # Handle BC files
    elif filename.lower().startswith('bc_2017_'):
        # Extract number after BC_2017_
        match = re.match(r'bc_2017_(\d+)', filename.lower())
        if match:
            number = match.group(1)
            return f"BC_{number}"
    
    return None


def find_matching_files(source_dir, participant_ids):
    """Find EEG files that match participant IDs."""
    matching_files = []
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Source directory does not exist: {source_dir}")
        return matching_files
    
    print(f"Scanning directory: {source_dir}")
    
    # Get all .set files
    set_files = list(source_path.glob("*.set"))
    print(f"Found {len(set_files)} .set files")
    
    for file_path in set_files:
        filename = file_path.name
        extracted_id = extract_participant_id_from_filename(filename)
        
        if extracted_id and extracted_id in participant_ids:
            matching_files.append(file_path)
            print(f"✓ Match found: {filename} -> {extracted_id}")
        else:
            if extracted_id:
                print(f"✗ No match: {filename} -> {extracted_id}")
            else:
                print(f"✗ Could not extract ID from: {filename}")
    
    return matching_files


def move_files_to_genial(files, source_dir):
    """Move files to GENIAL subfolder."""
    source_path = Path(source_dir)
    genial_path = source_path / "GENIAL"
    
    # Create GENIAL directory if it doesn't exist
    genial_path.mkdir(exist_ok=True)
    print(f"Created/verified GENIAL directory: {genial_path}")
    
    moved_count = 0
    for file_path in files:
        try:
            destination = genial_path / file_path.name
            shutil.move(str(file_path), str(destination))
            print(f"Moved: {file_path.name} -> GENIAL/{file_path.name}")
            moved_count += 1
        except Exception as e:
            print(f"Error moving {file_path.name}: {e}")
    
    print(f"Successfully moved {moved_count} files to GENIAL subfolder")
    return moved_count


def main():
    """Main function to orchestrate the file moving process."""
    # Define paths
    csv_path = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/FINAL_DATABASE_USED_IN_R.csv"
    source_dir = "/Volumes/NED_Backup3/COMBINED_Q1K_BC_2s"
    
    print("=== EEG File Mover ===")
    print(f"CSV file: {csv_path}")
    print(f"Source directory: {source_dir}")
    print()
    
    # Load participant IDs
    participant_ids = load_participant_ids(csv_path)
    if not participant_ids:
        print("No participant IDs loaded. Exiting.")
        return
    
    # Find matching files
    matching_files = find_matching_files(source_dir, participant_ids)
    
    if not matching_files:
        print("No matching files found.")
        return
    
    print(f"\nFound {len(matching_files)} matching files to move:")
    for file_path in matching_files:
        print(f"  - {file_path.name}")
    
    # Ask for confirmation
    response = input(f"\nMove {len(matching_files)} files to GENIAL subfolder? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Operation cancelled.")
        return
    
    # Move files
    moved_count = move_files_to_genial(matching_files, source_dir)
    print(f"\n=== Operation Complete ===")
    print(f"Moved {moved_count} files to GENIAL subfolder")


if __name__ == "__main__":
    main()
