### FIND UNTRANSFERRED EEG FILES AND MOVE THEM (GENiAL PROJECT)
# This script identifies files from the complete list that haven't been transferred yet
# and moves them to a retry folder for re-transfer
#
# Emmanuelle Coutu-Nadeau (Nov 2025)

import os
import shutil
from pathlib import Path

# Input paths
all_files_list = '/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/DATA/all_2s_eeg_list.txt'
transferred_files_list = '/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/DATA/transferred_eeg_list.txt'

# Output paths
missing_files_output = '/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/DATA/untransferred_eeg_list.txt'

# Source and destination directories (UPDATE THESE PATHS AS NEEDED)
source_dir = '/Volumes/NED_Backup3/COMBINED_Q1K_BC_2s/curated_list_for_genial_2s'  # Where the 2s files are currently stored
retry_dir = '/Volumes/NED_Backup3/COMBINED_Q1K_BC_2s/curated_list_for_genial_2s/retry_transfer'  # Where to move untransferred files

print('='*60)
print('FINDING UNTRANSFERRED EEG FILES')
print('='*60)

# Read all files list
print(f'\nReading complete file list from: {all_files_list}')
with open(all_files_list, 'r') as f:
    # Parse the file - it appears to have two columns separated by tabs/spaces
    all_files = []
    for line in f:
        line = line.strip()
        if line:
            # Split by whitespace and get all .set files
            parts = line.split()
            for part in parts:
                if part.endswith('_processed_2s.set'):
                    all_files.append(part)

print(f'Total files in complete list: {len(all_files)}')

# Read transferred files list
print(f'\nReading transferred files list from: {transferred_files_list}')
with open(transferred_files_list, 'r') as f:
    transferred_files = []
    for line in f:
        line = line.strip()
        if line:
            # Split by whitespace and get all .set files
            parts = line.split()
            for part in parts:
                if part.endswith('_processed_2s.set'):
                    transferred_files.append(part)

print(f'Total files already transferred: {len(transferred_files)}')

# Find untransferred files
transferred_set = set(transferred_files)
untransferred = [f for f in all_files if f not in transferred_set]
untransferred.sort()

print(f'\n{"="*60}')
print(f'FOUND {len(untransferred)} UNTRANSFERRED FILES')
print(f'{"="*60}')

# Save list of untransferred files
with open(missing_files_output, 'w') as f:
    for filename in untransferred:
        f.write(filename + '\n')

print(f'\n✅ List saved to: {missing_files_output}')

# Show first 10 untransferred files
if untransferred:
    print(f'\nFirst 10 untransferred files:')
    for i, filename in enumerate(untransferred[:10], 1):
        print(f'  {i}. {filename}')
    if len(untransferred) > 10:
        print(f'  ... and {len(untransferred) - 10} more')

# Move files automatically
print(f'\n{"="*60}')
print('MOVING FILES TO RETRY FOLDER')
print(f'{"="*60}')
print(f'Source directory: {source_dir}')
print(f'Destination directory: {retry_dir}')
print(f'\nMoving {len(untransferred)} files...')

# Create retry directory if it doesn't exist
os.makedirs(retry_dir, exist_ok=True)
print(f'\n✅ Created/verified retry directory: {retry_dir}')

# Move files
moved_count = 0
not_found_count = 0
error_count = 0

for i, filename in enumerate(untransferred, 1):
    source_path = os.path.join(source_dir, filename)
    dest_path = os.path.join(retry_dir, filename)
    
    if os.path.exists(source_path):
        try:
            shutil.move(source_path, dest_path)
            moved_count += 1
            if i % 10 == 0:
                print(f'  Progress: {i}/{len(untransferred)} files processed...')
        except Exception as e:
            print(f'  ❌ Error moving {filename}: {e}')
            error_count += 1
    else:
        print(f'  ⚠️  File not found: {filename}')
        not_found_count += 1

# Summary
print(f'\n{"="*60}')
print('MOVE OPERATION SUMMARY')
print(f'{"="*60}')
print(f'Successfully moved: {moved_count} files')
print(f'Not found in source: {not_found_count} files')
print(f'Errors: {error_count} files')
print(f'{"="*60}\n')

