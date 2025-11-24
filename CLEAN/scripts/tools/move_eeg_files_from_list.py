import os
import shutil
import argparse
from pathlib import Path

def load_subject_ids(id_list_file):
    """
    Load subject IDs or filenames from a text file.
    
    Args:
        id_list_file (str): Path to the file containing subject IDs or filenames (one per line)
    
    Returns:
        list: List of subject IDs or filenames (stripped of whitespace)
    """
    items = []
    with open(id_list_file, 'r') as f:
        for line in f:
            item = line.strip()
            if item:  # Skip empty lines
                items.append(item)
    return items

def move_files_from_list(source_dir, destination_dir, id_list_file, skip_rename=False, exact_filenames=False):
    """
    Move all files containing subject IDs from the list to a destination directory.
    
    Args:
        source_dir (str): Directory to search for files
        destination_dir (str): Directory to move files to
        id_list_file (str): Path to file containing subject IDs or exact filenames
        skip_rename (bool): If True, skip files that already exist instead of renaming them
        exact_filenames (bool): If True, treat list entries as exact filenames instead of IDs
    """
    source_path = Path(source_dir)
    dest_path = Path(destination_dir)
    
    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Load items from file
    if exact_filenames:
        print(f"Loading exact filenames from: {id_list_file}")
    else:
        print(f"Loading subject IDs from: {id_list_file}")
    
    items = load_subject_ids(id_list_file)
    
    if exact_filenames:
        print(f"Loaded {len(items)} filenames")
    else:
        print(f"Loaded {len(items)} subject IDs")
    print("-" * 50)
    
    # Store as subject_ids for backward compatibility
    subject_ids = items
    
    moved_items = []
    matching_files = []
    
    print(f"Searching for files in: {source_path}")
    print(f"Destination directory: {dest_path}")
    print("-" * 50)
    
    # Walk through the directory structure to find matching files
    for root, dirs, files in os.walk(str(source_path)):
        # Skip the destination directory if it's within source
        if str(dest_path) in root:
            continue
            
        for file_name in files:
            matched_id = None
            
            if exact_filenames:
                # Strategy: Exact filename match
                if file_name in subject_ids:
                    matched_id = file_name
            else:
                # Original ID-based matching strategies
                
                # Strategy 1 (Q1K): Extract ID before _rsrio/_RSRio/_RSRIO (case-insensitive)
                rsrio_pattern = None
                file_lower = file_name.lower()
                for pattern in ['_rsrio_', '_rsrio.']:
                    if pattern in file_lower:
                        idx = file_lower.find(pattern)
                        rsrio_pattern = file_name[:idx]  # Get everything before _rsrio
                        break
                
                if rsrio_pattern:
                    # For Q1K files, normalize underscores to dashes in the ID portion
                    # e.g., Q1K_HSJ_1525_1009_M1 -> Q1K_HSJ_1525-1009_M1
                    normalized_pattern = rsrio_pattern
                    if rsrio_pattern.startswith('Q1K_'):
                        # Split and rebuild: Q1K_HSJ_1525_1009_M1 -> parts
                        parts = rsrio_pattern.split('_')
                        if len(parts) >= 4:
                            # Rejoin with dashes for the numeric ID parts
                            # Q1K_HSJ_1525_1009 -> Q1K_HSJ_1525-1009
                            normalized_pattern = f"{parts[0]}_{parts[1]}_{parts[2]}-{parts[3]}"
                            if len(parts) > 4:
                                # Add any suffix like _M1, _F1, etc.
                                normalized_pattern += '_' + '_'.join(parts[4:])
                    
                    # Check for exact match first
                    for subject_id in subject_ids:
                        if subject_id == normalized_pattern or subject_id == rsrio_pattern:
                            matched_id = subject_id
                            break
                    
                    # If no exact match, check if any subject_id is a prefix of the normalized pattern
                    # e.g., Q1K_HSJ_1525-1009_P matches Q1K_HSJ_1525-1009_M1
                    if not matched_id:
                        for subject_id in subject_ids:
                            if (normalized_pattern.startswith(subject_id + '_') or 
                                normalized_pattern == subject_id or
                                rsrio_pattern.startswith(subject_id + '_') or 
                                rsrio_pattern == subject_id):
                                matched_id = subject_id
                                break
                
                # Strategy 2: Direct substring match (for files already moved or different naming)
                if not matched_id:
                    for subject_id in subject_ids:
                        if subject_id in file_name:
                            matched_id = subject_id
                            break
                
                # Strategy 3 (Brain Canada): Parse BC format: BC_YYYY_XXXX_ID_LETTER_...
                if not matched_id and file_name.startswith('BC_'):
                    parts = file_name.split('_')
                    if len(parts) >= 5:
                        # Extract: BC_[parts[2]]_[parts[4]]
                        try:
                            bc_id = f"BC_{parts[3]}_{parts[4]}"
                            # Check if this constructed ID matches any in the list
                            for subject_id in subject_ids:
                                # Match BC_ID_LETTER or BC_ID_LETTER+NUMBER (like S1, S2)
                                if subject_id.startswith(bc_id):
                                    matched_id = subject_id
                                    break
                        except:
                            pass
            
            if matched_id:
                matching_files.append((root, file_name, matched_id))
    
    total_files_found = len(matching_files)
    if total_files_found == 0:
        if exact_filenames:
            print("No matching files found for the provided filenames.")
        else:
            print("No matching files found for the provided subject IDs.")
        print("\nDebug: Showing sample files in directory...")
        
        # Debug: show some directory contents
        for root, dirs, files in os.walk(str(source_path)):
            if str(dest_path) in root:
                continue
            print(f"\nIn directory: {root}")
            sample_files = files[:5]
            if sample_files:
                print(f"  Sample files:")
                for f in sample_files:
                    print(f"    - {f}")
            if len(files) > 0:
                break  # Just show the first level with files
    else:
        if exact_filenames:
            print(f"Found {total_files_found} files matching provided filenames")
        else:
            print(f"Found {total_files_found} files matching subject IDs")
        print()
    
    # Move the files
    skipped_files = []
    for idx, (root, file_name, subject_id) in enumerate(matching_files, 1):
        source_file = Path(root) / file_name
        dest_file = dest_path / file_name
        
        # Check if file already exists
        if dest_file.exists():
            if skip_rename:
                # Skip this file without moving
                print(f"Skipping {idx}/{total_files_found}: {file_name} (matched: {subject_id})")
                print(f"  ⏭️  File already exists, skipping (--skip-rename enabled)")
                skipped_files.append(file_name)
                print()
                continue
            else:
                # Handle duplicate names by adding a number
                counter = 1
                while dest_file.exists():
                    name, ext = os.path.splitext(file_name)
                    dest_file = dest_path / f"{name}_{counter}{ext}"
                    counter += 1
                print(f"Moving {idx}/{total_files_found}: {file_name} (matched: {subject_id})")
                print(f"  ⚠️  File exists, renamed to: {dest_file.name}")
        else:
            print(f"Moving {idx}/{total_files_found}: {file_name} (matched: {subject_id})")
        
        try:
            # Move the file
            shutil.move(str(source_file), str(dest_file))
            moved_items.append(dest_file.name)
            print(f"  ✓ Moved successfully")
        except Exception as e:
            print(f"  ❌ Failed to move: {e}")
        print()
    
    print("-" * 50)
    print(f"Operation completed.")
    print(f"Total matching files found: {total_files_found}")
    print(f"Successfully moved: {len(moved_items)} files")
    
    if skipped_files:
        print(f"Skipped (already exist): {len(skipped_files)} files")
    
    # Report on items that had no matches
    matched_ids = set(subject_id for _, _, subject_id in matching_files)
    unmatched_ids = set(subject_ids) - matched_ids
    if unmatched_ids:
        if exact_filenames:
            print(f"\n⚠️  {len(unmatched_ids)} filenames had no matching files:")
        else:
            print(f"\n⚠️  {len(unmatched_ids)} subject IDs had no matching files:")
        for unmatched_id in sorted(unmatched_ids)[:10]:  # Show first 10
            print(f"  - {unmatched_id}")
        if len(unmatched_ids) > 10:
            print(f"  ... and {len(unmatched_ids) - 10} more")
    
    return moved_items

def main(source_directory, destination_directory, id_list_file, skip_rename=False, exact_filenames=False):
    """
    Main function to run the file moving operation.
    
    Args:
        source_directory (str): Path to the source directory to search for files
        destination_directory (str): Path to the destination directory where files will be moved
        id_list_file (str): Path to the file containing subject IDs or exact filenames
        skip_rename (bool): If True, skip files that already exist instead of renaming them
        exact_filenames (bool): If True, treat list entries as exact filenames instead of IDs
    """
    print("=== EEG File Moving Tool ===")
    print(f"Source: {source_directory}")
    print(f"Destination: {destination_directory}")
    print(f"List file: {id_list_file}")
    if exact_filenames:
        print(f"Match mode: Exact filenames")
    else:
        print(f"Match mode: Subject ID patterns")
    if skip_rename:
        print(f"Duplicate mode: Skip existing files (no renaming)")
    else:
        print(f"Duplicate mode: Rename duplicates")
    print("=" * 60)
    
    try:
        moved_items = move_files_from_list(source_directory, destination_directory, id_list_file, skip_rename, exact_filenames)
        print(f"\n✅ Operation completed successfully!")
        print(f"📁 Total files moved: {len(moved_items)}")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # EXAMPLE COMMAND:
    # python3 CLEAN/scripts/move_eeg_files_from_list.py "/Volumes/NED_Backup3/COMBINED_Q1K_BC_2s" "/Volumes/NED_Backup3/COMBINED_Q1K_BC_2s/curated_list_for_genial" "CLEAN/subject_id_list.txt"
    # python3 CLEAN/scripts/move_eeg_files_from_list.py "/Volumes/NED_Backup3/COMBINED_Q1K_BC_5s" "/Volumes/NED_Backup3/COMBINED_Q1K_BC_5s/curated_list_for_genial" "CLEAN/subject_id_list.txt"


    parser = argparse.ArgumentParser(description='Move files matching subject IDs or exact filenames from a list')
    parser.add_argument('source_directory', help='Path to the source directory to search for files')
    parser.add_argument('destination_directory', help='Path to the destination directory where files will be moved')
    parser.add_argument('id_list_file', help='Path to the file containing subject IDs or exact filenames (one per line)')
    parser.add_argument('--skip-rename', action='store_true', 
                        help='Skip files that already exist in destination instead of renaming them')
    parser.add_argument('--exact-filenames', action='store_true',
                        help='Treat list entries as exact filenames instead of subject ID patterns')
    
    args = parser.parse_args()
    main(args.source_directory, args.destination_directory, args.id_list_file, args.skip_rename, args.exact_filenames)

