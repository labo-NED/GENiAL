import os
import shutil
import argparse
from pathlib import Path

def move_all_files(source_dir, destination_dir):
    """
    Move all files from source directory (including subdirectories) to a single destination directory.
    
    Args:
        source_dir (str): Directory to search for files
        destination_dir (str): Directory to move files to
    """
    source_path = Path(source_dir)
    dest_path = Path(destination_dir)
    
    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    moved_items = []
    all_files = []

    print(f"Starting search for all files in: {source_path}")
    print(f"Destination directory: {dest_path}")
    print("-" * 50)
    
    # Walk through the directory structure to find all files
    for root, dirs, files in os.walk(str(source_path)):
        # Skip the destination directory if it's within source
        if str(dest_path) in root:
            continue
            
        for file_name in files:
            # Skip hidden files (starting with .)
            if not file_name.startswith('.'):
                all_files.append((root, file_name))

    total_files_found = len(all_files)
    if total_files_found == 0:
        print("No files found in the source directory.")
        print("\nDebug: Checking directory contents...")
        
        # Debug: show some directory contents
        for root, dirs, files in os.walk(str(source_path)):
            if str(dest_path) in root:
                continue
            print(f"\nIn directory: {root}")
            print(f"  Subdirectories: {dirs[:5]}")
            print(f"  Files: {files[:10]}")
            if len(files) > 0:
                break  # Just show the first level with files
    else:
        print(f"Found {total_files_found} files in total.")
        print()

    # Move the files
    for idx, (root, file_name) in enumerate(all_files, 1):
        source_file = Path(root) / file_name
        dest_file = dest_path / file_name

        # Handle duplicate names by adding a number
        counter = 1
        original_dest = dest_file
        while dest_file.exists():
            name, ext = os.path.splitext(file_name)
            dest_file = dest_path / f"{name}_{counter}{ext}"
            counter += 1

        if counter > 1:
            print(f"Moving {idx}/{total_files_found}: {file_name}")
            print(f"  ⚠️  File exists, renamed to: {dest_file.name}")
        else:
            print(f"Moving {idx}/{total_files_found}: {file_name}")
        
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
    print(f"Total files found: {total_files_found}")
    print(f"Successfully moved: {len(moved_items)} files")
    
    return moved_items

def main(source_directory, destination_directory):
    """
    Main function to run the file moving operation.
    
    Args:
        source_directory (str): Path to the source directory to search for files
        destination_directory (str): Path to the destination directory where files will be moved
    """
    print("=== EEG File Moving Tool (Move All Files) ===")
    print(f"Source: {source_directory}")
    print(f"Destination: {destination_directory}")
    print("=" * 60)
    
    # Confirm with user
    print("\n⚠️  WARNING: This will move ALL files from the source directory to the destination.")
    response = input("Do you want to continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Operation cancelled.")
        return
    
    try:
        moved_items = move_all_files(source_directory, destination_directory)
        print(f"\n✅ Operation completed successfully!")
        print(f"📁 Total files moved: {len(moved_items)}")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # EXAMPLE COMMAND:
    # python3 CLEAN/scripts/move_all_eeg_files.py "/Volumes/NED_Backup3/Q1K_Preprocessed_2s_Happe/5 - processed" "/Volumes/NED_Backup3/COMBINED_Q1K_BC_2s"
    # python3 CLEAN/scripts/move_all_eeg_files.py "/Volumes/NED_Backup3/Q1K_Preprocessed_5s_Happe/5 - processed" "/Volumes/NED_Backup3/COMBINED_Q1K_BC_5s"


    parser = argparse.ArgumentParser(description='Move all files from source to destination directory')
    parser.add_argument('source_directory', help='Path to the source directory to search for files')
    parser.add_argument('destination_directory', help='Path to the destination directory where files will be moved')
    
    args = parser.parse_args()
    main(args.source_directory, args.destination_directory)
