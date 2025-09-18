import os
import shutil
import argparse
from pathlib import Path

def move_rsrio_files(source_dir, destination_dir):
    """
    Move all files containing 'rsrio' (case-insensitive) in their name to a single destination directory.
    """
    source_path = Path(source_dir)
    dest_path = Path(destination_dir)
    
    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    moved_items = []
    rsrio_files = []

    print(f"Starting search for files containing 'rsrio' in: {source_path}")
    print(f"Destination directory: {dest_path}")
    print("-" * 50)
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(str(source_path)):
        for file_name in files:
            if 'rsrio' in file_name.lower():
                rsrio_files.append((root, file_name))

    total_files_found = len(rsrio_files)
    if total_files_found == 0:
        print("No files containing 'rsrio' found. Please check your source directory and file naming.")
        print("Looking for files containing 'rsrio' (case-insensitive)")
        print("\nDebug: Let me show you what's in the directory...")
        
        # Debug: show some directory contents
        for root, dirs, files in os.walk(str(source_path)):
            print(f"\nIn directory: {root}")
            print(f"  Files: {files[:10]}...")  # Show first 10
            if len(files) > 0:
                break  # Just show the first level with files
    else:
        print(f"Found {total_files_found} files containing 'rsrio' in total.")

    for idx, (root, file_name) in enumerate(rsrio_files, 1):
        source_file = Path(root) / file_name
        dest_file = dest_path / file_name

        # Handle duplicate names by adding a number
        counter = 1
        while dest_file.exists():
            name, ext = os.path.splitext(file_name)
            dest_file = dest_path / f"{name}_{counter}{ext}"
            counter += 1

        print(f"Moving {idx}/{total_files_found}: {file_name}")
        
        try:
            # Move the file
            shutil.move(str(source_file), str(dest_file))
            moved_items.append(dest_file.name)
            print(f"  ✓ Moved: {source_file} -> {dest_file}")
        except Exception as e:
            print(f"  ❌ Failed to move {source_file}: {e}")

    print("-" * 50)
    print(f"Search completed. Found {total_files_found} files containing 'rsrio' total.")
    print(f"Successfully moved {len(moved_items)} files")
    return moved_items

def main(source_directory, destination_directory):
    """
    Main function to run the rsrio file moving operation.
    
    Args:
        source_directory (str): Path to the source directory to search for rsrio files
        destination_directory (str): Path to the destination directory where rsrio files will be moved
    """
    print("=== rsrio File Moving Tool ===")
    print(f"Source: {source_directory}")
    print(f"Destination: {destination_directory}")
    print("=" * 40)
    
    try:
        moved_items = move_rsrio_files(source_directory, destination_directory)
        print(f"\n✅ Operation completed successfully!")
        print(f"📁 Total files moved: {len(moved_items)}")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # EXAMPLES COMMANDS
    # 2s
    # python3 Scripts/Tools/move_rsrio_files.py "/Volumes/NED_Backup3/Q1K_Preprocessed_2s_Happe/5 - processed" "/Volumes/NED_Backup3/COMBINED_Q1K_BC_2s"
    
    #5s
    # python3 Scripts/Tools/move_rsrio_files.py "/Volumes/NED_Backup3/Q1K_Preprocessed_5s_Happe/5 - processed" "/Volumes/NED_Backup3/COMBINED_Q1K_BC_5s"
    

    parser = argparse.ArgumentParser(description='Move files containing rsrio (case-insensitive) from source to destination directory')
    parser.add_argument('source_directory', help='Path to the source directory to search for rsrio files')
    parser.add_argument('destination_directory', help='Path to the destination directory where rsrio files will be moved')
    
    args = parser.parse_args()
    main(args.source_directory, args.destination_directory)
