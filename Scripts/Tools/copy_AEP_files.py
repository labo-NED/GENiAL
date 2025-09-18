import os
import shutil
import argparse
from pathlib import Path

def copy_aep_files_flat(source_dir, destination_dir):
    """
    Copy all directories containing 'AEP' in their name and ending with '.mff' to a single destination directory.
    .mff files are actually directories in EEG data.
    """
    source_path = Path(source_dir)
    dest_path = Path(destination_dir)
    
    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    copied_items = []
    aep_mff_paths = []

    print(f"Starting search for AEP .mff directories in: {source_path}")
    print(f"Destination directory: {dest_path}")
    print("-" * 50)
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(str(source_path)):
        # Check directories first (since .mff are directories)
        for dir_name in dirs:
            if 'AEP' in dir_name and dir_name.lower().endswith('.mff'):
                aep_mff_paths.append((root, dir_name, 'directory'))
        
        # Also check files in case some are actual files
        for file_name in files:
            if 'AEP' in file_name and file_name.lower().endswith('.mff'):
                aep_mff_paths.append((root, file_name, 'file'))

    total_items_found = len(aep_mff_paths)
    if total_items_found == 0:
        print("No AEP .mff items found. Please check your source directory and file naming.")
        print("Looking for items containing 'AEP' and ending with '.mff'")
        print("\nDebug: Let me show you what's in the directory...")
        
        # Debug: show some directory contents
        for root, dirs, files in os.walk(str(source_path)):
            print(f"\nIn directory: {root}")
            print(f"  Subdirectories: {dirs[:5]}...")  # Show first 5
            print(f"  Files: {files[:5]}...")  # Show first 5
            break  # Just show the first level
            
    else:
        print(f"Found {total_items_found} AEP .mff items in total.")

    for idx, (root, item_name, item_type) in enumerate(aep_mff_paths, 1):
        source_item = Path(root) / item_name
        dest_item = dest_path / item_name

        # Handle duplicate names by adding a number
        counter = 1
        while dest_item.exists():
            name, ext = os.path.splitext(item_name)
            dest_item = dest_path / f"{name}_{counter}{ext}"
            counter += 1

        print(f"Copying {idx}/{total_items_found}: {item_name} ({item_type})")
        
        if item_type == 'directory':
            # Copy entire directory tree
            shutil.copytree(source_item, dest_item)
        else:
            # Copy single file
            shutil.copy2(source_item, dest_item)
            
        copied_items.append(dest_item.name)
        print(f"  ✓ Copied: {source_item} -> {dest_item}")

    print("-" * 50)
    print(f"Search completed. Found {total_items_found} AEP .mff items total.")
    print(f"Successfully copied {len(copied_items)} items")
    return copied_items

def main(source_directory, destination_directory):
    """
    Main function to run the AEP .mff file copying operation.
    
    Args:
        source_directory (str): Path to the source directory to search for AEP .mff files
        destination_directory (str): Path to the destination directory where AEP .mff files will be copied
    """
    print("=== AEP .mff File/Directory Copying Tool ===")
    print(f"Source: {source_directory}")
    print(f"Destination: {destination_directory}")
    print("=" * 40)
    
    try:
        copied_items = copy_aep_files_flat(source_directory, destination_directory)
        print(f"\n✅ Operation completed successfully!")
        print(f"📁 Total items copied: {len(copied_items)}")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Example usage:
    # python3 copy_AEP_files.py /path/to/source/directory /path/to/destination/directory
    
    parser = argparse.ArgumentParser(description='Copy AEP .mff files/directories from source to destination directory')
    parser.add_argument('source_directory', help='Path to the source directory to search for AEP .mff files')
    parser.add_argument('destination_directory', help='Path to the destination directory where AEP .mff files will be copied')
    
    args = parser.parse_args()
    main(args.source_directory, args.destination_directory)