import os
import argparse
from pathlib import Path

def load_file_list(file_list_path):
    """
    Load specific filenames from a text file.
    
    Args:
        file_list_path (str): Path to the file containing filenames (one per line)
    
    Returns:
        list: List of filenames (stripped of whitespace)
    """
    filenames = []
    with open(file_list_path, 'r') as f:
        for line in f:
            filename = line.strip()
            if filename:  # Skip empty lines
                filenames.append(filename)
    return filenames

def rename_files_with_suffix(directory, suffix, file_list_path=None):
    """
    Rename files by adding a suffix (e.g., _2s or _5s) before the file extension.
    
    Args:
        directory (str): Directory containing files to rename
        suffix (str): Suffix to add (e.g., "_2s" or "_5s")
        file_list_path (str, optional): Path to file containing specific filenames to rename
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"❌ Error: Directory not found: {directory}")
        return []
    
    # Ensure suffix starts with underscore
    if not suffix.startswith('_'):
        suffix = '_' + suffix
    
    renamed_items = []
    files_to_rename = []
    
    print(f"Searching for files in: {dir_path}")
    print(f"Suffix to add: {suffix}")
    print("-" * 50)
    
    # If a file list is provided, only rename those files
    if file_list_path:
        print(f"Loading file list from: {file_list_path}")
        specific_filenames = load_file_list(file_list_path)
        print(f"Loaded {len(specific_filenames)} filenames from list")
        print()
        
        # Search for these specific files in the directory
        for root, dirs, files in os.walk(str(dir_path)):
            for file_name in files:
                if file_name in specific_filenames:
                    files_to_rename.append((root, file_name))
    else:
        # Rename all files in the directory
        for root, dirs, files in os.walk(str(dir_path)):
            for file_name in files:
                # Skip hidden files
                if not file_name.startswith('.'):
                    files_to_rename.append((root, file_name))
    
    total_files = len(files_to_rename)
    
    if total_files == 0:
        print("No files found to rename.")
        return []
    
    print(f"Found {total_files} file(s) to rename")
    print()
    
    # Rename the files
    for idx, (root, file_name) in enumerate(files_to_rename, 1):
        # Split filename and extension
        name, ext = os.path.splitext(file_name)
        
        # Check if suffix already exists in filename
        if suffix in name:
            print(f"Skipping {idx}/{total_files}: {file_name}")
            print(f"  ⚠️  Suffix '{suffix}' already present in filename")
            print()
            continue
        
        # Create new filename with suffix before extension
        new_file_name = f"{name}{suffix}{ext}"
        
        source_file = Path(root) / file_name
        dest_file = Path(root) / new_file_name
        
        # Check if destination file already exists
        if dest_file.exists():
            print(f"Skipping {idx}/{total_files}: {file_name}")
            print(f"  ⚠️  File already exists: {new_file_name}")
            print()
            continue
        
        print(f"Renaming {idx}/{total_files}:")
        print(f"  From: {file_name}")
        print(f"  To:   {new_file_name}")
        
        try:
            # Rename the file
            source_file.rename(dest_file)
            renamed_items.append(new_file_name)
            print(f"  ✓ Renamed successfully")
        except Exception as e:
            print(f"  ❌ Failed to rename: {e}")
        print()
    
    print("-" * 50)
    print(f"Operation completed.")
    print(f"Total files processed: {total_files}")
    print(f"Successfully renamed: {len(renamed_items)} files")
    print(f"Skipped: {total_files - len(renamed_items)} files")
    
    return renamed_items

def main():
    """
    Main function to run the file renaming operation.
    """
    parser = argparse.ArgumentParser(
        description='Rename EEG files by adding a suffix (e.g., _2s or _5s) before the file extension',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add _2s to all files in a directory
  python3 CLEAN/scripts/rename_eeg_files.py "/Volumes/NED_Backup3/COMBINED_Q1K_BC_2s" "_2s"
  
  # Add _5s to all files in a directory
  python3 CLEAN/scripts/rename_eeg_files.py "/Volumes/NED_Backup3/COMBINED_Q1K_BC_5s" "_5s"
  
  # Add _2s to specific files listed in a text file
  python3 CLEAN/scripts/rename_eeg_files.py "/Volumes/NED_Backup3/COMBINED_Q1K_BC_2s" "_2s" --file-list "CLEAN/file_list.txt"
        """
    )
    
    parser.add_argument('directory', help='Path to the directory containing files to rename')
    parser.add_argument('suffix', help='Suffix to add to filenames (e.g., "_2s" or "_5s")')
    parser.add_argument('--file-list', '-f', dest='file_list', 
                        help='Optional: Path to a text file containing specific filenames to rename (one per line)')
    
    args = parser.parse_args()
    
    print("=== EEG File Renaming Tool ===")
    print(f"Directory: {args.directory}")
    print(f"Suffix: {args.suffix}")
    if args.file_list:
        print(f"File list: {args.file_list}")
    print("=" * 60)
    print()
    
    # Confirm with user
    print("⚠️  WARNING: This will rename files in the directory.")
    response = input("Do you want to continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Operation cancelled.")
        return
    print()
    
    try:
        renamed_items = rename_files_with_suffix(args.directory, args.suffix, args.file_list)
        print(f"\n✅ Operation completed successfully!")
        print(f"📁 Total files renamed: {len(renamed_items)}")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

