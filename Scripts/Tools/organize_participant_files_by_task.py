import os
import shutil
from pathlib import Path
from collections import defaultdict

# ==== USER: ENTER YOUR PATHS HERE ====
# Example:
# source_directory = "/Users/yourname/EEG/ParticipantFolders"
# destination_directory = "/Users/yourname/EEG/OrganizedByTask"

source_directory = "/Volumes/LaCie/PEARL"  # <-- Enter the path to your participant folders here
destination_directory = "/Volumes/LaCie/PEARL - Organized by Task"  # <-- Enter the path to your destination folder here

# =====================================

def organize_files_by_task(source_dir, destination_dir):
    """
    Organize participant files by task type.
    Each participant folder contains .mff files with pattern: PAERL_XX_VX_[task_code]
    where task_code is one of: RSrio, Fixrs, aep, to, cin, vep

    Creates task-specific folders containing all participants' files for that task.
    """
    source_path = Path(source_dir)
    dest_path = Path(destination_dir)

    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    # Define task codes to look for
    task_codes = ['RSrio', 'Fixrs', 'aep', 'to', 'cin', 'vep']

    # Dictionary to store files by task
    files_by_task = defaultdict(list)

    print(f"Starting search for participant files in: {source_path}")
    print(f"Destination directory: {dest_path}")
    print(f"Looking for task codes: {', '.join(task_codes)}")
    print("-" * 60)

    # Walk through the directory structure
    for root, dirs, files in os.walk(str(source_path)):
        # Check directories first (since .mff are directories)
        for dir_name in dirs:
            if dir_name.lower().endswith('.mff'):
                # Extract task code from filename (case-insensitive)
                dir_name_lower = dir_name.lower()
                for task_code in task_codes:
                    if f'_{task_code.lower()}' in dir_name_lower:
                        full_path = Path(root) / dir_name
                        files_by_task[task_code].append((full_path, dir_name))
                        break

        # Also check files in case some are actual files
        for file_name in files:
            if file_name.lower().endswith('.mff'):
                # Extract task code from filename (case-insensitive)
                file_name_lower = file_name.lower()
                for task_code in task_codes:
                    if f'_{task_code.lower()}' in file_name_lower:
                        full_path = Path(root) / file_name
                        files_by_task[task_code].append((full_path, file_name))
                        break

    # Print summary of found files
    total_files = sum(len(files) for files in files_by_task.values())
    print(f"Found {total_files} files total across all tasks:")
    for task_code in task_codes:
        count = len(files_by_task[task_code])
        print(f"  {task_code}: {count} files")

    if total_files == 0:
        print("No files found. Please check your source directory and file naming.")
        print("Looking for files with pattern: PAERL_XX_VX_[task_code].mff")
        print("\nDebug: Let me show you what's in the directory...")

        # Debug: show some directory contents
        for root, dirs, files in os.walk(str(source_path)):
            print(f"\nIn directory: {root}")
            print(f"  Subdirectories: {dirs[:5]}...")  # Show first 5
            print(f"  Files: {files[:5]}...")  # Show first 5
            break  # Just show the first level
        return []

    # Create task folders and copy files
    copied_files = []

    for task_code in task_codes:
        if not files_by_task[task_code]:
            continue

        # Create task-specific folder
        task_folder = dest_path / task_code
        task_folder.mkdir(exist_ok=True)

        print(f"\nProcessing {task_code} task ({len(files_by_task[task_code])} files):")

        for idx, (source_file, file_name) in enumerate(files_by_task[task_code], 1):
            dest_file = task_folder / file_name

            # Handle duplicate names by adding participant info
            counter = 1
            while dest_file.exists():
                name, ext = os.path.splitext(file_name)
                dest_file = task_folder / f"{name}_{counter}{ext}"
                counter += 1

            print(f"  Copying {idx}/{len(files_by_task[task_code])}: {file_name}")

            try:
                if source_file.is_dir():
                    # Copy entire directory tree
                    shutil.copytree(source_file, dest_file)
                else:
                    # Copy single file
                    shutil.copy2(source_file, dest_file)

                copied_files.append(f"{task_code}/{dest_file.name}")
                print(f"    ✓ Copied: {source_file} -> {dest_file}")

            except Exception as e:
                print(f"    ❌ Error copying {file_name}: {e}")

    print("-" * 60)
    print(f"Organization completed!")
    print(f"Total files processed: {len(copied_files)}")

    # Show final structure
    print(f"\nFinal folder structure in {dest_path}:")
    for task_code in task_codes:
        task_folder = dest_path / task_code
        if task_folder.exists():
            file_count = len(list(task_folder.iterdir()))
            print(f"  {task_code}/: {file_count} files")

    return copied_files

def main():
    print("=== Participant File Organization by Task ===")
    print(f"Source: {source_directory}")
    print(f"Destination: {destination_directory}")
    print("=" * 50)

    # Check that user has entered the paths
    if not source_directory or not destination_directory:
        print("❌ Please set both 'source_directory' and 'destination_directory' at the top of this script before running.")
        return

    try:
        copied_files = organize_files_by_task(source_directory, destination_directory)
        print(f"\n✅ Operation completed successfully!")
        print(f"📁 Total files organized: {len(copied_files)}")

        # Show summary by task
        task_summary = defaultdict(int)
        for file_path in copied_files:
            task = file_path.split('/')[0]
            task_summary[task] += 1

        if task_summary:
            print(f"\n📊 Files organized by task:")
            for task, count in sorted(task_summary.items()):
                print(f"  {task}: {count} files")

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # To use: Set the source_directory and destination_directory variables at the top of this script.
    main()
