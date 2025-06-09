
import argparse
import os
from pathlib import Path
import shutil

def get_missing_preprocessed(raw_list_path, preprocessed_list_path, output_path='missing_preprocessed.txt'):
    with open(raw_list_path, 'r') as f:
        raw_files = [line.strip() for line in f if line.strip().endswith('.set') and '_processed' not in line]

    with open(preprocessed_list_path, 'r') as f:
        preprocessed_files = [line.strip() for line in f if line.strip().endswith('_processed.set')]

    raw_basenames = {Path(f).stem for f in raw_files}
    preprocessed_basenames = {Path(f).stem.replace('_processed', '') for f in preprocessed_files}

    missing = sorted(raw_basenames - preprocessed_basenames)

    with open(output_path, 'w') as f:
        for name in missing:
            f.write(name + '.set\n')

    print(f"Done! {len(missing)} files are missing preprocessing. See '{output_path}'.")

def move_missing_files(missing_list_path, source_dir, destination_dir):
    os.makedirs(destination_dir, exist_ok=True)

    with open(missing_list_path, 'r') as f:
        missing_files = [line.strip() for line in f if line.strip().endswith('.set')]

    for file_name in missing_files:
        source_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(destination_dir, file_name)

        if os.path.exists(source_path):
            shutil.move(source_path, dest_path)
            print(f"Moved: {file_name}")
        else:
            print(f"File not found: {file_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find and optionally move raw .set files that have not been preprocessed.')
    parser.add_argument('--raw', help='Path to text file listing raw .set files')
    parser.add_argument('--preprocessed', help='Path to text file listing preprocessed _processed.set files')
    parser.add_argument('--output', default='missing_preprocessed.txt', help='Where to save or read the list of missing files')
    parser.add_argument('--move', action='store_true', help='Move missing files to another folder')
    parser.add_argument('--source_dir', help='Folder where raw .set files are stored')
    parser.add_argument('--dest_dir', help='Folder to move missing .set files to')

    args = parser.parse_args()

    if not args.move:
        if not args.raw or not args.preprocessed:
            parser.error("--raw and --preprocessed are required unless using --move")
        get_missing_preprocessed(args.raw, args.preprocessed, args.output)

    if args.move:
        if not args.source_dir or not args.dest_dir:
            parser.error("--source_dir and --dest_dir must be provided with --move")
        move_missing_files(args.output, args.source_dir, args.dest_dir)
