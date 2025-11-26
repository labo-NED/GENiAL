#!/usr/bin/env bash

set -euo pipefail

# Paths
HD_LIST="/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/eeg_features_on_HD.txt"
LOCAL_LIST="/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/eeg_features_on_local.txt"

SRC_DIR="/Volumes/NED_Backup3/Q1K_BC_EEG_features/Q1K_BC_EEG_features"
DEST_DIR="/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/eeg_features"

echo "Source directory: $SRC_DIR"
echo "Destination directory: $DEST_DIR"
echo

mkdir -p "$DEST_DIR"

# Loop over all feature files that exist on the external HD
while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines
    [[ -z "$line" ]] && continue

    # Support either plain filenames or paths in the txt files
    fname="$(basename "$line")"
    src="$SRC_DIR/$fname"
    dest="$DEST_DIR/$fname"

    # If this file is listed as already computed locally AND exists locally, keep local
    if grep -qxF "$fname" "$LOCAL_LIST" 2>/dev/null && [[ -f "$dest" ]]; then
        echo "Keeping local version of $fname (already in Outputs)."
        continue
    fi

    # Otherwise, copy from external drive if it exists
    if [[ -f "$src" ]]; then
        if [[ -f "$dest" ]]; then
            echo "File $fname exists locally but is not in eeg_features_on_local.txt."
            echo "Not overwriting. Skipping."
        else
            echo "Copying $fname from HD to local Outputs..."
            cp "$src" "$dest"
        fi
    else
        echo "Warning: $src not found on external drive. Skipping."
    fi

done < "$HD_LIST"

echo
echo "Done syncing feature files."
