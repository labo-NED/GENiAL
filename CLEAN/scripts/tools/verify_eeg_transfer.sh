#!/usr/bin/env bash
set -euo pipefail

########################################
# CONFIG
########################################

# Local source directory
SRC_BASE="/Volumes/NED_Backup3/COMBINED_Q1K_BC_2s/curated_list_for_genial_2s"

# Remote host and user
REMOTE_HOST="rorqual.calculquebec.ca"
REMOTE_USER="emmacona"

# Remote destination
DEST_BASE="/home/emmacona/links/projects/def-lippes/emmacona/Q1K_BC_HAPPEv3_ICA/2s_epochs"

# Output files
LOCAL_LIST="$HOME/local_eeg_sizes.txt"
REMOTE_LIST="$HOME/remote_eeg_sizes.txt"
MISMATCH_FILE="$HOME/eeg_size_mismatches.txt"

########################################
# COLLECT FILE SIZES
########################################

echo "Collecting local file sizes..."
find "$SRC_BASE" -type f -name "*.set" -exec stat -f "%z %N" {} \; | \
  sed "s|$SRC_BASE/||" | \
  sort -k2 > "$LOCAL_LIST"

echo "Local files: $(wc -l < "$LOCAL_LIST" | xargs)"

echo "Collecting remote file sizes..."
ssh "$REMOTE_USER@$REMOTE_HOST" \
  "find '$DEST_BASE' -type f -name '*.set' -exec stat -c '%s %n' {} \;" | \
  sed "s|$DEST_BASE/||" | \
  sort -k2 > "$REMOTE_LIST"

echo "Remote files: $(wc -l < "$REMOTE_LIST" | xargs)"

########################################
# COMPARE AND FIND MISMATCHES
########################################

echo ""
echo "Comparing file sizes..."

# Clear the mismatch file
> "$MISMATCH_FILE"

# Track counts
missing_count=0
size_mismatch_count=0

# Read local files
while IFS= read -r line; do
  local_size=$(echo "$line" | awk '{print $1}')
  filename=$(echo "$line" | awk '{$1=""; print $0}' | sed 's/^ //')
  
  # Look for matching file in remote list
  remote_line=$(grep -F "$filename" "$REMOTE_LIST" || echo "")
  
  if [ -z "$remote_line" ]; then
    echo "MISSING: $filename" >> "$MISMATCH_FILE"
    ((missing_count++))
  else
    remote_size=$(echo "$remote_line" | awk '{print $1}')
    
    if [ "$local_size" != "$remote_size" ]; then
      echo "SIZE_MISMATCH: $filename (local: $local_size bytes, remote: $remote_size bytes)" >> "$MISMATCH_FILE"
      ((size_mismatch_count++))
    fi
  fi
done < "$LOCAL_LIST"

# Check for files on remote that aren't local (shouldn't happen but good to check)
extra_count=0
while IFS= read -r line; do
  filename=$(echo "$line" | awk '{$1=""; print $0}' | sed 's/^ //')
  
  if ! grep -qF "$filename" "$LOCAL_LIST"; then
    echo "EXTRA_ON_REMOTE: $filename" >> "$MISMATCH_FILE"
    ((extra_count++))
  fi
done < "$REMOTE_LIST"

########################################
# REPORT
########################################

echo ""
echo "========================================"
echo "VERIFICATION COMPLETE"
echo "========================================"
echo "Missing files:        $missing_count"
echo "Size mismatches:      $size_mismatch_count"
echo "Extra on remote:      $extra_count"
echo ""

if [ $((missing_count + size_mismatch_count + extra_count)) -eq 0 ]; then
  echo "✓ All files transferred successfully!"
  rm "$MISMATCH_FILE"
else
  echo "✗ Issues found. Details written to:"
  echo "  $MISMATCH_FILE"
  echo ""
  echo "Preview of issues:"
  head -20 "$MISMATCH_FILE"
  if [ $(wc -l < "$MISMATCH_FILE") -gt 20 ]; then
    echo "... (see $MISMATCH_FILE for full list)"
  fi
fi

echo ""
echo "File lists saved to:"
echo "  Local:  $LOCAL_LIST"
echo "  Remote: $REMOTE_LIST"

