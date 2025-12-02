#!/usr/bin/env bash
set -euo pipefail

########################################
# CONFIG
########################################

# Where your 2s EEGs live on the external drive
SRC_BASE="/Volumes/NED_Backup3/COMBINED_Q1K_BC_2s/curated_list_for_genial_2s/retry_transfer"

# Remote host and user
REMOTE_HOST="rorqual.calculquebec.ca"
REMOTE_USER="emmacona"

# Destination on Rorqual
DEST_BASE="/home/emmacona/links/projects/def-lippes/emmacona/Q1K_BC_HAPPEv3_ICA/2s_epochs"

########################################
# RSYNC
########################################

echo "Syncing:"
echo "  from: $SRC_BASE/"
echo "  to  : $REMOTE_USER@$REMOTE_HOST:$DEST_BASE/"
echo

rsync -avh --progress \
  --partial --append \
  --timeout=600 \
  "$SRC_BASE"/ \
  "$REMOTE_USER@$REMOTE_HOST:$DEST_BASE/"

echo
echo "Sync finished."

