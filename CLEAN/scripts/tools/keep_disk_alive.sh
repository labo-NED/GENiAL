#!/bin/bash
# Keep external disk alive to prevent macOS auto-eject
# Run this in a separate terminal: bash keep_disk_alive.sh

DISK="/Volumes/NED_Backup3"
KEEPALIVE_FILE="$DISK/.manual_keepalive"

echo "=== Disk Keep-Alive Script ==="
echo "Disk: $DISK"
echo "This will keep the disk active by reading/writing every 3 seconds"
echo "Press Ctrl+C to stop"
echo ""

# Check if disk exists
if [ ! -d "$DISK" ]; then
    echo "ERROR: Disk $DISK not found!"
    exit 1
fi

echo "✓ Disk found. Starting keep-alive loop..."
echo ""

# Counter for display
counter=0

# Trap Ctrl+C to clean up
trap "echo ''; echo 'Stopping...'; rm -f '$KEEPALIVE_FILE'; echo 'Cleaned up. Done.'; exit 0" INT

# Main loop - every 3 seconds
while true; do
    counter=$((counter + 1))
    
    # Write operation
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Ping $counter" >> "$KEEPALIVE_FILE"
    
    # Read operation
    ls "$DISK" > /dev/null 2>&1
    
    # Display status
    echo -ne "\r✓ Keep-alive ping $counter (every 3s)    "
    
    sleep 3
done

