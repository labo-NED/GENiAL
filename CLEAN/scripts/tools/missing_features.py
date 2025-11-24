import os
import glob
from collections import defaultdict

# Path to feature files
features_dir = '/Volumes/NED_Backup3/Q1K_BC_EEG_features/Q1K_BC_EEG_features'

# Find all feature CSV files
csv_pattern = os.path.join(features_dir, 'features_avg_*.csv')
csv_files = glob.glob(csv_pattern)

print(f"Total feature files found: {len(csv_files)}\n")

# Count files per participant and epoch type
participant_files = defaultdict(lambda: {'2s': 0, '5s': 0, 'total': 0})

for csv_file in csv_files:
    filename = os.path.basename(csv_file)
    
    # Determine epoch type
    epoch_type = None
    if filename.endswith('_2s.csv'):
        epoch_type = '2s'
    elif filename.endswith('_5s.csv'):
        epoch_type = '5s'
    
    # Extract base participant identifier (remove epoch and features_avg_ prefix)
    base_name = filename.replace('features_avg_', '').replace('_2s.csv', '').replace('_5s.csv', '')
    
    # Count
    if epoch_type:
        participant_files[base_name][epoch_type] += 1
        participant_files[base_name]['total'] += 1

# Analyze the counts
has_both = []
only_2s = []
only_5s = []
multiple_files = []

for participant, counts in sorted(participant_files.items()):
    if counts['2s'] > 0 and counts['5s'] > 0:
        has_both.append((participant, counts))
    elif counts['2s'] > 0:
        only_2s.append((participant, counts))
    elif counts['5s'] > 0:
        only_5s.append((participant, counts))
    
    if counts['total'] > 2:
        multiple_files.append((participant, counts))

# Print summary
print("="*60)
print("FILE COUNT SUMMARY")
print("="*60)
print(f"Participants with BOTH 2s and 5s files: {len(has_both)}")
print(f"Participants with ONLY 2s files: {len(only_2s)}")
print(f"Participants with ONLY 5s files: {len(only_5s)}")
print(f"Total unique participants: {len(participant_files)}")

if multiple_files:
    print(f"\nParticipants with MORE than 2 files: {len(multiple_files)}")
    for participant, counts in multiple_files[:10]:
        print(f"  {participant}: {counts['2s']} x 2s, {counts['5s']} x 5s (total: {counts['total']})")
    if len(multiple_files) > 10:
        print(f"  ... and {len(multiple_files) - 10} more")

print("\n" + "="*60)
print("PARTICIPANTS WITH ONLY 2s FILES (first 20):")
print("="*60)
for participant, counts in only_2s[:20]:
    print(f"{participant}: {counts['2s']} file(s)")

print("\n" + "="*60)
print("PARTICIPANTS WITH ONLY 5s FILES (first 20):")
print("="*60)
for participant, counts in only_5s[:20]:
    print(f"{participant}: {counts['5s']} file(s)")