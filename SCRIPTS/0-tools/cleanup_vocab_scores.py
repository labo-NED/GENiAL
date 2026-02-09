### CLEANUP VOCABULARY SCORES (GENiAL PROJECT)
# This script merges vocabulary test scores from different tests (WISC, WAIS, WIPPSI)
# into a single column and adds a test type indicator
#
# Emmanuelle Coutu-Nadeau (Nov 2025)

import csv
from collections import defaultdict

# Input and output paths
input_file = '/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/DATA/REDCAP_REPORTS/Q1K/Q1KDatabase-ECNvocabularyscaleds_DATA_2025-11-18_1859.csv'
output_file = '/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/DATA/REDCAP_REPORTS/Q1K/Q1KDatabase-ECNvocabularyscaleds_DATA_2025-11-18_1859_cleaned.csv'

# Read the CSV file
print(f'Reading data from: {input_file}')

rows_with_scores = []
patient_data = {}  # Dictionary to store one row per patient

with open(input_file, 'r') as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    total_rows = 0
    rows_with_participant = 0
    rows_with_scores_count = 0
    
    for row in reader:
        total_rows += 1
        
        # Skip rows without participant code
        participant_code = row.get('eeg_participant_code', '').strip()
        if not participant_code:
            continue
        
        rows_with_participant += 1
        
        # Determine score and test type
        vocab_score = ''
        test_type = ''
        
        wisc_score = row.get('wisc_vc_ss', '').strip()
        wais_score = row.get('wais_vocab_ss', '').strip()
        wippsi_score = row.get('wippsi23_block_vocabulary_ss', '').strip()
        
        if wisc_score:
            vocab_score = wisc_score
            test_type = 'wisc'
        elif wais_score:
            vocab_score = wais_score
            test_type = 'wais'
        elif wippsi_score:
            vocab_score = wippsi_score
            test_type = 'wippsi'
        
        # Only keep rows with a score
        if vocab_score and test_type:
            rows_with_scores_count += 1
            
            # Store only the first occurrence per patient
            if participant_code not in patient_data:
                patient_data[participant_code] = {
                    'record_id': row.get('record_id', ''),
                    'eeg_participant_code': participant_code,
                    'vocab_score': vocab_score,
                    'test': test_type,
                    'redcap_event_name': row.get('redcap_event_name', ''),
                    'redcap_repeat_instrument': row.get('redcap_repeat_instrument', ''),
                    'redcap_repeat_instance': row.get('redcap_repeat_instance', '')
                }

print(f'Original rows: {total_rows}')
print(f'Rows with participant codes: {rows_with_participant}')
print(f'Rows with scores: {rows_with_scores_count}')
print(f'Unique patients: {len(patient_data)}')

# Write output CSV
output_columns = ['record_id', 'eeg_participant_code', 'vocab_score', 'test', 
                  'redcap_event_name', 'redcap_repeat_instrument', 'redcap_repeat_instance']

with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=output_columns)
    writer.writeheader()
    
    for participant_code in sorted(patient_data.keys()):
        writer.writerow(patient_data[participant_code])

print(f'\n✅ Cleaned data saved to: {output_file}')

# Calculate summary statistics
test_counts = defaultdict(int)
scores = []

for data in patient_data.values():
    test_counts[data['test']] += 1
    try:
        scores.append(float(data['vocab_score']))
    except ValueError:
        pass

print(f'\n{"="*60}')
print('SUMMARY:')
print(f'{"="*60}')
print(f'Total unique patients: {len(patient_data)}')
print(f'\nTest type distribution:')
for test_type in sorted(test_counts.keys()):
    print(f'  {test_type}: {test_counts[test_type]}')

if scores:
    scores.sort()
    n = len(scores)
    mean_score = sum(scores) / n
    min_score = min(scores)
    max_score = max(scores)
    median_score = scores[n//2] if n % 2 == 1 else (scores[n//2-1] + scores[n//2]) / 2
    
    print(f'\nVocabulary score statistics:')
    print(f'  Count: {n}')
    print(f'  Mean: {mean_score:.2f}')
    print(f'  Median: {median_score:.1f}')
    print(f'  Min: {min_score:.0f}')
    print(f'  Max: {max_score:.0f}')

print(f'{"="*60}\n')

