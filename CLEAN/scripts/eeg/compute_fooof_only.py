### COMPUTE FOOOF FEATURES (GENiAL PROJECT)
# This script extracts FOOOF aperiodic parameters and periodic power from epoched EEG data
# 
# Features extracted:
#   - fooof_offset: Aperiodic offset (broadband power)
#   - fooof_exponent: Aperiodic exponent (1/f slope)
#   - Periodic power in frequency bands (delta, theta, alpha, beta, gamma, etc.)
#
# Emmanuelle Coutu-Nadeau (Dec 2025) - Based on compute_features.py

# ---------- IMPORT PACKAGES ----------
import numpy as np
import mne
from fooof import FOOOFGroup
import pickle
import time
import os
import glob
import pandas as pd

start_time = time.time()

# ------------ Paths ------------
# Detect if running locally or on cluster
IS_LOCAL = False  # Set to True for local runs, False for cluster runs

if IS_LOCAL:
    # Locally - no parallelization
    input_dir = '/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/EEG'
    output_dir = '/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/eeg_fooof_aperiodic'
    os.makedirs(output_dir, exist_ok=True)
else:
    # Cluster
    input_dir = '/home/emmacona/links/projects/def-lippes/emmacona/Q1K_BC_HAPPEv3_ICA/2s_epochs'
    output_dir = '/home/emmacona/links/projects/def-lippes/emmacona/Q1K_BC_HAPPEv3_ICA/Features/'
    os.makedirs(output_dir, exist_ok=True)

# Scan directory for .set files
listing = glob.glob(os.path.join(input_dir, '*.set'))

# Remove mac hidden files
files = [f for f in listing if not os.path.basename(f).startswith('._')]

nFiles = len(files)

if nFiles == 0:
    raise ValueError(f'No .set files found in: {input_dir}')

print(f'\n{"="*60}')
print(f'FOOOF FEATURE EXTRACTION')
print(f'{"="*60}')
print(f'Input directory:  {input_dir}')
print(f'Output directory: {output_dir}')
print(f'Found {nFiles} files to process')
if IS_LOCAL:
    print('*** RUNNING IN LOCAL MODE ***')
else:
    print('*** RUNNING IN CLUSTER MODE ***')
print(f'{"="*60}\n')

# ------------ Common Params ------------
Fs = 1000
n_chans = 108

# Set n_cores: 1 for local (no parallelization), use SLURM value for cluster
if IS_LOCAL:
    n_cores = 1
    print('Parallelization disabled (n_cores=1)\n')
else:
    n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    print(f'Using {n_cores} cores for parallelization\n')

excluded_chans = ['48','119','43','49','56','63','68','73','81','88','94','99','107','113','120','125','126','127','128','17','129']
included_chans = [str(idx) for idx in range(1, 130) if str(idx) not in excluded_chans]

# FOOOF parameters
FMIN = 0.5
FMAX = 80
MIN_PEAK_HEIGHT = 0.1
PEAK_WIDTH_LIMITS = (1, 12)

# Frequency bands for periodic power computation
FREQ_BANDS = {'delta': [1,4],
              'theta': [4,8],
              'alpha': [8,13],
              'beta': [13,30],
              'gamma': [30,80],
              'low_gamma': [30,59], 
              'high_gamma': [61,80]}

# ---------- MAIN PROCESSING LOOP ----------
processed_count = 0
error_count = 0

for file_idx, filepath in enumerate(files):
    filename = os.path.basename(filepath)
    
    # Define output filenames (will be overwritten if they exist)
    output_base = filename.replace('.set', '').replace('_processed', '')
    out_pkl = os.path.join(output_dir, f'fooof_aperiodic_{output_base}.pkl')
    out_csv = os.path.join(output_dir, f'fooof_aperiodic_{output_base}.csv')
    
    print(f'\n[{file_idx+1}/{nFiles}] Processing: {filename}')
    
    try:
        # Load epochs
        print('  Loading epochs...')
        epochs = mne.io.read_epochs_eeglab(filepath, verbose=False)
        
        # Keep only bad channels that actually exist in this file
        bad_existing = [ch for ch in excluded_chans if ch in epochs.ch_names]
        epochs.info['bads'] = bad_existing
        
        epoch_data = epochs.get_data(picks='all')[:, :n_chans]  # excluding bad channels
        n_epochs = epoch_data.shape[0]
        
        print(f'  Shape: {epoch_data.shape} (epochs x channels x samples)')
        
        if n_epochs == 0:
            print('  WARNING: No epochs found, skipping...')
            error_count += 1
            continue
        
        # Compute PSD
        print('  Computing PSD...')
        psd_start = time.time()
        spec = epochs.compute_psd(method='welch', fmin=FMIN, fmax=FMAX, n_jobs=n_cores, verbose=False)
        freqs = spec.freqs
        powers = spec.get_data()
        
        # Average PSD across epochs for FOOOF fitting
        avg_powers = powers.mean(axis=0)  # average across epochs: (n_chans, n_freqs)
        psd_time = time.time() - psd_start
        print(f'  PSD computed in {psd_time:.2f}s')
        
        # Fit FOOOF
        print('  Fitting FOOOF...')
        fooof_start = time.time()
        fg = FOOOFGroup(min_peak_height=MIN_PEAK_HEIGHT, 
                        peak_width_limits=PEAK_WIDTH_LIMITS,
                        verbose=False)
        fg.fit(freqs, avg_powers, freq_range=[FMIN, FMAX], n_jobs=n_cores)
        fooof_time = time.time() - fooof_start
        print(f'  FOOOF fitted in {fooof_time:.2f}s')
        
        # Extract aperiodic parameters
        aper_params = fg.get_params('aperiodic')  # (n_chans, 2) - offset and exponent
        
        offset = aper_params[:, 0]
        exponent = aper_params[:, 1]
        
        print(f'  Offset range:   [{offset.min():.4f}, {offset.max():.4f}]')
        print(f'  Exponent range: [{exponent.min():.4f}, {exponent.max():.4f}]')
        
        # Extract periodic spectrum using the same method as compute_features.py
        # (full spectrum - aperiodic spectrum = periodic/oscillatory component)
        print('  Computing periodic spectrum...')
        per_spec_avg = np.zeros((n_chans, len(freqs)))
        for ch in range(n_chans):
            # Extract periodic spectrum by subtracting aperiodic from full spectrum
            # This captures all oscillatory activity, not just detected peaks
            full_spectrum = fg.get_fooof(ch).get_data(component='full', space='linear')
            aperiodic_spectrum = fg.get_fooof(ch).get_data(component='aperiodic', space='linear')
            periodic_spectrum = full_spectrum - aperiodic_spectrum
            per_spec_avg[ch] = periodic_spectrum  # Store for band averaging
        
        # Compute periodic power in frequency bands
        print('  Computing periodic power in frequency bands...')
        periodic_powers = {}
        for band_name, band_freqs in FREQ_BANDS.items():
            band_mask = np.logical_and(freqs >= band_freqs[0], freqs < band_freqs[1])
            per_power_band = np.mean(per_spec_avg[:, band_mask], axis=1)  # (n_chans,)
            periodic_powers[f'pow_per_{band_name}'] = per_power_band
            print(f'    pow_per_{band_name}: range [{per_power_band.min():.6e}, {per_power_band.max():.6e}]')
        
        # Create results dictionary
        results = {
            'filename': filename,
            'n_epochs': n_epochs,
            'n_channels': n_chans,
            'channel_list': included_chans,
            'fooof_offset': offset,
            'fooof_exponent': exponent,
            'freqs': freqs,
            'avg_powers': avg_powers,
            'periodic_spectrum': per_spec_avg,  # Full periodic spectrum
            'periodic_powers': periodic_powers,  # Periodic power in frequency bands
            'fooof_group': fg  # Save the entire FOOOFGroup object for later inspection
        }
        
        # Save to pickle
        with open(out_pkl, 'wb') as f:
            pickle.dump(results, f)
        print(f'  Saved: {os.path.basename(out_pkl)}')
        
        # Save to CSV (channel-wise)
        csv_data = {
            'channel': included_chans,
            'fooof_offset': offset,
            'fooof_exponent': exponent
        }
        # Add periodic power columns
        for band_name, band_power in periodic_powers.items():
            csv_data[band_name] = band_power
        
        df = pd.DataFrame(csv_data)
        df.to_csv(out_csv, index=False)
        print(f'  Saved: {os.path.basename(out_csv)}')
        
        processed_count += 1
        
    except Exception as e:
        print(f'  ERROR: {e}')
        error_count += 1
        continue

# ---------- SUMMARY ----------
total_time = time.time() - start_time

print(f'\n{"="*60}')
print(f'PROCESSING COMPLETE')
print(f'{"="*60}')
print(f'Total files:      {nFiles}')
print(f'Processed:        {processed_count}')
print(f'Errors:           {error_count}')
print(f'Total time:       {total_time/60:.2f} minutes ({total_time:.1f} seconds)')
if processed_count > 0:
    print(f'Average per file: {total_time/processed_count:.1f} seconds')
print(f'Output directory: {output_dir}')
print(f'{"="*60}\n')

