### COMPUTE EEG FEATURES FOR RESTING-STATE DATA (GENiAL PROJECT)
# This script extracts spectral and complexity features from epoched EEG data
# 
# Features extracted:
#   From 2s epochs: Hurst exponent, band powers, FOOOF parameters, periodic powers
#   From 5s epochs: Fractal dimensions, sample entropy, MSE, complexity index
#
# Laurent Caplette (2025) based on code by Saeideh Davoodi
# Emmanuelle Coutu-Nadeau (Nov 2025) - Adapted for GENiAL resting-state analysis

# ---------- IMPORT PACKAGES ----------
import numpy as np
import mne
from mne_features.feature_extraction import FeatureExtractor
from matplotlib import pyplot as plt
from fooof import FOOOFGroup
import pickle, time
from neurokit2 import entropy_multiscale
from joblib import Parallel, delayed
import os
import glob

start_time = time.time()

# ------------ User Toggle ------------
EPOCH_2S = True  # Set to True to process 2s epochs, False to skip
EPOCH_5S = False  # Set to True to process 5s epochs, False to skip

SKIP_PROCESSED = False # Set to True if want to skip files that already have outputs

# ------------ Paths ------------
# Detect if running locally (if root_dir points to external volume)
IS_LOCAL = False  # Set to True for local runs, False for cluster runs

if IS_LOCAL:
    # Locally - no parallelization
    dir_2s = '/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/DATA/EEG'
    dir_5s = ''  # Not used
    output_dir = '/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/DATA/OUTPUTS/eeg_features'
    os.makedirs(output_dir, exist_ok=True)
else:
    # Cluster
    root_dir = '/home/emmacona/links/projects/def-lippes/emmacona/Q1K_BC_HAPPEv3_ICA/'
    dir_2s = os.path.join(root_dir, '2s_epochs')
    dir_5s = os.path.join(root_dir, '5s_epochs')
    output_dir = '/home/emmacona/links/projects/def-lippes/emmacona/Q1K_BC_HAPPEv3_ICA/Features/'
    os.makedirs(output_dir, exist_ok=True)

# Scan directories for .set files
listing_2s = glob.glob(os.path.join(dir_2s, '*.set')) if EPOCH_2S else []
listing_5s = glob.glob(os.path.join(dir_5s, '*.set')) if EPOCH_5S else []

# Remove mac hidden files
files_2s = [f for f in listing_2s if not os.path.basename(f).startswith('._')]
files_5s = [f for f in listing_5s if not os.path.basename(f).startswith('._')]

# If running locally, filter to specific files only
if IS_LOCAL:
    # Process all files in the EEG directory (no filtering)
    print(f'\n*** RUNNING IN LOCAL MODE - Processing all files from EEG directory ***\n')

# Combine all files to process
all_files = []
if EPOCH_2S:
    all_files.extend([(f, '2s') for f in files_2s])
if EPOCH_5S:
    all_files.extend([(f, '5s') for f in files_5s])

nFiles = len(all_files)

if nFiles == 0:
    raise ValueError(f'No .set files found in: {dir_2s} or {dir_5s}')

print(f'\nFound {nFiles} files to process')
if EPOCH_2S:
    print(f'  - {len(files_2s)} files in 2s directory')
if EPOCH_5S:
    print(f'  - {len(files_5s)} files in 5s directory')
print()

# ------------ Common Params ------------
Fs = 1000
n_chans = 108

# Set n_cores: 1 for local (no parallelization), use SLURM value for cluster
if IS_LOCAL:
    n_cores = 1
    print('*** Local mode: Parallelization disabled (n_cores=1) ***\n')
else:
    n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    print(f'Cluster mode: Using {n_cores} cores for parallelization\n')

maxscale = 40  # maximum scale for MSE
excluded_chans = ['48','119','43','49','56','63','68','73','81','88','94','99','107','113','120','125','126','127','128','17','129']  # Already excluded after HAPPEv3_ICA
included_chans = [str(idx) for idx in range(1, 130) if str(idx) not in excluded_chans]

# ---------- CONSTANTS ----------
FREQ_BANDS = {'delta': [1,4],
              'theta': [4,8],
              'alpha': [8,13],
              'beta': [13,30],
              'gamma': [30,80],
              'low_gamma': [30,59], 
              'high_gamma': [61,80]}

# ---------- HELPER FUNCTIONS ----------
# function to compute mse
def compute_mse(data, maxscale):
    mse_temp = entropy_multiscale(data,scale=maxscale, dimension=2, r=0.15)
    return mse_temp[1]['Value']

# ---------- MAIN FUNCTION ----------
for file_idx, (filepath, epoch_type) in enumerate(all_files):
    filename = os.path.basename(filepath)

    # ----- Define output files -----
    output_base = filename.replace('.set', '').replace('_processed', '')
    out_pkl = os.path.join(output_dir, f'features_{output_base}.pkl')
    out_csv = os.path.join(output_dir, f'features_avg_{output_base}.csv')
    
    # ----- Skip files that are already processed -----
    if SKIP_PROCESSED and os.path.exists(out_pkl) and os.path.exists(out_csv):  
        print(f'\nSkipping file {file_idx+1}/{nFiles}: {filename} ({epoch_type}) - outputs already exist.\n')
        continue            
    else:
        print(f'\nProcessing file {file_idx+1}/{nFiles}: {filename} ({epoch_type})\n')
    
    features = {}
    mse_array = None
    features['channel_list'] = included_chans
    features['filename'] = filename
    features['epoch_type'] = epoch_type

    # ========== EXTRACT FEATURES FROM 2S EPOCHS ==========
    if epoch_type == '2s':
        print(f'Loading 2s epochs...')
        try:
            epochs_short = mne.io.read_epochs_eeglab(filepath)
        except Exception as e:
            print(f'Error reading 2s file {filename}: {e}')
            continue
        
        # Keep only bad channels that actually exist in this file
        bad_existing = [ch for ch in excluded_chans if ch in epochs_short.ch_names]
        epochs_short.info['bads'] = bad_existing
        print(np.shape(epochs_short.get_data(picks='all')))
        epoch_data = epochs_short.get_data(picks='all')[:,:n_chans] # excluding bad channels
        n_epochs = epoch_data.shape[0]
        features_short = ['hurst', 'pow_delta', 'pow_theta', 'pow_alpha','pow_beta', 'pow_gamma', 'pow_low_gamma', 'pow_high_gamma',
                            'fooof_offset', 'fooof_exponent', 'pow_per_delta', 'pow_per_theta', 'pow_per_alpha', 'pow_per_beta', 'pow_per_gamma',
                            'pow_per_low_gamma', 'pow_per_high_gamma']

        if n_epochs>0:
            # Power features (absolute power bands, hurst)
            print('\nExtracting non-FOOOF 2s features...')
            previous_time = time.time()
            selected_funcs = ['hurst_exp', 'pow_freq_bands']
            feat_params = {'pow_freq_bands__freq_bands': FREQ_BANDS,
                            'pow_freq_bands__normalize': False, # absolute powers
                            'pow_freq_bands__psd_method': 'welch'}
            
            fe = FeatureExtractor(sfreq=Fs, selected_funcs=selected_funcs, params=feat_params)
            feat_arr = fe.fit_transform(epoch_data)
            
            for f, feature in enumerate(features_short[:8]):
                features[feature] = feat_arr[:,n_chans*f:n_chans*(f+1)]
            
            comp_time = time.time() - previous_time
            print(f'Computation time: {comp_time:.3f}s\n')

            # FOOOF features (periodic bands, offset, and exponent)
            print('\nExtracting FOOOF features...')
            previous_time = time.time()
            spec = epochs_short.compute_psd(method='welch', fmin=0.5, fmax=80, n_jobs=n_cores) # compute PSD on epoched data
            freqs = spec.freqs
            powers = spec.get_data()
            
            # Average PSD across epochs for FOOOF fitting
            avg_powers = powers.mean(axis=0)  # average across epochs: (n_chans, n_freqs)
            
            # Fit FOOOF on average PSD
            fg = FOOOFGroup(min_peak_height=0.1, peak_width_limits=(1,12))
            fg.fit(freqs, avg_powers, freq_range=[0.5,80], n_jobs=n_cores)
            
            # Extract aperiodic parameters and periodic spectrum from average fit
            aper_params_avg = fg.get_params('aperiodic')  # (n_chans, 2) - offset and exponent
            per_spec_avg = np.zeros((n_chans, len(freqs)))
            
            for ch in range(n_chans):
                # Extract periodic spectrum by subtracting aperiodic from full spectrum
                # This captures ALL oscillatory activity, not just detected peaks
                full_spectrum = fg.get_fooof(ch).get_data(component='full',space='linear')
                aperiodic_spectrum = fg.get_fooof(ch).get_data(component='aperiodic',space='linear')
                periodic_spectrum = full_spectrum - aperiodic_spectrum
                per_spec_avg[ch] = periodic_spectrum  # Store periodic spectrum for band averaging
            
            # Store FOOOF parameters (replicated across epochs to maintain structure)
            for f, feature in enumerate(features_short[8:10]):
                features[feature] = np.tile(aper_params_avg[:, f], (n_epochs, 1))  # (n_epochs, n_chans)
            
            # Compute periodic power in frequency bands
            for feature, band_freqs in zip(features_short[10:], FREQ_BANDS.values()):
                band_mask = np.logical_and(freqs >= band_freqs[0], freqs < band_freqs[1])
                per_power_band = np.mean(per_spec_avg[:, band_mask], axis=1)  # (n_chans,)
                features[feature] = np.tile(per_power_band, (n_epochs, 1))  # (n_epochs, n_chans)
                print(f'  {feature}: range [{per_power_band.min():.6e}, {per_power_band.max():.6e}]')
            
            comp_time = time.time() - previous_time
            print(f'Computation time: {comp_time:.3f}s\n')

        else:
            print('There are no 2s epochs. Not computing 2s features.')

    # ========== EXTRACT FEATURES FROM 5S EPOCHS ==========
    elif epoch_type == '5s':
        print(f'Loading 5s epochs...')
        try:
            epochs_long = mne.io.read_epochs_eeglab(filepath)
        except Exception as e:
            print(f'Error reading 5s file {filename}: {e}')
            continue

        bad_existing = [ch for ch in excluded_chans if ch in epochs_long.ch_names]
        epochs_long.info['bads'] = bad_existing
        epoch_data = epochs_long.get_data(picks='all')[:, :n_chans]  # excluding bad channels
        n_epochs = epoch_data.shape[0]
        selected_funcs = ['higuchi_fd', 'katz_fd', 'samp_entropy']
        features_long = ['higuchi_fd', 'katz_fd', 'samp_entropy',
                            'CI', 'CI_lowscale', 'CI_highscale']
        feat_params = {'higuchi_fd__kmax': 8}

        if n_epochs>0:
            # get non-MSE 5s features
            print('\nExtracting non-MSE 5s features...')
            previous_time = time.time()
            fe = FeatureExtractor(sfreq=Fs, selected_funcs=selected_funcs, params=feat_params)
            feat_arr = fe.fit_transform(epoch_data)
            for f, feature in enumerate(features_long[:3]):
                features[feature] = feat_arr[:,n_chans*f:n_chans*(f+1)]
            comp_time = time.time() - previous_time
            print(f'Computation time: {comp_time:.3f}s\n')

            # MSE/CI
            print('\nExtracting MSE features...')
            previous_time = time.time()
            scales = np.arange(1,maxscale+1)
            joblist = []
            for tr in range(n_epochs):
                for ch in range(n_chans):
                    joblist.append(delayed(compute_mse)(epoch_data[tr,ch],maxscale))
            with Parallel(n_jobs=n_cores) as parallel:
                results = parallel(joblist)
            mse_array = np.array(results).reshape((n_epochs,n_chans,len(scales)))
            features['CI'] = np.trapz(mse_array, x=scales, axis=-1)
            features['CI_lowscale'] = np.trapz(mse_array[:, :, :maxscale//2], x=scales[:maxscale//2], axis=-1)
            features['CI_highscale'] = np.trapz(mse_array[:, :, maxscale//2:], x=scales[maxscale//2:], axis=-1)
            comp_time = time.time() - previous_time
            print(f'Computation time: {comp_time:.3f}s\n')

        else:
            print('There are no 5s epochs. Not computing 5s features.')

    # ========== SAVE OUTPUTS ==========
    # Create output filename base (remove .set extension)
    output_base = filename.replace('.set', '').replace('_processed', '')
    
    # save complete MSE values in separate file
    if mse_array is not None:
        np.save(os.path.join(output_dir, f'mse_{output_base}.npy'), mse_array)

    # save feature dictionary in pkl file (all segments)
    with open(os.path.join(output_dir, f'features_{output_base}.pkl'), 'wb') as file:
        pickle.dump(features, file)

    # average across segments, convert and save to CSV
    feature_names = [key for key in features.keys() if key not in ['channel_list', 'filename', 'epoch_type']] # get feature names
    if len(feature_names) > 0:
        header = 'channel,'
        data_avg = np.zeros((len(feature_names)+1,n_chans))
        k = 0
        data_avg[0] = included_chans
        for feature in feature_names:
            k += 1
            header += f'{feature},'
            try:
                data_avg[k] = features[feature].mean(0) # average across segments
            except:
                data_avg[k] = np.nan # nan because not computed
        header = header[:-1] # remove last comma
        np.savetxt(os.path.join(output_dir, f'features_avg_{output_base}.csv'), data_avg.T, delimiter=',', header=header, comments='')
    else:
        print(f'Warning: No features computed for {filename}, skipping CSV output.')


total_time = time.time() - start_time
print(f'\n{"="*60}')
print(f'Completed processing {nFiles} files')
print(f'Total time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)')
if nFiles > 0:
    print(f'Average time per file: {total_time/nFiles:.1f} seconds')
print(f'Output directory: {output_dir}')
print(f'{"="*60}\n')