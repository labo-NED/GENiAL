### COMPUTE FEATURES IN PYTHON FOR GENiAL PROJECT
# Adapted from Laurent Caplette (2025) based on code by Saeideh Davoodi
# Emmanuelle Coutu-Nadeau (2025) - adapted for Q1K & BC data

import numpy as np
import mne
from mne_features.feature_extraction import FeatureExtractor
from matplotlib import pyplot as plt
from scipy.signal import welch
from fooof import FOOOF, FOOOFGroup, fit_fooof_3d
from fooof.objs import combine_fooofs
import pickle, csv, time
from neurokit2 import entropy_multiscale
from joblib import Parallel, delayed
import argparse
import os

start_time = time.time()

# === EXAMPLE COMMANDS ===
# To run with 2s preprocessed: python3 Scripts/feature_extraction.py --fooof --bandpower --directory "/Volumes/NED_Backup3/COMBINED_Q1K_BC_2s" --output-dir "/Volumes/NED_Backup3/Features/"

# To run with 5s preprocessed: python3 Scripts/feature_extraction.py --entropy --mse --directory "/Volumes/NED_Backup3/COMBINED_Q1K_BC_5s" --output-dir "/Volumes/NED_Backup3/Features/"


# === Argument Parsing ===
parser = argparse.ArgumentParser(description='EEG Feature Extraction for GENiAL Project')
parser.add_argument('--hurst', action='store_true', help='Extract Hurst exponent')
parser.add_argument('--fooof', action='store_true', help='Extract FOOOF features')
parser.add_argument('--bandpower', action='store_true', help='Extract band power features')
parser.add_argument('--entropy', action='store_true', help='Extract entropy features')
parser.add_argument('--mse', action='store_true', help='Extract multiscale entropy (MSE) features')
parser.add_argument('--directory', type=str, default=os.getcwd(), help='Directory with EEG .set files')
parser.add_argument('--output-dir', type=str, help='Output directory for features (default: same as input directory)')
args = parser.parse_args()

# Set output directory
if args.output_dir:
    output_dir = args.output_dir
else:
    output_dir = args.directory

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Print which features are being extracted
print('Extracting features:')
if args.hurst:
    print(' - Hurst exponent')
if args.fooof:
    print(' - FOOOF features')
if args.bandpower:
    print(' - Band power features')
if args.entropy:
    print(' - Entropy')
if args.mse:
    print(' - Multiscale Entropy (MSE)')

# === Parameters ===
Fs = 1000
n_cores = 4  # Adjust based on your system
maxscale = 40  # maximum scale for MSE

# function to compute mse
def compute_mse(data, maxscale):
    mse_temp = entropy_multiscale(data,scale=maxscale, dimension=2, r=0.15)
    return mse_temp[1]['Value']

# === List EEG files ===
file_names = [f for f in os.listdir(args.directory) if f.endswith('.set') and not f.startswith('._')]

# === Initialize feature containers ===
features = {}
mse = {}

# === Loop through each participant ===
for sub, fname in enumerate(file_names):
    print(f'\nExtracting features for {fname}\n')
    
    # Load epochs data
    epochs = mne.io.read_epochs_eeglab(os.path.join(args.directory, fname))
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    Fs = int(epochs.info['sfreq'])
    n_epochs, n_channels, n_times = data.shape
    
    # Initialize features for this subject
    features[fname] = {}

    if n_epochs > 0:
        print(f'Processing {n_epochs} epochs with {n_channels} channels')
        
        # Extract features based on command line arguments
        if args.hurst or args.bandpower:
            print('\nExtracting basic features...')
            previous_time = time.time()
            
            # Define frequency bands
            freq_bands = {'delta': [1,4],
                         'theta': [4,8],
                         'alpha': [8,13],
                         'beta': [13,30],
                         'gamma': [30,80],
                         'low_gamma': [30,59], 
                         'high_gamma': [61,80]}
            
            # Extract features using mne-features
            selected_funcs = []
            if args.hurst:
                selected_funcs.append('hurst_exp')
            if args.bandpower:
                selected_funcs.append('pow_freq_bands')
            
            if selected_funcs:
                feat_params = {'pow_freq_bands__freq_bands': freq_bands,
                              'pow_freq_bands__normalize': False,  # absolute powers
                              'pow_freq_bands__psd_method': 'welch'}
                fe = FeatureExtractor(sfreq=Fs, selected_funcs=selected_funcs, params=feat_params)
                feat_arr = fe.fit_transform(data)
                
                # Store features
                if args.hurst:
                    features[fname]['hurst'] = feat_arr[:, :n_channels]
                if args.bandpower:
                    features[fname]['pow_delta'] = feat_arr[:, n_channels*1:n_channels*2]
                    features[fname]['pow_theta'] = feat_arr[:, n_channels*2:n_channels*3]
                    features[fname]['pow_alpha'] = feat_arr[:, n_channels*3:n_channels*4]
                    features[fname]['pow_beta'] = feat_arr[:, n_channels*4:n_channels*5]
                    features[fname]['pow_gamma'] = feat_arr[:, n_channels*5:n_channels*6]
                    features[fname]['pow_low_gamma'] = feat_arr[:, n_channels*6:n_channels*7]
                    features[fname]['pow_high_gamma'] = feat_arr[:, n_channels*7:n_channels*8]
            
            comp_time = time.time() - previous_time
            print(f'Basic features computation time: {comp_time:.3f}s\n')

        # FOOOF features
        if args.fooof:
            print('\nExtracting FOOOF features...')
            previous_time = time.time()
            
            # Compute PSD
            spec = epochs.compute_psd(method='welch', fmin=0.5, fmax=80, n_jobs=n_cores)
            freqs = spec.freqs
            powers = spec.get_data()
            
            # Fit FOOOF
            fg = FOOOFGroup(min_peak_height=0.1, peak_width_limits=(1,12))
            fgs = fit_fooof_3d(fg, freqs, powers, freq_range=[0.5,80], n_jobs=n_cores)
            
            # Extract aperiodic parameters
            aper_params = np.zeros((n_epochs, n_channels, 2))
            per_spec = np.zeros(powers.shape)
            
            for tr in range(n_epochs):
                aper_params[tr] = fgs[tr].get_params('aperiodic')
                for ch in range(n_channels):
                    per_spec[tr, ch] = fgs[tr].get_fooof(ch).get_data(component='peak', space='linear')
            
            features[fname]['fooof_offset'] = aper_params[:, :, 0]
            features[fname]['fooof_exp'] = aper_params[:, :, 1]
            
            # Extract periodic band powers
            for band_name, band_freqs in freq_bands.items():
                band_mask = np.logical_and(freqs >= band_freqs[0], freqs < band_freqs[1])
                features[fname][f'pow_per_{band_name}'] = np.mean(per_spec[:, :, band_mask], axis=2)
            
            comp_time = time.time() - previous_time
            print(f'FOOOF computation time: {comp_time:.3f}s\n')

        # Entropy features
        if args.entropy:
            print('\nExtracting entropy features...')
            previous_time = time.time()
            
            selected_funcs = ['higuchi_fd', 'katz_fd', 'samp_entropy']
            feat_params = {'higuchi_fd__kmax': 8}
            fe = FeatureExtractor(sfreq=Fs, selected_funcs=selected_funcs, params=feat_params)
            feat_arr = fe.fit_transform(data)
            
            features[fname]['higuchi_fd'] = feat_arr[:, :n_channels]
            features[fname]['katz_fd'] = feat_arr[:, n_channels:n_channels*2]
            features[fname]['samp_entropy'] = feat_arr[:, n_channels*2:n_channels*3]
            
            comp_time = time.time() - previous_time
            print(f'Entropy computation time: {comp_time:.3f}s\n')

        # MSE features
        if args.mse:
            print('\nExtracting MSE features...')
            previous_time = time.time()
            
            scales = np.arange(1, maxscale+1)
            joblist = []
            for tr in range(n_epochs):
                for ch in range(n_channels):
                    joblist.append(delayed(compute_mse)(data[tr, ch], maxscale))
            
            with Parallel(n_jobs=n_cores) as parallel:
                results = parallel(joblist)
            
            mse[fname] = np.array(results).reshape((n_epochs, n_channels, len(scales)))
            features[fname]['CI'] = np.trapezoid(mse[fname], scales, axis=-1)
            features[fname]['CI_lowscale'] = np.trapezoid(mse[fname][:, :, :maxscale//2], scales[:maxscale//2], axis=-1)
            features[fname]['CI_highscale'] = np.trapezoid(mse[fname][:, :, maxscale//2:], scales[maxscale//2:], axis=-1)
            
            comp_time = time.time() - previous_time
            print(f'MSE computation time: {comp_time:.3f}s\n')

    else:
        print('No epochs found for this file.')

# === Save features ===
print('\nSaving features...')

# Save complete MSE values if computed
if mse:
    np.save(os.path.join(output_dir, 'mse_features.npy'), mse)

# Save feature dictionary
with open(os.path.join(output_dir, 'features.pkl'), 'wb') as file:
    pickle.dump(features, file)

# Save features to CSV
if features:
    import pandas as pd
    
    # Flatten the features dictionary to a DataFrame
    csv_data = []
    
    for fname in file_names:
        if fname in features:
            for epoch_idx in range(features[fname][list(features[fname].keys())[0]].shape[0]):
                for ch_idx in range(features[fname][list(features[fname].keys())[0]].shape[1]):
                    row = {
                        'filename': fname,
                        'epoch': epoch_idx,
                        'channel': ch_idx
                    }
                    
                    # Add all feature values for this epoch/channel
                    for feature_name, feature_data in features[fname].items():
                        row[feature_name] = feature_data[epoch_idx, ch_idx]
                    
                    csv_data.append(row)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(csv_data)
    df.to_csv(os.path.join(output_dir, 'features.csv'), index=False)

total_time = time.time() - start_time
print(f'\nTotal time: {total_time:.3f}s\n')