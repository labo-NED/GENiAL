import os
import numpy as np
import mne
from scipy.signal import welch
from scipy.signal.windows import hamming
import pickle
import argparse

# === Helper Functions ===

def estimate_hurst_exponent(signal):
    # Placeholder: Replace with your actual Hurst exponent estimation
    # e.g., use hurst from hurst package or implement your own
    from hurst import compute_Hc
    H, c, data = compute_Hc(signal, kind='price', simplified=True)
    return H

def fooof_placeholder(f, pxx, f_range, settings):
    # Placeholder: Replace with actual FOOOF implementation
    # Return a dict with the same keys as in MATLAB
    # You can use the fooof Python package for real implementation
    return {
        'ap_fit': np.zeros_like(pxx),
        'power_spectrum': pxx,
        'fooofed_spectrum': pxx,
        'aperiodic_params': [0, 0]
    }

def compute_entropy(signal):
    # Placeholder for entropy calculation
    return np.nan

def compute_mse(signal):
    # Placeholder for MSE calculation
    return np.nan

# === EXAMPLE COMMANDS ===
# To run with 2s preprocessed Q1K data: `python Scripts/feature_extraction.py --hurst --fooof --bandpower --directory "/Volumes/NED_Backup3/Q1K_Preprocessed_2s_Happe/5 - processed"`
# To run with 2s preprocessed BC data: `python Scripts/feature_extraction.py --hurst --fooof --bandpower --directory "/Volumes/NED_Backup3/BC_preprocessed/2s-preprocessed/5 - processed"`

# To run with 5s preprocessed Q1K data: `python Scripts/feature_extraction.py --entropy --mse --directory "/Volumes/NED_Backup3/Q1K_Preprocessed_5s_Happe/5 - processed"`
# To run with 5s preprocessed BC data: `python Scripts/feature_extraction.py --entropy --mse --directory "/Volumes/NED_Backup3/BC_preprocessed/5s-preprocessed/5 - processed"`

# === Argument Parsing ===
parser = argparse.ArgumentParser(description='EEG Feature Extraction')
parser.add_argument('--hurst', action='store_true', help='Extract Hurst exponent')
parser.add_argument('--fooof', action='store_true', help='Extract FOOOF features')
parser.add_argument('--bandpower', action='store_true', help='Extract band power features')
parser.add_argument('--entropy', action='store_true', help='Extract entropy features')
parser.add_argument('--mse', action='store_true', help='Extract multiscale entropy (MSE) features')
parser.add_argument('--directory', type=str, default=os.getcwd(), help='Directory with EEG .set files')
args = parser.parse_args()

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
f_range = [1, 40]
settings = {'peak_width_limits': [0.5, 18], 'max_n_peaks': 10}

# === List EEG files ===
file_names = [f for f in os.listdir(args.directory) if f.endswith('.set') and not f.startswith('._')]

# === Initialize feature containers ===
feature = {}
fooof_results_t = {}
aperiodic_component_t = {}
full_spectrum_t = {}
periodic_component_t = {}
fooofed_spectrum_t = {}
offset = []
exponent = []
periodic_PSD = []
periodic_PSD_m = []
PSD = []
relative_PSD = []
hurst = []
pxx_t = []
names = []
entropy_vals = []
mse_vals = []
# Optional fields
APF_fooof = []
APF_fooof_ROI = []
sgf_t = []

# === Loop through each participant ===
for sub, fname in enumerate(file_names):
    names.append(fname)
    epochs = mne.io.read_epochs_eeglab(os.path.join(args.directory, fname))
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    Fs = int(epochs.info['sfreq'])
    n_epochs, n_channels, n_times = data.shape

    # Loop through each epoch and channel
    for epoch in range(n_epochs):
        for ch in range(n_channels):
            signal = data[epoch, ch, :]
            # --- Hurst Exponent ---
            if args.hurst:
                hurst1 = estimate_hurst_exponent(signal)
                hurst.append(hurst1)

            # --- Power Spectral Density (Welch) ---
            window = hamming(1000)
            N = len(signal) // 2
            noverlap = len(window) // 2
            f, pxx1 = welch(signal, fs=Fs, window=window, nperseg=len(window), noverlap=noverlap, nfft=N)
            pxx_t.append(pxx1)

            # --- FOOOF Spectral Decomposition ---
            if args.fooof:
                fooof_results = fooof_placeholder(f, pxx1, f_range, settings)
                fooof_results_t[(sub, epoch, ch)] = fooof_results

                # --- Absolute Band Power ---
                f1 = np.where((f >= 1) & (f <= 3.5))[0]     # Delta
                f2 = np.where((f >= 4) & (f <= 7))[0]       # Theta
                f3 = np.where((f >= 7.5) & (f <= 12.5))[0]  # Alpha
                f4 = np.where((f >= 13) & (f <= 30))[0]     # Beta
                f5 = np.where((f >= 31) & (f <= 57))[0]     # Gamma (low)
                f6 = np.where((f >= 62) & (f <= 80))[0]     # Gamma (high)
                f7 = np.where((f >= 0.5) & (f <= 80))[0]    # Full spectrum

                PSD_row = [
                    np.mean(pxx1[f1]),
                    np.mean(pxx1[f2]),
                    np.mean(pxx1[f3]),
                    np.mean(pxx1[f4]),
                    np.mean(pxx1[f5]),
                    np.mean(pxx1[f6]),
                    np.mean(pxx1[f7])
                ]
                PSD.append(PSD_row)

                # --- Relative Band Power ---
                rel_row = [PSD_row[band] / sum(PSD_row) for band in range(len(PSD_row)-1)]
                relative_PSD.append(rel_row)

                # --- FOOOF Components ---
                aperiodic_component = fooof_results['ap_fit']
                full_spectrum = fooof_results['power_spectrum']
                periodic_component = full_spectrum - aperiodic_component

                aperiodic_component_t[(sub, epoch, ch)] = aperiodic_component
                full_spectrum_t[(sub, epoch, ch)] = full_spectrum
                periodic_component_t[(sub, epoch, ch)] = periodic_component
                fooofed_spectrum_t[(sub, epoch, ch)] = fooof_results['fooofed_spectrum']

                offset.append(fooof_results['aperiodic_params'][0])
                exponent.append(fooof_results['aperiodic_params'][1])

                # --- Periodic Band Power (AUC and Mean) ---
                periodic_PSD_row = [
                    np.trapz(periodic_component[f1]),
                    np.trapz(periodic_component[f2]),
                    np.trapz(periodic_component[f3]),
                    np.trapz(periodic_component[f4]),
                    np.trapz(periodic_component[f5[f5<=40]])
                ]
                periodic_PSD.append(periodic_PSD_row)

                periodic_PSD_m_row = [
                    np.mean(periodic_component[f1]),
                    np.mean(periodic_component[f2]),
                    np.mean(periodic_component[f3]),
                    np.mean(periodic_component[f4]),
                    np.mean(periodic_component[f5[f5<=40]])
                ]
                periodic_PSD_m.append(periodic_PSD_m_row)

            # --- Entropy ---
            if args.entropy:
                entropy_val = compute_entropy(signal)
                entropy_vals.append(entropy_val)

            # --- MSE ---
            if args.mse:
                mse_val = compute_mse(signal)
                mse_vals.append(mse_val)

            # --- Optional fields (placeholders) ---
            # If you have these, compute and append them here
            # APF_fooof.append(...)
            # APF_fooof_ROI.append(...)
            # sgf_t.append(...)

# === Save features ===
feature['name'] = names
if args.hurst:
    feature['hurst'] = np.array(hurst)
if args.fooof:
    feature['fooof_results_t'] = fooof_results_t
    feature['aperiodic_component_t'] = aperiodic_component_t
    feature['full_spectrum_t'] = full_spectrum_t
    feature['periodic_component_t'] = periodic_component_t
    feature['fooofed_spectrum_t'] = fooofed_spectrum_t
    feature['offset'] = np.array(offset)
    feature['exponent'] = np.array(exponent)
if args.bandpower:
    feature['PSD'] = np.array(PSD)
    feature['relative_PSD'] = np.array(relative_PSD)
    if args.fooof:
        feature['periodic_PSD'] = np.array(periodic_PSD)
        feature['periodic_PSD_m'] = np.array(periodic_PSD_m)
feature['pxx_t'] = np.array(pxx_t)
if args.entropy:
    feature['entropy'] = np.array(entropy_vals)
if args.mse:
    feature['mse'] = np.array(mse_vals)
# Optional fields
if len(APF_fooof) > 0:
    feature['APF_fooof'] = APF_fooof
if len(APF_fooof_ROI) > 0:
    feature['APF_fooof_ROI'] = APF_fooof_ROI
if len(sgf_t) > 0:
    feature['sgf_t'] = sgf_t

with open(os.path.join(args.directory, 'feature.pkl'), 'wb') as f:
    pickle.dump(feature, f)

print('Feature extraction complete and saved to feature.pkl')