import os
import numpy as np
import mne
from scipy.signal import welch, windows
import pickle
import pandas as pd
from mne_features.feature_extraction import FeatureExtractor
from fooof import FOOOF
from neurokit2 import entropy_multiscale

def estimate_hurst_exponent(signal):
    """
    Estimate Hurst exponent using mne-features (same as Cinema project)
    """
    try:
        # Use mne-features for consistent Hurst calculation
        fe = FeatureExtractor(sfreq=1000, selected_funcs=['hurst_exp'])
        # Reshape signal to match expected format (n_epochs, n_channels, n_times)
        signal_reshaped = signal.reshape(1, 1, -1)
        hurst_value = fe.fit_transform(signal_reshaped)
        return hurst_value[0, 0]  # Extract scalar value
    except Exception as e:
        print(f"Warning: Hurst calculation failed: {e}")
        return np.nan

def compute_mse(data, maxscale=20):
    """
    Compute multiscale entropy using neurokit2 (same as Cinema project)
    """
    try:
        mse_temp = entropy_multiscale(data, scale=maxscale, dimension=2, r=0.15)
        return mse_temp[1]['Value']  # Return MSE values across scales
    except Exception as e:
        print(f"Warning: MSE calculation failed: {e}")
        return np.full(maxscale, np.nan)

def fooof_fit(f, pxx, f_range, settings):
    """
    Fit FOOOF model to power spectrum (same as Cinema project)
    """
    try:
        # Initialize FOOOF model with settings
        fm = FOOOF(
            min_peak_height=settings.get('min_peak_height', 0.1),
            peak_width_limits=settings.get('peak_width_limits', [0.5, 18]),
            max_n_peaks=settings.get('max_n_peaks', 10)
        )
        
        # Fit the model
        fm.fit(f, pxx, f_range)
        
        # Get aperiodic fit
        ap_fit = fm._ap_fit(f)
        
        # Get periodic component (peaks)
        fooofed_spectrum = fm.power_spectrum_
        
        # Get aperiodic parameters [offset, exponent]
        aperiodic_params = fm.get_params('aperiodic')
        
        return {
            'ap_fit': ap_fit,
            'power_spectrum': pxx,
            'fooofed_spectrum': fooofed_spectrum,
            'aperiodic_params': aperiodic_params
        }
    except Exception as e:
        print(f"Warning: FOOOF fitting failed: {e}")
        # Return placeholder values on failure
        return {
            'ap_fit': np.zeros_like(pxx),
            'power_spectrum': pxx,
            'fooofed_spectrum': pxx,
            'aperiodic_params': [0, 0]
        }

# --- Load EEG Files ---

## TO BE UPDATED MANUALLY ##
directory = '/Volumes/NED_Backup3/COMBINED_Q1K_BC_2s'
project_folder = 'COMBINED_Q1K_BC_2s'
entropy = False
mse = False
hurst = False
fooof = True
bandpower = True
############################

file_names = [f for f in os.listdir(directory) if f.endswith('.set')]
names = [f for f in file_names if not f.startswith('._')]  # Remove macOS hidden files

f_range = [1, 40]
settings = {'peak_width_limits': [0.5, 18], 'max_n_peaks': 10}

# Initialize containers
aperiodic_component_t = {}
fooof_results_t = {}
fooofed_spectrum_t = {}
full_spectrum_t = {}
offset = {}
exponent = {}
periodic_component_t = {}
periodic_PSD = {}
periodic_PSD_m = {}
PSD = {}
relative_PSD = {}
hurst = {}
pxx_t = {}

# Loop through each participant
for sub, id in enumerate(names):
    eeg = mne.io.read_raw_eeglab(os.path.join(directory, id), preload=True)
    EEG_label = eeg.ch_names
    Fs = int(eeg.info['sfreq'])
    data = eeg.get_data()  # shape: (n_channels, n_times)
    n_chans = data.shape[0]
    n_trials = 1
    if data.ndim == 3:
        n_trials = data.shape[2]
    else:
        n_trials = 1
        data = data[:, :, np.newaxis]  # (n_channels, n_times, 1)

    for ch in range(n_chans):
        # Initialize variables for this channel
        hurst1 = []
        pxx1 = []
        f = None
        pxx_mean = None
        fooof_results = None
        
        # --- Multiscale Entropy (MSE) & Complexity Index ---
        if mse:
            maxscale = 20
            mse_values = []
            for tr in range(n_trials):
                signal = data[ch, :, tr]
                mse_vals = compute_mse(signal, maxscale)
                mse_values.append(mse_vals)
            mse_mean = np.mean(mse_values, axis=0)
            scales = np.arange(1, maxscale + 1)
            CI = np.trapz(mse_mean, scales)
            CI_lowScale = np.trapz(mse_mean[:maxscale//2], scales[:maxscale//2])
            CI_highScale = np.trapz(mse_mean[maxscale//2:], scales[maxscale//2:])
            # Store MSE results (you may want to add these to feature dictionary)
            print(f"Channel {ch}: MSE computed - CI: {CI:.4f}")

        # --- Sample Entropy ---
        if entropy:
            sampe_values = []
            for tr in range(n_trials):
                signal = data[ch, :, tr]
                # Add sample entropy calculation here if needed
                # sampe_values.append(sampen(signal, 2, 0.5))
            # SE_mean = np.mean(sampe_values)
            print(f"Channel {ch}: Sample entropy computation not yet implemented")

        # Process signals for other features
        for tr in range(n_trials):
            signal = data[ch, :, tr]

            # --- Hurst Exponent ---
            if hurst:
                hurst1.append(estimate_hurst_exponent(signal))

            # --- Power Spectral Density (Welch) ---
            # Always compute PSD as it's needed for FOOOF and band power
            window = windows.hamming(1000)
            N = int(len(signal) / 2)
            noverlap = int(len(window) / 2)
            f, pxx = welch(signal, fs=Fs, window=window, nperseg=len(window), noverlap=noverlap, nfft=N)
            pxx1.append(pxx)

        # Store Hurst results
        if hurst:
            hurst.setdefault(ch, {})[sub] = np.nanmean(hurst1)

        # Store PSD results
        pxx1 = np.array(pxx1)
        pxx_mean = np.mean(pxx1, axis=0)
        pxx_t.setdefault(ch, {})[sub] = pxx_mean

        # --- FOOOF Spectral Decomposition ---
        if fooof:
            fooof_results = fooof_fit(f, pxx_mean, f_range, settings)
            fooof_results_t.setdefault(sub, {})[ch] = fooof_results

        # --- Band Power Calculations ---
        if bandpower:
            # Define frequency bands
            f1 = np.where((f >= 1) & (f <= 3.5))[0]      # Delta
            f2 = np.where((f >= 4) & (f <= 7))[0]        # Theta
            f3 = np.where((f >= 7.5) & (f <= 12.5))[0]   # Alpha
            f4 = np.where((f >= 13) & (f <= 30))[0]      # Beta
            f5 = np.where((f >= 31) & (f <= 57))[0]      # Gamma (low)
            f6 = np.where((f >= 62) & (f <= 80))[0]      # Gamma (high)
            f7 = np.where((f >= 0.5) & (f <= 80))[0]     # Full spectrum

            # --- Absolute Band Power ---
            PSD.setdefault(sub, {}).setdefault(ch, {})[1] = np.mean(pxx_mean[f1]) if len(f1) > 0 else np.nan
            PSD[sub][ch][2] = np.mean(pxx_mean[f2]) if len(f2) > 0 else np.nan
            PSD[sub][ch][3] = np.mean(pxx_mean[f3]) if len(f3) > 0 else np.nan
            PSD[sub][ch][4] = np.mean(pxx_mean[f4]) if len(f4) > 0 else np.nan
            PSD[sub][ch][5] = np.mean(pxx_mean[f5]) if len(f5) > 0 else np.nan
            PSD[sub][ch][6] = np.mean(pxx_mean[f6]) if len(f6) > 0 else np.nan
            PSD[sub][ch][7] = np.mean(pxx_mean[f7]) if len(f7) > 0 else np.nan

            # --- Relative Band Power ---
            psd_vals = [PSD[sub][ch][i] for i in range(1, 8)]
            psd_sum = np.nansum(psd_vals)
            for band in range(1, 7):
                if psd_sum > 0:
                    relative_PSD.setdefault(sub, {}).setdefault(ch, {})[band] = PSD[sub][ch][band] / psd_sum
                else:
                    relative_PSD.setdefault(sub, {}).setdefault(ch, {})[band] = np.nan

        # --- FOOOF Components (only if FOOOF was computed) ---
        if fooof and fooof_results is not None:
            aperiodic_component = fooof_results['ap_fit']
            full_spectrum = fooof_results['power_spectrum']
            periodic_component = full_spectrum - aperiodic_component

            aperiodic_component_t.setdefault(sub, {})[ch] = aperiodic_component
            full_spectrum_t.setdefault(sub, {})[ch] = full_spectrum
            periodic_component_t.setdefault(sub, {})[ch] = periodic_component
            fooofed_spectrum_t.setdefault(sub, {})[ch] = fooof_results['fooofed_spectrum']

            offset.setdefault(sub, {})[ch] = fooof_results['aperiodic_params'][0]
            exponent.setdefault(sub, {})[ch] = fooof_results['aperiodic_params'][1]

            # --- Periodic Band Power (AUC and Mean) ---
            if bandpower:
                periodic_PSD.setdefault(sub, {}).setdefault(ch, {})[1] = np.trapz(periodic_component[f1]) if len(f1) > 0 else np.nan
                periodic_PSD[sub][ch][2] = np.trapz(periodic_component[f2]) if len(f2) > 0 else np.nan
                periodic_PSD[sub][ch][3] = np.trapz(periodic_component[f3]) if len(f3) > 0 else np.nan
                periodic_PSD[sub][ch][4] = np.trapz(periodic_component[f4]) if len(f4) > 0 else np.nan
                # For f5(f5<=40): restrict to f <= 40
                f5_40 = f5[f[f5] <= 40]
                periodic_PSD[sub][ch][5] = np.trapz(periodic_component[f5_40]) if len(f5_40) > 0 else np.nan

                periodic_PSD_m.setdefault(sub, {}).setdefault(ch, {})[1] = np.mean(periodic_component[f1]) if len(f1) > 0 else np.nan
                periodic_PSD_m[sub][ch][2] = np.mean(periodic_component[f2]) if len(f2) > 0 else np.nan
                periodic_PSD_m[sub][ch][3] = np.mean(periodic_component[f3]) if len(f3) > 0 else np.nan
                periodic_PSD_m[sub][ch][4] = np.mean(periodic_component[f4]) if len(f4) > 0 else np.nan
                periodic_PSD_m[sub][ch][5] = np.mean(periodic_component[f5_40]) if len(f5_40) > 0 else np.nan

# Save only variables that exist and were computed
feature = {}
feature['name'] = names

# Always save PSD as it's computed for all features
feature['pxx_t'] = pxx_t

# Save features based on boolean flags
if hurst:
    feature['hurst'] = hurst

if fooof:
    feature['aperiodic_component_t'] = aperiodic_component_t
    feature['fooof_results_t'] = fooof_results_t
    feature['fooofed_spectrum_t'] = fooofed_spectrum_t
    feature['full_spectrum_t'] = full_spectrum_t
    feature['offset'] = offset
    feature['exponent'] = exponent
    feature['periodic_component_t'] = periodic_component_t

if bandpower:
    feature['PSD'] = PSD
    feature['relative_PSD'] = relative_PSD

# Periodic features depend on both FOOOF and bandpower
if fooof and bandpower:
    feature['periodic_PSD'] = periodic_PSD
    feature['periodic_PSD_m'] = periodic_PSD_m

# Print summary of computed features
print(f"\nFeature extraction summary:")
print(f"- Hurst exponent: {'✓' if hurst else '✗'}")
print(f"- FOOOF spectral decomposition: {'✓' if fooof else '✗'}")
print(f"- Band power analysis: {'✓' if bandpower else '✗'}")
print(f"- MSE/Complexity Index: {'✓' if mse else '✗'}")
print(f"- Sample entropy: {'✓' if entropy else '✗'}")
print(f"- Total features saved: {len(feature)}")

# Only include optional fields if they exist
if 'APF_fooof' in locals():
    feature['APF_fooof'] = APF_fooof
if 'APF_fooof_ROI' in locals():
    feature['APF_fooof_ROI'] = APF_fooof_ROI
if 'sgf_t' in locals():
    feature['sgf_t'] = sgf_t

# Save as pickle file
with open(os.path.join(directory, f'{project_folder}_feature.pkl'), 'wb') as f:
    pickle.dump(feature, f)

# Save as CSV file

def flatten_feature_dict(feature_dict, names):
    """
    Flatten the nested feature dictionary into a format suitable for CSV export
    """
    csv_data = []
    
    for sub_idx, name in enumerate(names):
        row = {'subject_id': name, 'subject_index': sub_idx}
        
        # Get all channels for this subject
        channels = set()
        for feature_name, feature_data in feature_dict.items():
            if feature_name in ['name']:
                continue
            if isinstance(feature_data, dict):
                for ch_idx in feature_data.keys():
                    if isinstance(feature_data[ch_idx], dict) and sub_idx in feature_data[ch_idx]:
                        channels.add(ch_idx)
        
        # For each channel, extract all features
        for ch_idx in sorted(channels):
            for feature_name, feature_data in feature_dict.items():
                if feature_name in ['name']:
                    continue
                
                try:
                    if isinstance(feature_data, dict) and ch_idx in feature_data:
                        if isinstance(feature_data[ch_idx], dict) and sub_idx in feature_data[ch_idx]:
                            value = feature_data[ch_idx][sub_idx]
                            
                            # Handle different data types
                            if isinstance(value, np.ndarray):
                                if value.ndim == 1:
                                    # For 1D arrays (like PSD bands), create separate columns
                                    for i, val in enumerate(value):
                                        col_name = f'ch{ch_idx}_{feature_name}_band{i+1}'
                                        row[col_name] = val
                                else:
                                    # For higher dimensional arrays, take mean
                                    row[f'ch{ch_idx}_{feature_name}'] = np.nanmean(value)
                            elif isinstance(value, (list, tuple)):
                                # For lists/tuples, take mean
                                row[f'ch{ch_idx}_{feature_name}'] = np.nanmean(value)
                            else:
                                # For scalar values
                                row[f'ch{ch_idx}_{feature_name}'] = value
                except Exception as e:
                    # Skip problematic entries
                    continue
        
        csv_data.append(row)
    
    return pd.DataFrame(csv_data)

# Create CSV export
try:
    df = flatten_feature_dict(feature, names)
    csv_path = os.path.join(directory, f'{project_folder}_feature_extraction_results.csv')
    df.to_csv(csv_path, index=False)
    print(f'CSV export complete: {csv_path}')
    print(f'CSV contains {len(df)} rows and {len(df.columns)} columns')
except Exception as e:
    print(f'Warning: CSV export failed: {e}')
    print('Pickle file saved successfully, but CSV export encountered an error')

print('Feature extraction complete and saved to feature.pkl')
