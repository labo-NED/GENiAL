# Date created: 12/18/2024
# Author: Emmanuelle CN
# 
# Note - SNR is defined as the ratio of signal power to noise power

#### Imports
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Parameters
input_folder = 'Data/6Hz'
output_folder = 'Output/6Hz'
snr_file = 'Output/SNR_output.csv'
frequency = 6 # 6 or 40Hz

# Get .set EEG file paths from input_folder
file_paths = [os.path.join(input_folder, f) 
              for f in os.listdir(input_folder) 
              if f.endswith('.set')]

snr_results = []

# Process each EEG file from the folder
for file_path in file_paths:
    # Load EEG data
    epochs = mne.io.read_epochs_eeglab(file_path)
    data = epochs.get_data()  # Shape is (n_epochs, n_channels, n_times)
    sfreq = epochs.info['sfreq']  # Sampling frequency

    # Filename
    base_name = os.path.basename(file_path)
    base_name = os.path.splitext(base_name)[0]  # Remove the extension
    parts = base_name.split('_')
    filename = '_'.join(parts[:4]) # Keep only Q1K_HSJ_XXXX-XXXX_M1/S1...

    # Process mean of 128 channels to begin (for simplicity)
    eeg_data = np.mean(data, axis=(0, 1))  # Mean across epochs and channels

    # Compute Fourier Transform (time -> freq)
    n = len(eeg_data)
    freqs = np.fft.fftfreq(n, d=1/sfreq)
    fft_values = np.fft.fft(eeg_data)
    power_spectrum = np.abs(fft_values) ** 2

    # Signal and noise
    signal_idx = np.where((freqs > frequency - 0.5) & (freqs < frequency + 0.5))[0] # +/- 0.5 Hz Bandwidth
    # Note - optimize this parameter / play with it, could be more 0.8, 1..
    # TODO : optimize + visualize

    noise_idx = np.where((freqs > 4) & (freqs < 8))[0] # between 4 and 8 (target: 6Hz)
    # Note - decalage de frequences... to be optimized as well
    # TODO : optimize + visualize

    signal_power = np.sum(power_spectrum[signal_idx]) # Sum of the power within the signal indices
    noise_power = np.mean(power_spectrum[noise_idx]) # Average of the power across the noise indices

    # Calculate SNR
    SNR = 10 * np.log10(signal_power / noise_power)

    # Save SNR
    snr_results.append({"Filename": filename, "SNR": SNR})

    # # Save the plot to the output folder
    # plt.figure(figsize=(10, 6))
    # plt.plot(freqs, power_spectrum)
    # plt.xlim([0, 50])
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power Spectrum')
    # plt.title(f'Power Spectrum for {filename}')
    # plt.savefig(os.path.join(output_folder, f'{filename}_power_spectrum.png'))
    # plt.close()

snr_df = pd.DataFrame(snr_results)

# Save the DataFrame to a CSV file
snr_df.to_csv(snr_file, index=False)