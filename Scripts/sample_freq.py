# Date created: 12/18/2024
# Author: Emmanuelle CN
# Description: Extraction of sample frequency information from raw EEG (.set) file

#### Imports
import mne

# Load the EEG epochs data
epochs = mne.io.read_epochs_eeglab('filepath.set')

# Get the sampling frequency
sampling_freq = epochs.info['sfreq']

print("Sampling frequency:", sampling_freq, "Hz")