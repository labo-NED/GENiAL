import mne

def read_eeg(eeg_file):
    raw = mne.io.read_raw_eeglab(eeg_file)
    return raw

data = read_eeg('/Volumes/NED_Backup3/Q1K_Preprocessed_s_Happe/5 - processed/Q1K_HSJ_1525-1212_M1_RS_20250306_105556_processed.set')

print(data)