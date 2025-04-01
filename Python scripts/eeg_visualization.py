import numpy as np
import mne


raw = mne.io.read_raw_edf('Data/RS/sub-100100F1_ses-01_task-RS_run-1_eeg.edf')
raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)
raw.plot(duration=5, n_channels=128)