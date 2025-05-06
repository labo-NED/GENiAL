# Date created: 12/18/2024
# Author: Emmanuelle CN

import mne
import numpy as np
import os
import tempfile
import shutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Plot spectral density
bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 100)
}

# Load  EEG data
folder_path = 'Data/6Hz'
PDF_file_path = 'Output/6Hz/EEG_PSD_plot.pdf'

def save_figure_to_pdf(fig, pdf):
    temp_dir = tempfile.mkdtemp()
    try:
        temp_img_path = os.path.join(temp_dir, 'temp_image.png')
        fig.savefig(temp_img_path)
        img = plt.imread(temp_img_path)
        fig_img = plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
        plt.imshow(img)
        plt.axis('off')
        pdf.savefig(fig_img)
        plt.close(fig_img)
    finally:
        shutil.rmtree(temp_dir)

with PdfPages(PDF_file_path) as pdf:
    for file in os.listdir(folder_path):
        if file.endswith('.set'):
            file_path = os.path.join(folder_path, file)
            epochs = mne.io.read_epochs_eeglab(file_path)

            # # Check for any existing bad channel annotations in the data
            # # TODO need to identify bad channels to simplify future analysis
            # print("Bad channels:", epochs.info['bads'])

            # # Plot the epochs data
            # epochs.plot(n_epochs=10, 
            #             n_channels=len(epochs.ch_names),  # Change the number of epochs to view different parts of the EEG file
            #             scalings='auto')

            epochs.plot_psd(tmax=np.inf, fmax=100, average=False, show=False)
            fig = plt.gcf()
            fig.suptitle(f'PSD for {file}')
            save_figure_to_pdf(fig, pdf)
            plt.close(fig)

print("All PSD plots have been saved into the PDF.")