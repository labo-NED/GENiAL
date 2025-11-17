### COMPUTE FEATURES IN PYTHON FOR CINEMA PROJECT, WITH MSE IN PARALLEL
# Beware: Remains to be fully tested
# Laurent Caplette (2025) based on code by Saeideh Davoodi
# Emmanuelle Coutu-Nadeau (Nov2025) based on code by Laurent Caplette

# ---------- IMPORT PACKAGES ----------
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


start_time = time.time()

proj_dir = '/project/def-lippes/cinema'
Fs = 1000
n_chans = 108 ### excluding pre-excluded channels and reference
n_cores = 64 ### DEPENDS ON BATCH SCRIPT
maxscale = 40 # maximum scale for MSE
excluded_chans = ['48','119','43','49','56','63','68','73','81','88','94','99','107','113','120','125','126','127','128','17','129'] # w/ ref
included_chans = [str(idx) for idx in range(1,130) if str(idx) not in excluded_chans]

suj_list = [73] ### list of subject IDs to process ##### TO EDIT

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
for suj_idx, suj in enumerate(suj_list):

    print(f'\nExtracting features for subject {suj}\n')
    file_prefix = f'CON_ATT_{suj:03}_IC_EEG_EYE'

    features = {}
    mse_array = None
    features['channel_list'] = included_chans

    epochs_short = mne.read_epochs(proj_dir+'/preprocessed/'+file_prefix+'-2s-epo.fif')
    epochs_short.info['bads'] = excluded_chans # will be removed when data is loaded ###
    epochs_long = mne.read_epochs(proj_dir+'/preprocessed/'+file_prefix+'-5s-MSE_epo.fif')
    epochs_long.info['bads'] = excluded_chans # will be removed when data is loaded ###

    # extract features from 2s epochs
    print(np.shape(epochs_short.get_data(picks='all')))
    epoch_data = epochs_short.get_data(picks='all')[:,:n_chans] # excluding bad channels
    n_epochs = epoch_data.shape[0]
    features_short = ['hurst', 'pow_delta', 'pow_theta', 'pow_alpha','pow_beta', 'pow_gamma', 'pow_low_gamma', 'pow_high_gamma',\
                     'fooof_offset', 'fooof_exp', 'pow_per_delta', 'pow_per_theta', 'pow_per_alpha', 'pow_per_beta', 'pow_per_gamma',\
                     'pow_per_low_gamma', 'pow_per_high_gamma']

    if n_epochs>0:
        # get some features using mne-features ### to parallelize?
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

        # get other features using FOOOF
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
            per_spec_avg[ch] = fg.get_fooof(ch).get_data(component='peak', space='linear')
        
        # Store FOOOF parameters (replicated across epochs to maintain structure)
        for f, feature in enumerate(features_short[8:10]):
            features[feature] = np.tile(aper_params_avg[:, f], (n_epochs, 1))  # (n_epochs, n_chans)
        
        # Compute periodic power in frequency bands
        for feature, band_freqs in zip(features_short[10:], FREQ_BANDS.values()):
            band_mask = np.logical_and(freqs >= band_freqs[0], freqs < band_freqs[1])
            per_power_band = np.mean(per_spec_avg[:, band_mask], axis=1)  # (n_chans,)
            features[feature] = np.tile(per_power_band, (n_epochs, 1))  # (n_epochs, n_chans)
        
        comp_time = time.time() - previous_time
        print(f'Computation time: {comp_time:.3f}s\n')

    else:
        print('There are no 2s epochs. Not computing 2s features.')

    # extract epochs from 5s epochs
    epoch_data = epochs_long.get_data(picks='all')[:,:n_chans] # excluding bad channels
    n_epochs = epoch_data.shape[0]    
    selected_funcs = ['higuchi_fd', 'katz_fd', 'samp_entropy']
    features_long = ['higuchi_fd', 'katz_fd', 'samp_entropy', 'CI', 'CI_lowscale', 'CI_highscale']
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
        features['CI'] = np.trapezoid(mse_array,scales,axis=-1)
        features['CI_lowscale'] = np.trapezoid(mse_array[:,:,:maxscale//2],scales[:maxscale//2],axis=-1)
        features['CI_highscale'] = np.trapezoid(mse_array[:,:,maxscale//2:],scales[maxscale//2:],axis=-1)
        comp_time = time.time() - previous_time
        print(f'Computation time: {comp_time:.3f}s\n')

    else:
        print('There are no 5s epochs. Not computing 5s features.')

    # save complete MSE values in separate file ###
    if mse_array is not None:
        np.save(proj_dir+f'/features/mse_suj{suj:03}.npy', mse_array)

    # save feature dictionary in pkl file (all segments)
    with open(proj_dir+f'/features/features_suj{suj:03}.pkl', 'wb') as file:
        pickle.dump(features, file)

    # average across segments, convert and save to CSV
    feature_names = [key for key in features.keys() if key != 'channel_list'] # get feature names
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
    np.savetxt(proj_dir+f'/features/features_avg_suj{suj:03}.csv', data_avg.T, delimiter=',', header=header, comments='')

total_time = time.time() - start_time
print(f'\nTotal time: {total_time:.3f}s\n')

