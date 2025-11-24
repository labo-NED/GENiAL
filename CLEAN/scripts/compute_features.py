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
from scipy.signal import welch
from fooof import FOOOF, FOOOFGroup, fit_fooof_3d
from fooof.objs import combine_fooofs
import pickle, csv, time
from neurokit2 import entropy_multiscale
from joblib import Parallel, delayed
import os
import glob
import threading


start_time = time.time()

# ------------ DISK KEEP-ALIVE (prevents macOS from ejecting external drives) ------------
def keep_disk_alive(stop_event, keepalive_path, data_dir, interval=5):
    """
    Aggressively keeps disk active to prevent macOS from ejecting it.
    Args:
        stop_event: threading.Event to signal when to stop
        keepalive_path: path to the keepalive file
        data_dir: directory with data files to periodically list
        interval: seconds between disk accesses (default: 5s)
    """
    while not stop_event.is_set():
        try:
            # Write to keepalive file
            with open(keepalive_path, 'a') as f:
                f.write(f'{time.time()}\n')
                f.flush()  # Force write to disk
                os.fsync(f.fileno())  # Ensure it's written to disk immediately
            
            # Also do a read operation - list directory
            if os.path.exists(data_dir):
                _ = os.listdir(data_dir)
            
            time.sleep(interval)
        except Exception as e:
            print(f'Warning: Keep-alive failed - {e}')
            break

# ------------ User Toggle ------------
EPOCH_2S = True  # Set to True to process 2s epochs, False to skip
EPOCH_5S = True  # Set to True to process 5s epochs, False to skip

# ------------ Paths ------------
# Detect if running locally (if root_dir points to external volume)
IS_LOCAL = True  # Set to True for local runs, False for cluster runs

if IS_LOCAL:
    # Locally - no parallelization
    root_dir = '/Volumes/NED_Backup3/'
    dir_2s = os.path.join(root_dir, 'COMBINED_Q1K_BC_2s/curated_list_for_genial_2s/')
    dir_5s = os.path.join(root_dir, 'COMBINED_Q1K_BC_5s/curated_list_for_genial_5s/')
    output_dir = '/Volumes/NED_Backup3/Q1K_BC_EEG_features/Q1K_BC_EEG_features'
    os.makedirs(output_dir, exist_ok=True)
else:
    # Cluster
    root_dir = '/home/emmacona/projects/def-lippes/emmacona'
    dir_2s = os.path.join(root_dir, 'COMBINED_BC_Q1K_PREPROCESSED_RS_EEG_2s')
    dir_5s = os.path.join(root_dir, 'COMBINED_BC_Q1K_PREPROCESSED_RS_EEG_5s')
    output_dir = '/scratch/emmacona/Q1K_BC_EEG_features'
    os.makedirs(output_dir, exist_ok=True)

# Scan directories for .set files
listing_2s = glob.glob(os.path.join(dir_2s, '*.set')) if EPOCH_2S else []
listing_5s = glob.glob(os.path.join(dir_5s, '*.set')) if EPOCH_5S else []

# Remove mac hidden files
files_2s = [f for f in listing_2s if not os.path.basename(f).startswith('._')]
files_5s = [f for f in listing_5s if not os.path.basename(f).startswith('._')]

# If running locally, filter to specific files only
if IS_LOCAL:
    SPECIFIC_FILES = [
        'Q1K_HSJ_1525-1162_P_RSRio_20250130_125409_processed_2s.set',
        'Q1K_HSJ_1525-1093_S1_RS_20241213_104138_processed_2s.set',
        'Q1K_HSJ_100131_P_RSRio_20240416_114218_processed_2s.set',
        'Q1K_HSJ_1525-1121_P_RS_20241121_011407_processed_2s.set',
        'Q1K_HSJ_1525-1042_P_RS_20240715_103721_processed_2s.set',
        'Q1K_HSJ_100114_P_RS_20240307_114603_processed_2s.set',
        'BC_2017_83073_889938_P_processed_2s.set',
        'BC_2017_83046_889929_S2_processed_2s.set',
        'BC_2017_83059_889934_P_processed_2s.set',
        'Q1K_HSJ_1525-1212_P_RS_20250306_121010_processed_2s.set',
        'Q1K_HSJ_100150_P_RSRio_20240507_030310_processed_2s.set',
        'Q1K_HSJ_1525-1222_S1_RSRio_20250502_094809_processed_2s.set',
        'Q1K_HSJ_10064_P_RS_20240620_104742_processed_2s.set',
        'Q1K_HSJ_1525-1187_P_RSRio_20250317_015350_processed_2s.set',
        'BC_2017_82908_889890_P_processed_2s.set',
        'Q1K_HSJ_100100_P_RSRIO_20240223_121015_processed_2s.set',
        'BC_2017_82864_889818_P_processed_2s.set',
        'Q1K_HSJ_100131_S1_RSRio_20240416_101805_processed_2s.set',
        'Q1K_HSJ_10064_P_RSRio_20240620_104223_processed_2s.set',
        'Q1K_HSJ_1525-1159_P_RS_20250214_124839_processed_2s.set',
        'Q1K_HSJ_1525-1187_P_RS_20250317_015813_processed_2s.set',
        'Q1K_HSJ_100111_P_RS_20240412_105623_processed_2s.set',
        'Q1K_HSJ_1525-1093_P_RS_20241213_093552_processed_2s.set',
        'BC_2017_82942_889899_S1_processed_2s.set',
        'Q1K_HSJ_1525-1147_P_RSRio_20250122_011522_processed_2s.set',
        'Q1K_HSJ_1525-1106_P_RS_20241122_103934_processed_2s.set',
        'Q1K_HSJ_1525-1212_P_RSRio_20250306_120428_processed_2s.set',
        'Q1K_HSJ_1525-1143_P_RSRio_20250121_110055_processed_2s.set',
        'Q1K_HSJ_100157_P_RSRio_20240521_122626_processed_2s.set',
        'Q1K_HSJ_1525-1112_P_RSRio_20241209_010626_processed_2s.set',
        'BC_2017_82934_889897_P_processed_2s.set',
        'BC_2017_82968_889906_S1_processed_2s.set',
        'Q1K_HSJ_100128_P_RS_20240705_112203_processed_2s.set',
        'Q1K_HSJ_1525-1093_S1_RSRio_20241213_112756_processed_2s.set',
        'Q1K_HSJ_1525-1006_P_RS_20240523_114556_processed_2s.set',
        'Q1K_HSJ_1525-1143_P_RS_20250121_110548_processed_2s.set',
        'Q1K_HSJ_1525-1102_P_RSRio_20241008_095126_processed_2s.set',
        'Q1K_HSJ_1525-1203_S2_RS_20250311_093628_processed_2s.set',
        'BC_2017_83023_889924_P_processed_2s.set',
        'Q1K_HSJ_1525-1078_P_RS_20241202_122524_processed_2s.set',
        'Q1K_HSJ_1525-1102_P_RS_20241008_095726_processed_2s.set',
        'Q1K_HSJ_1525-1093_S1_RS_20241213_113519_processed_2s.set',
        'Q1K_HSJ_1525-1130_S1_RS_20250217_101835_processed_2s.set',
        'Q1K_HSJ_1525-1083_S1_RS_20241004_104929_processed_2s.set',
        'Q1K_HSJ_1525-1192_S1_RS_20250307_120849_processed_2s.set',
        'BC_2017_83145_889961_S1_processed_2s.set',
        'Q1K_HSJ_1525-1203_P_RSRio_20250311_011928_processed_2s.set',
        'Q1K_HSJ_1525-1143_S1_RS_20250121_121738_processed_2s.set',
        'Q1K_HSJ_10083_P_RS_20240306_111140_processed_2s.set',
        'Q1K_HSJ_100152_P_RS_20240510_010111_processed_2s.set',
        'Q1K_HSJ_1525-1109_P_RS_20241204_124809_processed_2s.set',
        'Q1K_HSJ_10050_P_RS_20240527_095703_processed_2s.set',
        'Q1K_HSJ_1525-1147_S1_RSRio_20250122_105734_processed_2s.set',
        'Q1K_HSJ_1525-1102_S1_RS_20241008_025044_processed_2s.set',
        'Q1K_HSJ_1525-1203_S3_RS_20250311_104202_processed_2s.set',
        'Q1K_HSJ_1525-1028_P_RS_20240704_124219_processed_2s.set',
        'Q1K_HSJ_1525-1057_P_RS_20240722_120201_processed_2s.set',
        'Q1K_HSJ_1525-1028_S1_RS_20240704_092930_processed_2s.set',
        'Q1K_HSJ_1525-1106_P_RSRio_20241122_103458_processed_2s.set',
        'Q1K_HSJ_1525-1134_S1_RSRio_20241219_102224_processed_2s.set',
        'Q1K_HSJ_10083_P_RSRIO_20240306_110659_processed_2s.set',
        'Q1K_HSJ_1525-1083_P_RS_20241004_125113_processed_2s.set',
        'Q1K_HSJ_100100_P_RS_20240223_121442_processed_2s.set',
        'Q1K_HSJ_100162_P_RS_20240530_114409_processed_2s.set',
        'BC_2017_82501_889508_S2_processed_2s.set',
        'Q1K_HSJ_100162_P_RSRio_20240530_113840_processed_2s.set',
        'Q1K_HSJ_1525-1200_P_RSRio_20250226_105957_processed_2s.set',
        'BC_2017_82892_889885_S1_processed_2s.set',
        'Q1K_HSJ_1525-1143_S1_RSRio_20250121_121220_processed_2s.set',
        'Q1K_HSJ_1525-1169_P_RS_20250303_110640_processed_2s.set',
        'Q1K_HSJ_1525-1209_P_RS_20250319_031314_processed_2s.set',
        'BC_2017_83038_889927_S2_processed_2s.set',
        'Q1K_HSJ_1525-1001_P_RS_20240524_015707_processed_2s.set',
        'BC_2017_83210_889987_S2_processed_2s.set',
        'Q1K_HSJ_100114_S2_RSRIO_20240307_012037_processed_2s.set',
        'BC_2017_83123_889955_S1_processed_2s.set',
        'BC_2017_83045_889929_S1_processed_2s.set',
        'Q1K_HSJ_1525-1024_P_RS_20240625_125100_processed_2s.set',
        'BC_2017_82893_889885_S2_processed_2s.set',
        'BC_2017_83146_889962_P_processed_2s.set',
        'Q1K_HSJ_1525-1121_P_RSRio_20241121_010850_processed_2s.set',
        'Q1K_HSJ_1525-1147_P_RS_20250122_012131_processed_2s.set',
        'Q1K_HSJ_1525-1195_S2_RSRio_20250321_111640_processed_2s.set',
        'Q1K_HSJ_100128_P_RSRio_20240705_111748_processed_2s.set',
        'BC_2017_82904_889889_P_processed_2s.set',
        'Q1K_HSJ_100131_P_RS_20240416_115140_processed_2s.set',
        'Q1K_HSJ_1525-1182_P_RS_20250228_104006_processed_2s.set',
        'Q1K_HSJ_1525_1009_P_RSRio_20240806_033726_processed_2s.set',
        'Q1K_HSJ_1525-1089_S1_RS_20240920_111923_processed_2s.set',
        'Q1K_HSJ_100111_P_RSRIO_20240412_105049_processed_2s.set',
        'BC_2017_83119_889954_S_processed_2s.set',
        'BC_2017_83026_889925_P_processed_2s.set',
        'Q1K_HSJ_1525-1114_P_RS_20241120_093856_processed_2s.set',
        'BC_2017_83142_889961_P_processed_2s.set',
        'Q1K_HSJ_1525-1089_P_RS_20240920_124841_processed_2s.set',
        'Q1K_HSJ_100129_P_RS_20240426_104656_processed_2s.set',
        'BC_2017_82889_889885_P_processed_2s.set',
        'Q1K_HSJ_100152_P_RSRio_20240510_125642_processed_2s.set',
        'Q1K_HSJ_1525-1121_S1_RS_20241121_110437_processed_2s.set',
        'Q1K_HSJ_10050_P_RSRio_20240527_095216_processed_2s.set',
        'Q1K_HSJ_1525-1200_P_RS_20250226_110431_processed_2s.set',
        'Q1K_HSJ_1525-1130_P_RS_20250217_011322_processed_2s.set',
        'Q1K_HSJ_1525-1102_S1_RSRio_20241008_024634_processed_2s.set',
        'Q1K_HSJ_100126_P_RS_20240314_025247_processed_2s.set',
        'Q1K_HSJ_100129_P_RSRio_20240426_104327_processed_2s.set',
        'Q1K_HSJ_1525-1222_S1_RS_20250502_095310_processed_2s.set',
        'Q1K_HSJ_1525-1093_S1_RSRio_20241213_103649_processed_2s.set',
        'BC_2017_83090_889944_P_processed_2s.set',
        'Q1K_HSJ_1525-1040_P_RS_20241021_121648_processed_2s.set',
        'Q1K_HSJ_1525-1159_S2_RS_20250214_113249_processed_2s.set',
        'BC_2017_83033_889926_S1_processed_2s.set',
        'Q1K_HSJ_1525-1209_P_RSRio_20250319_030852_processed_2s.set',
        'Q1K_HSJ_1525-1114_P_RSRio_20241120_093100_processed_2s.set',
        'Q1K_HSJ_1525-1195_S1_RSRio_20250321_024722_processed_2s.set',
        'Q1K_HSJ_1525-1192_P_RSRio_20250307_011113_processed_2s.set',
        'BC_2017_82911_889890_S1_processed_2s.set',
        'Q1K_HSJ_100150_P_RS_20240507_030724_processed_2s.set',
        'Q1K_HSJ_100147_P_RSRio_20240501_104743_processed_2s.set',
        'Q1K_HSJ_1525-1102_S2_RSRio_20241008_014010_processed_2s.set',
        'Q1K_HSJ_1525-1028_S2_RS_20240704_111819_processed_2s.set',
        'Q1K_HSJ_1525-1159_S1_RS_20250214_020004_processed_2s.set',
        'BC_2017_83102_889948_P_processed_2s.set',
        'Q1K_HSJ_1525-1118_P_RSRio_20241129_113029_processed_2s.set',
        'Q1K_HSJ_1525-1147_S1_RS_20250122_110311_processed_2s.set',
        'Q1K_HSJ_100126_P_RSRIO_20240314_024758_processed_2s.set',
        'Q1K_HSJ_1525-1109_P_RSRio_20241204_124416_processed_2s.set',
        'Q1K_HSJ_1525-1102_S2_RS_20241008_014527_processed_2s.set',
        'Q1K_HSJ_10064_S1_RS_20240620_114943_processed_2s.set',
        'Q1K_HSJ_1525-1033_P_RS_20240708_124358_processed_2s.set',
        'Q1K_HSJ_1525-1203_S2_RSRio_20250311_093156_processed_2s.set',
        'Q1K_HSJ_1525-1130_P_RSRio_20250217_010858_processed_2s.set',
        'BC_2017_83037_889927_S1_processed_2s.set',
        'BC_2017_83004_889918_P_processed_2s.set',
        'BC_2017_82912_889890_S2_processed_2s.set',
        'Q1K_HSJ_1525-1112_P_RS_20241209_011021_processed_2s.set',
        'Q1K_HSJ_1525-1093_P_RSRio_20241213_093147_processed_2s.set',
        'BC_2017_82894_889886_P_processed_2s.set',
        'Q1K_HSJ_100114_S2_RS_20240307_012445_processed_2s.set',
        'Q1K_HSJ_1525-1256_P_RSRio_20250501_010720_processed_2s.set',
        'Q1K_HSJ_1525-1080_P_RS_20240823_020457_processed_2s.set',
        'Q1K_HSJ_1525-1021_P_RS_20240827_120957_processed_2s.set',
        'Q1K_HSJ_100131_P_RS_20240416_114544_processed_2s.set',
        'Q1K_HSJ_1525-1159_P_RSRio_20250214_124304_processed_2s.set',
        'BC_2017_82931_889896_P_processed_2s.set',
        'Q1K_HSJ_1525-1195_S1_RS_20250321_025129_processed_2s.set',
        'Q1K_HSJ_1525-1256_P_RS_20250501_011133_processed_2s.set',
        'Q1K_HSJ_1525-1182_P_RSRio_20250228_103607_processed_2s.set',
        'Q1K_HSJ_100114_P_RSRIO_20240307_114051_processed_2s.set',
        'Q1K_HSJ_1525-1169_P_RSRio_20250303_110216_processed_2s.set',
        'BC_2017_83116_889954_P_processed_2s.set',
        'Q1K_HSJ_100162_S1_RS_20240530_101720_processed_2s.set',
        'Q1K_HSJ_1525-1045_S1_RS_20240805_110059_processed_2s.set',
        'Q1K_HSJ_1525-1159_S2_RSRio_20250214_112636_processed_2s.set',
        'Q1K_HSJ_100157_P_RS_20240521_123152_processed_2s.set',
        'Q1K_HSJ_100131_S1_RS_20240416_102202_processed_2s.set',
        'Q1K_HSJ_1525-1203_S3_RSRio_20250311_103812_processed_2s.set',
        'Q1K_HSJ_1525-1134_S1_RS_20241219_102733_processed_2s.set',
        'Q1K_HSJ_10043_P_RS_20240328_114417_processed_2s.set',
        'Q1K_HSJ_1525-1195_P_RS_20250321_015232_processed_2s.set',
        'Q1K_HSJ_1525-1203_P_RS_20250311_012324_processed_2s.set',
        'Q1K_HSJ_100147_P_RS_20240501_105202_processed_2s.set',
        'BC_2017_83035_889927_P_processed_2s.set',
        'Q1K_HSJ_10064_S1_RSRio_20240620_114541_processed_2s.set',
        'Q1K_HSJ_10053_P_RSRIO_20240419_101544_processed_2s.set',
        'BC_2017_82955_889903_S1_processed_2s.set',
        'Q1K_HSJ_1525-1134_P_RSRio_20241219_113118_processed_2s.set',
        'Q1K_HSJ_1525-1195_S2_RS_20250321_112115_processed_2s.set',
        'Q1K_HSJ_1525-1045_P_RS_20240805_120946_processed_2s.set',
        'Q1K_HSJ_1525-1118_P_RS_20241129_113417_processed_2s.set',
        'Q1K_HSJ_1525-1130_S1_RSRio_20250217_101328_processed_2s.set',
        'Q1K_HSJ_1525-1089_S1_RSRio_20240920_111356_processed_2s.set',
        'Q1K_HSJ_1525-1026_P_RS_20240613_023013_processed_2s.set',
        'Q1K_HSJ_10043_P_RSRio_20240328_113119_processed_2s.set',
        'Q1K_HSJ_1525-1067_P_RS_20240816_120910_processed_2s.set',
        'Q1K_HSJ_100162_S1_RSRio_20240530_101050_processed_2s.set',
    ]
    
    # Filter files to only those in SPECIFIC_FILES
    files_2s = [f for f in files_2s if os.path.basename(f) in SPECIFIC_FILES]
    files_5s = [f for f in files_5s if os.path.basename(f) in SPECIFIC_FILES]
    
    print(f'\n*** RUNNING IN LOCAL MODE - Processing only {len(SPECIFIC_FILES)} specific files ***\n')

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
excluded_chans = ['48','119','43','49','56','63','68','73','81','88','94','99','107','113','120','125','126','127','128','17','129']  # w/ ref
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

# ---------- START DISK KEEP-ALIVE FOR LOCAL RUNS ----------
keepalive_thread = None
stop_keepalive = None
if IS_LOCAL:
    keepalive_path = os.path.join(output_dir, '.keepalive')
    stop_keepalive = threading.Event()
    keepalive_thread = threading.Thread(
        target=keep_disk_alive, 
        args=(stop_keepalive, keepalive_path, dir_2s, 5),  # Check every 5 seconds
        daemon=True
    )
    keepalive_thread.start()
    print('*** Disk keep-alive thread started (every 5 seconds - prevents auto-eject) ***\n')

# ---------- MAIN FUNCTION ----------
try:
    for file_idx, (filepath, epoch_type) in enumerate(all_files):
        filename = os.path.basename(filepath)

        # ----- Skip files that are already processed -----
        output_base = filename.replace('.set', '').replace('_processed', '')
        out_pkl = os.path.join(output_dir, f'features_{output_base}.pkl')
        out_csv = os.path.join(output_dir, f'features_avg_{output_base}.csv')

        if os.path.exists(out_pkl) and os.path.exists(out_csv):
            print(f'\nSkipping file {file_idx+1}/{nFiles}: {filename} ({epoch_type}) - outputs already exist.\n')
            continue
        # ------------------------------------------------------
        
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

finally:
    # Stop the keep-alive thread
    if IS_LOCAL and stop_keepalive is not None:
        stop_keepalive.set()
        if keepalive_thread is not None:
            keepalive_thread.join(timeout=2)
        print('\n*** Disk keep-alive thread stopped ***\n')
        # Clean up keepalive file
        keepalive_path = os.path.join(output_dir, '.keepalive')
        if os.path.exists(keepalive_path):
            os.remove(keepalive_path)

total_time = time.time() - start_time
print(f'\n{"="*60}')
print(f'Completed processing {nFiles} files')
print(f'Total time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)')
if nFiles > 0:
    print(f'Average time per file: {total_time/nFiles:.1f} seconds')
print(f'Output directory: {output_dir}')
print(f'{"="*60}\n')

