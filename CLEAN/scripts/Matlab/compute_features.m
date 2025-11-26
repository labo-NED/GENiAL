% COMPUTE EEG FEATURES FOR RESTING-STATE DATA (GENiAL PROJECT)
% This script extracts spectral and complexity features from epoched EEG data
% 
% Features extracted:
%   From 2s epochs: Hurst exponent, band powers, FOOOF parameters, periodic powers
%   From 5s epochs: Fractal dimensions, sample entropy, MSE, complexity index
%
% Requirements:
%   - EEGLAB toolbox
%   - fooof_mat (FOOOF MATLAB wrapper)
%   - estimate_hurst_exponent.m (custom function)
%   - get_mse.m (optional, custom MSE implementation used if available)
%
% Laurent Caplette (2025) based on code by Saeideh Davoodi
% Emmanuelle Coutu-Nadeau (Nov 2025) - Adapted for GENiAL resting-state analysis
% Converted to MATLAB (Nov 2025)

clear; clc;

% Add required toolboxes to the path
addpath('/Users/emmanuelle.coutu-nadeau/Code/NED LAB/tools/eeglab2022.0');
addpath(genpath('/Users/emmanuelle.coutu-nadeau/Code/NED LAB/tools/fooof_mat'));

% Add custom functions (if they exist as separate files)
tools_dir = '/Users/emmanuelle.coutu-nadeau/Code/NED LAB/tools';
if exist(fullfile(tools_dir, 'estimate_hurst_exponent.m'), 'file')
    addpath(tools_dir);
end
if exist(fullfile(tools_dir, 'get_mse.m'), 'file')
    addpath(tools_dir);
end

% Initialize EEGLAB (suppress GUI)
fprintf('Initializing EEGLAB...\n');
eeglab nogui;
close all;  % Close any windows that might have opened

fprintf('Starting feature extraction...\n');
tic; % Start timing

% ------------ User Toggle ------------
EPOCH_2S = true;  % Set to true to process 2s epochs, false to skip
EPOCH_5S = true;  % Set to true to process 5s epochs, false to skip

% ------------ Paths ------------
% root_dir = '/home/emmacona/projects/def-lippes/emmacona';
% dir_2s = fullfile(root_dir, 'COMBINED_BC_Q1K_PREPROCESSED_RS_EEG_2s');
% dir_5s = fullfile(root_dir, 'COMBINED_BC_Q1K_PREPROCESSED_RS_EEG_5s');
root_dir = '/Volumes/NED_Backup3/';
dir_2s = fullfile(root_dir, 'COMBINED_Q1K_BC_2s/curated_list_for_genial_2s/');
dir_5s = fullfile(root_dir, 'COMBINED_Q1K_BC_5s/curated_list_for_genial_5s/');

% Save outputs in /scratch folder
% output_dir = '/scratch/emmacona/Q1K_BC_EEG_features';
output_dir = '/Volumes/NED_Backup3/Features';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Scan directories for .set files
files_2s = {};
files_5s = {};
if EPOCH_2S
    files_2s = dir(fullfile(dir_2s, '*.set'));
    files_2s = {files_2s.name}';
    % Remove hidden files
    files_2s = files_2s(~startsWith(files_2s, '._'));
end
if EPOCH_5S
    files_5s = dir(fullfile(dir_5s, '*.set'));
    files_5s = {files_5s.name}';
    % Remove hidden files
    files_5s = files_5s(~startsWith(files_5s, '._'));
end

% Combine all files to process
all_files = {};
all_types = {};
if EPOCH_2S
    for i = 1:length(files_2s)
        all_files{end+1} = fullfile(dir_2s, files_2s{i});
        all_types{end+1} = '2s';
    end
end
if EPOCH_5S
    for i = 1:length(files_5s)
        all_files{end+1} = fullfile(dir_5s, files_5s{i});
        all_types{end+1} = '5s';
    end
end

nFiles = length(all_files);

if nFiles == 0
    error('No .set files found in: %s or %s', dir_2s, dir_5s);
end

fprintf('\nFound %d files to process\n', nFiles);
if EPOCH_2S
    fprintf('  - %d files in 2s directory\n', length(files_2s));
end
if EPOCH_5S
    fprintf('  - %d files in 5s directory\n', length(files_5s));
end
fprintf('\n');

% ------------ Common Params ------------
Fs = 1000;
n_chans = 108;
maxscale = 40;  % maximum scale for MSE

excluded_chans = [48, 119, 43, 49, 56, 63, 68, 73, 81, 88, 94, 99, 107, 113, 120, 125, 126, 127, 128, 17, 129];
all_chan_nums = 1:129;
included_chans = all_chan_nums(~ismember(all_chan_nums, excluded_chans));

% ---------- CONSTANTS ----------
freq_bands = struct();
freq_bands.delta = [1, 4];
freq_bands.theta = [4, 8];
freq_bands.alpha = [8, 13];
freq_bands.beta = [13, 30];
freq_bands.gamma = [30, 80];
freq_bands.low_gamma = [30, 59];
freq_bands.high_gamma = [61, 80];

band_names = {'delta', 'theta', 'alpha', 'beta', 'gamma', 'low_gamma', 'high_gamma'};

% Verify EEGLAB is properly initialized
if ~exist('pop_loadset', 'file')
    error('EEGLAB not properly initialized. pop_loadset function not found.');
end

% Verify required functions
if ~exist('estimate_hurst_exponent', 'file')
    error('estimate_hurst_exponent function not found. Please add it to your MATLAB path.');
end
if ~exist('fooof', 'file')
    error('FOOOF function not found. Please add fooof_mat to your MATLAB path.');
end

fprintf('All required functions found.\n\n');

% Setup parallel pool
try
    if isempty(gcp('nocreate'))
        n_cores = str2double(getenv('SLURM_CPUS_PER_TASK'));
        if isnan(n_cores)
            n_cores = 8;
        end
        parpool(n_cores);
    end
catch
    warning('Could not create parallel pool. Running serially.');
end

% ---------- MAIN LOOP ----------
for file_idx = 1:nFiles
    filepath = all_files{file_idx};
    epoch_type = all_types{file_idx};
    [~, filename, ext] = fileparts(filepath);
    filename_full = [filename, ext];
    
    % ----- Skip files that are already processed -----
    output_base = strrep(strrep(filename, '.set', ''), '_processed', '');
    out_mat = fullfile(output_dir, sprintf('features_%s.mat', output_base));
    out_csv = fullfile(output_dir, sprintf('features_avg_%s.csv', output_base));
    
    if exist(out_mat, 'file') && exist(out_csv, 'file')
        fprintf('\nSkipping file %d/%d: %s (%s) - outputs already exist.\n\n', ...
            file_idx, nFiles, filename_full, epoch_type);
        continue;
    end
    
    fprintf('\nProcessing file %d/%d: %s (%s)\n\n', ...
        file_idx, nFiles, filename_full, epoch_type);
    
    % Initialize features structure
    features = struct();
    features.filename = filename_full;
    features.epoch_type = epoch_type;
    mse_array = [];
    
    % ========== EXTRACT FEATURES FROM 2S EPOCHS ==========
    if strcmp(epoch_type, '2s')
        fprintf('Loading 2s epochs...\n');
        try
            EEG = pop_loadset('filename', filename_full, 'filepath', fileparts(filepath));
        catch ME
            fprintf('Error reading 2s file %s: %s\n', filename_full, ME.message);
            continue;
        end
        
        % Determine which channels to use (only those that exist in this file)
        actual_n_chans = EEG.nbchan;
        file_included_chans = included_chans(included_chans <= actual_n_chans);
        n_chans_file = length(file_included_chans);
        
        fprintf('File has %d channels, using %d channels (excluding bad channels)\n', ...
            actual_n_chans, n_chans_file);
        
        % Get epoch data (epochs x channels x timepoints)
        epoch_data = EEG.data(file_included_chans, :, :);  % Select only included channels
        epoch_data = permute(epoch_data, [3, 1, 2]);  % (epochs x channels x timepoints)
        n_epochs = size(epoch_data, 1);
        n_samples = size(epoch_data, 3);
        
        fprintf('Data shape: %d epochs x %d channels x %d samples\n', ...
            n_epochs, n_chans_file, n_samples);
        
        % Store channel list for this file
        features.channel_list = file_included_chans;
        
        features_short = {'hurst', 'pow_delta', 'pow_theta', 'pow_alpha', 'pow_beta', ...
            'pow_gamma', 'pow_low_gamma', 'pow_high_gamma', ...
            'fooof_offset', 'fooof_exp', 'pow_per_delta', 'pow_per_theta', ...
            'pow_per_alpha', 'pow_per_beta', 'pow_per_gamma', ...
            'pow_per_low_gamma', 'pow_per_high_gamma'};
        
        if n_epochs > 0
            % ===== Compute Hurst exponent =====
            fprintf('\nExtracting Hurst exponents...\n');
            previous_time = tic;
            hurst_vals = zeros(n_epochs, n_chans_file);
            warning_count = 0;
            
            % Temporarily turn off polyfit warning
            warning_state = warning('query', 'MATLAB:polyfit:RepeatedPointsOrRescale');
            warning('off', 'MATLAB:polyfit:RepeatedPointsOrRescale');
            
            for ep = 1:n_epochs
                for ch = 1:n_chans_file
                    signal = squeeze(epoch_data(ep, ch, :));
                    try
                        hurst_vals(ep, ch) = estimate_hurst_exponent(signal);
                    catch ME
                        % If computation fails, set to NaN
                        warning_count = warning_count + 1;
                        hurst_vals(ep, ch) = NaN;
                    end
                end
            end
            
            % Restore warning state
            warning(warning_state.state, 'MATLAB:polyfit:RepeatedPointsOrRescale');
            
            if warning_count > 0
                fprintf('  Warning: Hurst computation failed for %d epoch-channel pairs (set to NaN)\n', warning_count);
            end
            
            features.hurst = hurst_vals;
            fprintf('Computation time: %.3f s\n\n', toc(previous_time));
            
            % ===== Compute band powers =====
            fprintf('\nExtracting band powers...\n');
            previous_time = tic;
            for b = 1:length(band_names)
                band_name = band_names{b};
                freq_range = freq_bands.(band_name);
                pow_vals = zeros(n_epochs, n_chans_file);
                
                for ep = 1:n_epochs
                    for ch = 1:n_chans_file
                        signal = squeeze(epoch_data(ep, ch, :));
                        pow_vals(ep, ch) = compute_band_power(signal, Fs, freq_range);
                    end
                end
                features.(['pow_' band_name]) = pow_vals;
            end
            fprintf('Computation time: %.3f s\n\n', toc(previous_time));
            
            % ===== Compute FOOOF features =====
            fprintf('\nExtracting FOOOF features...\n');
            previous_time = tic;
            
            % FOOOF settings (adjust peak width limits to be above frequency resolution)
            freq_range = [1, 80];
            fooof_settings = struct('peak_width_limits', [1, 18], 'max_n_peaks', 10);
            
            % Compute PSD for all epochs and channels
            window = hamming(round(Fs * 2));  % 2 second window
            noverlap = round(length(window) / 2);
            nfft = round(length(window));
            
            fooof_offset = zeros(n_chans_file, 1);
            fooof_exp = zeros(n_chans_file, 1);
            per_spec_all = cell(n_chans_file, 1);
            fooof_freqs = [];  % Store FOOOF frequencies
            
            for ch = 1:n_chans_file
                % Compute PSD for each epoch and average
                ch_psds = [];
                for ep = 1:n_epochs
                    signal = squeeze(epoch_data(ep, ch, :));
                    [pxx, freqs] = pwelch(signal, window, noverlap, nfft, Fs);
                    ch_psds = [ch_psds; pxx'];
                end
                avg_psd = mean(ch_psds, 1);
                
                % Fit FOOOF model
                try
                    fooof_results = fooof(freqs, avg_psd, freq_range, fooof_settings, true);
                    fooof_offset(ch) = fooof_results.aperiodic_params(1);
                    fooof_exp(ch) = fooof_results.aperiodic_params(2);
                    
                    % Extract periodic component and frequencies
                    per_spec_all{ch} = fooof_results.power_spectrum - fooof_results.ap_fit;
                    
                    % Store the frequency vector from FOOOF (same for all channels)
                    if ch == 1
                        fooof_freqs = fooof_results.freqs;
                    end
                catch ME
                    warning('FOOOF failed for channel %d: %s', ch, ME.message);
                    fooof_offset(ch) = NaN;
                    fooof_exp(ch) = NaN;
                    per_spec_all{ch} = zeros(size(freqs));
                    if ch == 1
                        fooof_freqs = freqs;  % Fallback to pwelch freqs
                    end
                end
            end
            
            % Store FOOOF parameters (replicated across epochs)
            features.fooof_offset = repmat(fooof_offset', n_epochs, 1);
            features.fooof_exp = repmat(fooof_exp', n_epochs, 1);
            
            % Compute periodic power in frequency bands using FOOOF frequencies
            for b = 1:length(band_names)
                band_name = band_names{b};
                freq_range_band = freq_bands.(band_name);
                band_mask = fooof_freqs >= freq_range_band(1) & fooof_freqs < freq_range_band(2);
                
                per_power_band = zeros(n_chans_file, 1);
                for ch = 1:n_chans_file
                    per_spec = per_spec_all{ch};
                    if any(band_mask) && length(per_spec) == length(band_mask)
                        per_power_band(ch) = mean(per_spec(band_mask));
                    else
                        per_power_band(ch) = NaN;
                    end
                end
                features.(['pow_per_' band_name]) = repmat(per_power_band', n_epochs, 1);
            end
            
            fprintf('Computation time: %.3f s\n\n', toc(previous_time));
            
        else
            fprintf('There are no 2s epochs. Not computing 2s features.\n');
        end
        
    % ========== EXTRACT FEATURES FROM 5S EPOCHS ==========
    elseif strcmp(epoch_type, '5s')
        fprintf('Loading 5s epochs...\n');
        try
            EEG = pop_loadset('filename', filename_full, 'filepath', fileparts(filepath));
        catch ME
            fprintf('Error reading 5s file %s: %s\n', filename_full, ME.message);
            continue;
        end
        
        % Determine which channels to use (only those that exist in this file)
        actual_n_chans = EEG.nbchan;
        file_included_chans = included_chans(included_chans <= actual_n_chans);
        n_chans_file = length(file_included_chans);
        
        fprintf('File has %d channels, using %d channels (excluding bad channels)\n', ...
            actual_n_chans, n_chans_file);
        
        % Get epoch data (epochs x channels x timepoints)
        epoch_data = EEG.data(file_included_chans, :, :);
        epoch_data = permute(epoch_data, [3, 1, 2]);  % (epochs x channels x timepoints)
        n_epochs = size(epoch_data, 1);
        
        % Store channel list for this file
        features.channel_list = file_included_chans;
        
        features_long = {'higuchi_fd', 'katz_fd', 'samp_entropy', ...
            'CI', 'CI_lowscale', 'CI_highscale'};
        
        if n_epochs > 0
            % ===== Compute fractal dimensions and sample entropy =====
            fprintf('\nExtracting non-MSE 5s features...\n');
            previous_time = tic;
            
            higuchi_vals = zeros(n_epochs, n_chans_file);
            katz_vals = zeros(n_epochs, n_chans_file);
            samp_ent_vals = zeros(n_epochs, n_chans_file);
            
            for ep = 1:n_epochs
                for ch = 1:n_chans_file
                    signal = squeeze(epoch_data(ep, ch, :));
                    higuchi_vals(ep, ch) = compute_higuchi_fd(signal, 8);
                    katz_vals(ep, ch) = compute_katz_fd(signal);
                    samp_ent_vals(ep, ch) = compute_sample_entropy(signal);
                end
            end
            
            features.higuchi_fd = higuchi_vals;
            features.katz_fd = katz_vals;
            features.samp_entropy = samp_ent_vals;
            
            fprintf('Computation time: %.3f s\n\n', toc(previous_time));
            
            % ===== Compute MSE and CI =====
            fprintf('\nExtracting MSE features...\n');
            previous_time = tic;
            
            m = 2;  % embedding dimension
            r = 0.15;  % tolerance (matching Python's default of 0.15 * std)
            scales = 1:maxscale;
            
            % Check if get_mse function is available
            use_get_mse = exist('get_mse', 'file');
            
            if use_get_mse
                fprintf('Using get_mse function...\n');
                mse_array = zeros(n_epochs, n_chans_file, maxscale);
                for ch = 1:n_chans_file
                    % get_mse expects (channel x samples x epochs)
                    ch_data = squeeze(epoch_data(:, ch, :))';  % (samples x epochs)
                    [mean_mse_ch, ~, ~, ~] = get_mse(ch_data, m, r, scales);
                    mse_array(:, ch, :) = mean_mse_ch';  % Transpose back to (epochs x scales)
                end
            else
                fprintf('Using custom MSE implementation...\n');
                mse_array = zeros(n_epochs, n_chans_file, maxscale);
                % Use parallel processing for MSE computation
                parfor ep = 1:n_epochs
                    for ch = 1:n_chans_file
                        signal = squeeze(epoch_data(ep, ch, :));
                        mse_array(ep, ch, :) = compute_mse(signal, maxscale, m, r);
                    end
                end
            end
            
            % Compute complexity indices using trapezoidal integration
            features.CI = squeeze(trapz(scales, mse_array, 3));
            features.CI_lowscale = squeeze(trapz(scales(1:maxscale/2), ...
                mse_array(:, :, 1:maxscale/2), 3));
            features.CI_highscale = squeeze(trapz(scales(maxscale/2+1:end), ...
                mse_array(:, :, maxscale/2+1:end), 3));
            
            fprintf('Computation time: %.3f s\n\n', toc(previous_time));
            
        else
            fprintf('There are no 5s epochs. Not computing 5s features.\n');
        end
    end
    
    % ========== SAVE OUTPUTS ==========
    output_base = strrep(strrep(filename, '.set', ''), '_processed', '');
    
    % Save complete MSE values in separate file
    if ~isempty(mse_array)
        save(fullfile(output_dir, sprintf('mse_%s.mat', output_base)), 'mse_array');
    end
    
    % Save feature structure in .mat file
    save(fullfile(output_dir, sprintf('features_%s.mat', output_base)), 'features');
    
    % Average across epochs and save to CSV
    feature_fields = fieldnames(features);
    feature_fields = feature_fields(~ismember(feature_fields, ...
        {'channel_list', 'filename', 'epoch_type'}));
    
    if ~isempty(feature_fields) && isfield(features, 'channel_list')
        % Get the channel list for this file
        file_chans = features.channel_list;
        n_chans_csv = length(file_chans);
        
        % Build header
        header = 'channel';
        for i = 1:length(feature_fields)
            header = [header, ',', feature_fields{i}];
        end
        
        % Compute averages
        data_avg = [file_chans', zeros(n_chans_csv, length(feature_fields))];
        for i = 1:length(feature_fields)
            feat_data = features.(feature_fields{i});
            if ~isempty(feat_data)
                data_avg(:, i+1) = mean(feat_data, 1)';
            else
                data_avg(:, i+1) = NaN;
            end
        end
        
        % Write CSV
        fid = fopen(fullfile(output_dir, sprintf('features_avg_%s.csv', output_base)), 'w');
        fprintf(fid, '%s\n', header);
        fclose(fid);
        dlmwrite(fullfile(output_dir, sprintf('features_avg_%s.csv', output_base)), ...
            data_avg, '-append', 'delimiter', ',', 'precision', '%.6f');
    else
        fprintf('Warning: No features computed for %s, skipping CSV output.\n', filename_full);
    end
end

total_time = toc;
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('Completed processing %d files\n', nFiles);
fprintf('Total time: %.2f minutes (%.1f seconds)\n', total_time/60, total_time);
if nFiles > 0
    fprintf('Average time per file: %.1f seconds\n', total_time/nFiles);
end
fprintf('Output directory: %s\n', output_dir);
fprintf('%s\n\n', repmat('=', 1, 60));

% ========== HELPER FUNCTIONS ==========

function pow = compute_band_power(signal, fs, freq_range)
    % Compute power in a frequency band using Welch's method
    window = hamming(round(fs * 2));
    noverlap = round(length(window) / 2);
    nfft = 2^nextpow2(length(window));
    [pxx, f] = pwelch(signal, window, noverlap, nfft, fs);
    
    % Find frequencies in the band
    idx = f >= freq_range(1) & f < freq_range(2);
    pow = mean(pxx(idx));
end

function fd = compute_higuchi_fd(x, kmax)
    % Compute Higuchi fractal dimension
    N = length(x);
    L = zeros(1, kmax);
    
    for k = 1:kmax
        Lk = zeros(1, k);
        for m = 1:k
            Lmk = 0;
            maxI = floor((N-m)/k);
            for i = 1:maxI
                Lmk = Lmk + abs(x(m+i*k) - x(m+(i-1)*k));
            end
            Lmk = Lmk * (N-1) / (maxI * k);
            Lk(m) = Lmk;
        end
        L(k) = mean(Lk);
    end
    
    % Fit line in log-log space
    x_log = log(1:kmax);
    y_log = log(L);
    p = polyfit(x_log, y_log, 1);
    fd = -p(1);
end

function fd = compute_katz_fd(x)
    % Compute Katz fractal dimension
    n = length(x);
    d = sqrt(sum(diff(x).^2 + 1));  % Total length
    a = sqrt((x(end) - x(1))^2 + (n-1)^2);  % Diameter
    if a == 0
        fd = 1;
    else
        fd = log10(n) / (log10(n) + log10(a/d));
    end
end

function se = compute_sample_entropy(x, m, r)
    % Compute sample entropy
    % m: embedding dimension (default 2)
    % r: tolerance (default 0.2 * std(x))
    if nargin < 2, m = 2; end
    if nargin < 3, r = 0.2 * std(x); end
    
    N = length(x);
    phi = zeros(1, 2);
    
    for j = 1:2
        m_temp = m + j - 1;
        patterns = zeros(N - m_temp, m_temp + 1);
        for i = 1:(N - m_temp)
            patterns(i, :) = x(i:(i + m_temp));
        end
        
        count = zeros(N - m_temp, 1);
        for i = 1:(N - m_temp)
            template = patterns(i, 1:m_temp);
            for k = 1:(N - m_temp)
                if k ~= i
                    if max(abs(patterns(k, 1:m_temp) - template)) <= r
                        count(i) = count(i) + 1;
                    end
                end
            end
        end
        phi(j) = sum(count) / (N - m_temp);
    end
    
    if phi(1) == 0 || phi(2) == 0
        se = 0;
    else
        se = -log(phi(2) / phi(1));
    end
end

function mse = compute_mse(x, maxscale, m, r)
    % Compute multiscale entropy
    % x: signal
    % maxscale: maximum scale
    % m: embedding dimension (default 2)
    % r: tolerance (default 0.15 * std(x))
    if nargin < 3, m = 2; end
    if nargin < 4, r = 0.15 * std(x); end
    
    mse = zeros(1, maxscale);
    for scale = 1:maxscale
        % Coarse-grain the time series
        if scale == 1
            y = x;
        else
            n_segments = floor(length(x) / scale);
            y = zeros(1, n_segments);
            for i = 1:n_segments
                y(i) = mean(x((i-1)*scale + 1 : i*scale));
            end
        end
        
        % Compute sample entropy of coarse-grained series
        mse(scale) = compute_sample_entropy(y, m, r);
    end
end

