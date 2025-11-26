clear;
clc;
eeglab; close;

% Features extracted:
%   From 2s epochs: Hurst exponent, band powers, FOOOF parameters, periodic powers
%   From 5s epochs: Fractal dimensions, sample entropy, MSE, complexity index

%% --- Configuration ---
% Set to '2s' or '5s' to choose epoch length
epoch_length = '2s';  % Change this to '5s' to use 5-second epochs

%% --- Load EEG Files ---
if strcmp(epoch_length, '2s')
    directory = '/Volumes/NED_Backup3/COMBINED_Q1K_BC_2s/curated_list_for_genial_2s';
elseif strcmp(epoch_length, '5s')
    directory = '/Volumes/NED_Backup3/COMBINED_Q1K_BC_5s/curated_list_for_genial_5s';  % Update this path as needed
else
    error('Invalid epoch_length. Must be either ''2s'' or ''5s''.');
end 
listing = dir(fullfile(directory, '*.set'));
file_names = {listing.name};
names = file_names(~startsWith(file_names, '._')); % Remove macOS hidden files

f_range = [1, 40];
settings = struct('peak_width_limits', [0.5, 18], 'max_n_peaks', 10);

% Channels to exclude from analysis
excluded_chans = [48, 119, 43, 49, 56, 63, 68, 73, 81, 88, 94, 99, 107, 113, 120, 125, 126, 127, 128, 17, 129];

%% Loop through each participant
for sub = 1:length(names)
    id = names{sub};
    EEG = pop_loadset(fullfile(directory, id)); % Load EEG file
    EEG_label = {EEG.chanlocs.labels};
    Fs = EEG.srate;

    for ch = 1:EEG.nbchan
        
        % Skip excluded channels
        if ismember(ch, excluded_chans)
            continue;
        end

        %% --- 5s EPOCH FEATURES: Entropy-based measures ---
        if strcmp(epoch_length, '5s')
            %% --- Multiscale Entropy (MSE) & Complexity Index ---
            m = 2; r = 0.5; scales = 1:20;
            [mean_mse(ch,:,sub), ~, ~, scales] = get_mse(squeeze(EEG.data(ch,:,:)), m, r, scales);
            
            ci = squeeze(mean_mse(ch,:,sub));
            CI(sub,ch) = trapz(scales, ci);
            CI_lowScale(sub,ch) = trapz(scales(1:floor(end/2)), ci(1:floor(end/2)));
            CI_highScale(sub,ch) = trapz(scales(floor(end/2)+1:end), ci(floor(end/2)+1:end));
            
            %% --- Sample Entropy ---
            for tr = 1:size(EEG.data, 3)
                signal = squeeze(EEG.data(ch,:,tr));
                sampe(tr) = sampen(signal, 2, 0.5);
            end
            SE(sub,ch) = mean(sampe);
        end
        
        %% --- 2s EPOCH FEATURES: Hurst, Band Powers, FOOOF, Periodic ---
        if strcmp(epoch_length, '2s')
            for tr = 1:size(EEG.data, 3)
                signal = squeeze(EEG.data(ch,:,tr));

                %% --- Hurst Exponent ---
                hurst1(tr) = estimate_hurst_exponent(signal);

                %% --- Power Spectral Density (Welch) ---
                window = hamming(1000);
                N = length(signal)/2;
                noverlap = length(window)/2;
                [pxx1(tr,:), f] = pwelch(signal, window, noverlap, N, Fs);
            end

            hurst(ch,sub) = mean(hurst1);
            
            pxx = mean(pxx1, 1);
            pxx_t(ch,sub,:) = pxx;

            %% --- FOOOF Spectral Decomposition ---
            fooof_results = fooof(f', squeeze(pxx)', f_range, settings, true);
            fooof_results_t{sub,ch} = fooof_results;

            %% --- Absolute Band Power ---
            f1 = find(f>=1 & f<=3.5);     % Delta
            f2 = find(f>=4 & f<=7);       % Theta
            f3 = find(f>=7.5 & f<=12.5);  % Alpha
            f4 = find(f>=13 & f<=30);     % Beta
            f5 = find(f>=31 & f<=57);     % Gamma (low)
            f6 = find(f>=62 & f<=80);     % Gamma (high)
            f7 = find(f>=0.5 & f<=80);    % Full spectrum

            PSD(sub,ch,1) = mean(pxx(f1));
            PSD(sub,ch,2) = mean(pxx(f2));
            PSD(sub,ch,3) = mean(pxx(f3));
            PSD(sub,ch,4) = mean(pxx(f4));
            PSD(sub,ch,5) = mean(pxx(f5));
            PSD(sub,ch,6) = mean(pxx(f6));
            PSD(sub,ch,7) = mean(pxx(f7));

            %% --- Relative Band Power ---
            for band = 1:size(PSD,3)-1
                relative_PSD(sub,ch,band) = PSD(sub,ch,band) / sum(PSD(sub,ch,:));
            end

            %% --- FOOOF Components ---
            aperiodic_component = fooof_results.ap_fit;
            full_spectrum = fooof_results.power_spectrum;
            periodic_component = full_spectrum - aperiodic_component;

            aperiodic_component_t(sub,ch,:) = aperiodic_component;
            full_spectrum_t(sub,ch,:) = full_spectrum;
            periodic_component_t(sub,ch,:) = periodic_component;
            fooofed_spectrum_t(sub,ch,:) = fooof_results.fooofed_spectrum;

            offset(sub,ch) = fooof_results.aperiodic_params(1);
            exponent(sub,ch) = fooof_results.aperiodic_params(2);

            %% --- Periodic Band Power (AUC and Mean) ---
            periodic_PSD(sub,ch,1) = trapz(periodic_component(f1));
            periodic_PSD(sub,ch,2) = trapz(periodic_component(f2));
            periodic_PSD(sub,ch,3) = trapz(periodic_component(f3));
            periodic_PSD(sub,ch,4) = trapz(periodic_component(f4));
            periodic_PSD(sub,ch,5) = trapz(periodic_component(f5(f5<=40)));

            periodic_PSD_m(sub,ch,1) = mean(periodic_component(f1));
            periodic_PSD_m(sub,ch,2) = mean(periodic_component(f2));
            periodic_PSD_m(sub,ch,3) = mean(periodic_component(f3));
            periodic_PSD_m(sub,ch,4) = mean(periodic_component(f4));
            periodic_PSD_m(sub,ch,5) = mean(periodic_component(f5(f5<=40)));
        end
    end
end

% Save only variables that were computed for this epoch length
clear feature
feature.name = names;
feature.epoch_length = epoch_length;

if strcmp(epoch_length, '2s')
    % 2s epoch features
    feature.aperiodic_component_t = aperiodic_component_t;
    feature.fooof_results_t = fooof_results_t;
    feature.fooofed_spectrum_t = fooofed_spectrum_t;
    feature.full_spectrum_t = full_spectrum_t;
    feature.offset = offset;
    feature.exponent = exponent;
    feature.periodic_component_t = periodic_component_t;
    feature.periodic_PSD = periodic_PSD;
    feature.periodic_PSD_m = periodic_PSD_m;
    feature.PSD = PSD;
    feature.relative_PSD = relative_PSD;
    feature.hurst = hurst';
    feature.pxx_t = pxx_t;
    
    % Only include optional fields if they exist
    if exist('APF_fooof', 'var'), feature.APF_fooof = APF_fooof; end
    if exist('APF_fooof_ROI', 'var'), feature.APF_fooof_ROI = APF_fooof_ROI; end
    if exist('sgf_t', 'var'), feature.sgf_t = sgf_t; end
    
elseif strcmp(epoch_length, '5s')
    % 5s epoch features
    feature.mean_mse = mean_mse;
    feature.CI = CI;
    feature.CI_lowScale = CI_lowScale;
    feature.CI_highScale = CI_highScale;
    feature.SE = SE;
end

% Save features to specified output directory
output_dir = '/Volumes/NED_Backup3/Features';
save(fullfile(output_dir, 'feature.mat'), 'feature');
