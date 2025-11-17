clear; clc;
eeglab; close;

%% ------------ User toggle ------------
EPOCH_SEC = 2;     % set to 2 or 5

%% ------------ Paths ------------
root_dir = '/Volumes/NED_Backup3';
directory = fullfile(root_dir, sprintf('COMBINED_Q1K_BC_%ds', EPOCH_SEC), 'GENIAL');

listing    = dir(fullfile(directory, '*.set'));
file_names = {listing.name};
names      = file_names(~startsWith(file_names, '._')); % remove mac hidden files
nSub       = numel(names);

if nSub == 0
    error('No .set files found in: %s', directory);
end

%% ------------ Common params ------------
f_range  = [1, 40];
settings = struct('peak_width_limits', [0.5, 18], 'max_n_peaks', 10, 'aperiodic_mode', 'fixed');

% Entropy params (5 s)
m = 2;
r_frac = 0.15;   % with per-epoch z-scoring
scales = 1:20;

%% ------------ Probe first file for sizes ------------
EEG0  = pop_loadset(fullfile(directory, names{1}));
Fs    = EEG0.srate;
nCh   = EEG0.nbchan;
epLen = size(EEG0.data, 2);   % samples per epoch
nTr0  = size(EEG0.data, 3);
clear EEG0;

%% Welch settings tuned by epoch length
if EPOCH_SEC == 2
    win   = hamming(round(Fs*1.0));    % 1 s window
    nover = floor(numel(win)/2);        % 50% overlap
    nfft  = 2^nextpow2(epLen);
else
    % if you ever want PSD on 5 s, keep a longer window
    win   = hamming(round(Fs*2.0));
    nover = floor(numel(win)/2);
    nfft  = 2^nextpow2(epLen);
end

% get PSD frequency grid length
[~, f_probe] = pwelch(double(randn(epLen, 1)), win, nover, nfft, Fs);
nF = numel(f_probe);

%% ------------ Prealloc by mode ------------
% Always keep name and f
feature = struct();
feature.name = names;
feature.f    = f_probe;
feature.epoch_seconds = EPOCH_SEC;

if EPOCH_SEC == 2
    % bands up to 40 Hz
    bands = struct( ...
        'delta', [1, 3.5], ...
        'theta', [4, 7], ...
        'alpha', [7.5, 12.5], ...
        'beta',  [13, 30], ...
        'full',  [1, 40] ...
    );
    band_names = fieldnames(bands);
    nBands = numel(band_names);

    hurst_legacy      = nan(nCh, nSub);
    pxx_t             = nan(nCh, nSub, nF);

    fooof_results_t       = cell(nSub, nCh);
    aperiodic_component_t = nan(nSub, nCh, nF);
    full_spectrum_t       = nan(nSub, nCh, nF);
    periodic_component_t  = nan(nSub, nCh, nF);
    fooofed_spectrum_t    = nan(nSub, nCh, nF);
    offset                = nan(nSub, nCh);
    exponent              = nan(nSub, nCh);

    PSD          = nan(nSub, nCh, nBands);
    relative_PSD = nan(nSub, nCh, nBands);
    periodic_PSD   = nan(nSub, nCh, nBands-1);  % exclude 'full'
    periodic_PSD_m = nan(nSub, nCh, nBands-1);

else % EPOCH_SEC == 5
    % Entropy arrays
    nSc = numel(scales);
    MSE_mean      = nan(nCh, nSc, nSub);
    CI_all        = nan(nSub, nCh);
    CI_low        = nan(nSub, nCh);
    CI_high       = nan(nSub, nCh);
    SampleEntropy = nan(nSub, nCh);

    % RSA Hurst
    hurst_RSA_mean = nan(nCh, nSub);
end

%% ------------ Main loop ------------
for sub = 1:nSub
    EEG = pop_loadset(fullfile(directory, names{sub}));

    for ch = 1:EEG.nbchan
        nTr = size(EEG.data, 3);

        if EPOCH_SEC == 2
            % ---------- 2 s mode: PSD + FOOOF + legacy Hurst ----------
            hurst1 = nan(nTr, 1);
            pxx1   = nan(nTr, nF);

            for tr = 1:nTr
                sig = double(squeeze(EEG.data(ch, :, tr)));
                sig = detrend(sig, 1);

                % legacy hurst (user function already available)
                hurst1(tr) = estimate_hurst_exponent(sig);

                % PSD
                [pxx1(tr, :), f] = pwelch(sig, win, nover, nfft, EEG.srate);
            end

            % average across epochs
            hurst_legacy(ch, sub) = mean(hurst1, 'omitnan');
            pxx = mean(pxx1, 1, 'omitnan');
            pxx_t(ch, sub, :) = pxx;

            % FOOOF on 1 to 40 Hz slice
            keep = f >= f_range(1) & f <= f_range(2);
            f_foof = f(keep);
            pxx_foof = pxx(keep);

            fooof_res = fooof(f_foof', pxx_foof', f_range, settings, true);
            fooof_results_t{sub, ch} = fooof_res;

            ap  = fooof_res.ap_fit(:)';
            sp  = fooof_res.power_spectrum(:)';
            per = sp - ap;

            aperiodic_component_t(sub, ch, keep) = ap;
            full_spectrum_t(sub,      ch, keep) = sp;
            periodic_component_t(sub,  ch, keep) = per;
            fooofed_spectrum_t(sub,    ch, keep) = fooof_res.fooofed_spectrum(:)';

            offset(sub, ch)   = fooof_res.aperiodic_params(1);
            exponent(sub, ch) = fooof_res.aperiodic_params(2);

            % Absolute and relative band power
            for b = 1:nBands
                br  = bands.(band_names{b});
                idx = f >= br(1) & f <= br(2);
                PSD(sub, ch, b) = mean(pxx(idx), 'omitnan');
            end
            % Relative: divide by 'full'
            idx_full = strcmp(band_names, 'full');
            tot = PSD(sub, ch, idx_full);
            for b = 1:nBands
                if isfinite(PSD(sub, ch, b)) && isfinite(tot) && tot > 0
                    relative_PSD(sub, ch, b) = PSD(sub, ch, b) / tot;
                else
                    relative_PSD(sub, ch, b) = NaN;
                end
            end

            % Periodic band power (area and mean) from FOOOF periodic component
            for b = 1:nBands-1
                br = bands.(band_names{b});
                idx = f >= br(1) & f <= br(2) & keep;
                if any(idx)
                    % map periodic part to the f grid slice
                    per_slice = nan(size(f));
                    per_slice(keep) = per; % per is on f_foof
                    pseg = per_slice(idx);
                    fseg = f(idx);
                    periodic_PSD(sub, ch, b)   = trapz(fseg, pseg);
                    periodic_PSD_m(sub, ch, b) = mean(pseg, 'omitnan');
                else
                    periodic_PSD(sub, ch, b)   = NaN;
                    periodic_PSD_m(sub, ch, b) = NaN;
                end
            end

        else
            % ---------- 5 s mode: Entropy + Hurst via RSA ----------
            % per-epoch accumulators
            mse_accum = nan(nTr, numel(scales)); % will fill per epoch then average per channel
            se_accum  = nan(nTr, 1);
            hRSA_acc  = nan(nTr, 1);

            for tr = 1:nTr
                sig = double(squeeze(EEG.data(ch, :, tr)));

                % z-score per epoch for stable r
                sigz = (sig - mean(sig)) / std(sig);

                % MSE per epoch
                % get_mse should return a row vector over 'scales'
                [mse_row, ~, ~, ~] = get_mse(sigz, m, r_frac, scales);
                mse_accum(tr, :) = mse_row;

                % Sample entropy per epoch
                se_accum(tr) = sampen(sigz, m, r_frac);

                % RSA Hurst per epoch
                hRSA_acc(tr) = hurst_RSA(sig);
            end

            % average across epochs
            MSE_mean(ch, :, sub) = mean(mse_accum, 1, 'omitnan');
            SampleEntropy(sub, ch) = mean(se_accum, 1, 'omitnan');
            hurst_RSA_mean(ch, sub) = mean(hRSA_acc, 1, 'omitnan');

            % Complexity Index from MSE curve
            ciCurve = squeeze(MSE_mean(ch, :, sub)); % 1 x nSc
            x = scales;
            mid = floor(numel(x) / 2);
            CI_all(sub, ch)  = trapz(x, ciCurve);
            CI_low(sub, ch)  = trapz(x(1:mid), ciCurve(1:mid));
            CI_high(sub, ch) = trapz(x(mid+1:end), ciCurve(mid+1:end));
        end
    end
end

%% ------------ Pack and save ------------
if EPOCH_SEC == 2
    feature.f_range_fooof         = f_range;
    feature.hurst_legacy          = hurst_legacy';         % [sub x ch]
    feature.pxx_t                 = pxx_t;                 % [ch x sub x f]
    feature.fooof_results_t       = fooof_results_t;
    feature.aperiodic_component_t = aperiodic_component_t; % [sub x ch x f]
    feature.full_spectrum_t       = full_spectrum_t;
    feature.periodic_component_t  = periodic_component_t;
    feature.fooofed_spectrum_t    = fooofed_spectrum_t;
    feature.offset                = offset;
    feature.exponent              = exponent;
    feature.band_names            = band_names;
    feature.PSD                   = PSD;
    feature.relative_PSD          = relative_PSD;
    feature.periodic_PSD          = periodic_PSD;
    feature.periodic_PSD_m        = periodic_PSD_m;

    save(fullfile(directory, 'feature_2s.mat'), 'feature', '-v7.3');

else % 5 s
    feature.scales         = scales;
    feature.m              = m;
    feature.r_fraction     = r_frac;
    feature.MSE_mean       = MSE_mean;      % [ch x scale x sub]
    feature.CI             = CI_all;        % [sub x ch]
    feature.CI_lowScale    = CI_low;        % [sub x ch]
    feature.CI_highScale   = CI_high;       % [sub x ch]
    feature.SampleEntropy  = SampleEntropy; % [sub x ch]
    feature.hurst_RSA      = hurst_RSA_mean'; % [sub x ch]

    save(fullfile(directory, 'feature_5s.mat'), 'feature', '-v7.3');
end

%% ------------ Helper: RSA Hurst ------------
function H = hurst_RSA(signal)
    % Rescaled Range Analysis for Hurst exponent
    x = detrend(double(signal(:)), 1);
    N = numel(x);
    % window sizes from 10 samples up to N/4, log spaced
    nVals = unique(floor(logspace(log10(10), log10(max(20, N/4)), 20)));
    RS = nan(size(nVals));
    for i = 1:numel(nVals)
        n = nVals(i);
        if n < 10 || n > N, continue; end
        nBlocks = floor(N / n);
        if nBlocks < 2, RS(i) = NaN; continue; end
        rsb = nan(nBlocks, 1);
        for b = 1:nBlocks
            seg = x((b-1)*n + 1 : b*n);
            Y = cumsum(seg - mean(seg));
            R = max(Y) - min(Y);
            S = std(seg);
            if S > 0
                rsb(b) = R / S;
            end
        end
        RS(i) = mean(rsb, 'omitnan');
    end
    good = isfinite(RS) & RS > 0;
    if nnz(good) >= 5
        p = polyfit(log(nVals(good)), log(RS(good)), 1);
        H = p(1);
    else
        H = NaN;
    end
end
