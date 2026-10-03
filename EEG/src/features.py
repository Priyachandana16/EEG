"""
Feature Extraction & Adaptive Personal Baseline Normalization
STEW EEG Cognitive Stress System
"""

import numpy as np
from scipy import signal as scipy_signal
import pywt
from sklearn.decomposition import PCA

CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 45.0)
}


def extract_welch_psd_features(window_data, fs=128):
    """
    Computes Welch PSD for canonical EEG bands across 14 channels.
    
    window_data: shape (14, 256)
    returns: psd_features shape (14 * 5,) = 70 features, and band_powers dict per channel
    """
    n_channels = window_data.shape[0]
    psd_feats = []
    band_powers_by_channel = {ch: {} for ch in CHANNELS}

    for ch_idx in range(n_channels):
        freqs, psd = scipy_signal.welch(window_data[ch_idx], fs=fs, nperseg=128, noverlap=64)
        ch_name = CHANNELS[ch_idx]
        
        for band_name, (f_min, f_max) in BANDS.items():
            mask = (freqs >= f_min) & (freqs <= f_max)
            trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
            power = trapz_fn(psd[mask], freqs[mask]) if np.any(mask) else 1e-8
            power = max(float(power), 1e-8)
            psd_feats.append(power)
            band_powers_by_channel[ch_name][band_name] = power

    return np.array(psd_feats, dtype=np.float32), band_powers_by_channel


def extract_stress_ratios(band_powers_by_channel):
    """
    Computes Neurological Stress Ratios across channels:
    - Theta / Beta (TBR)
    - Alpha / Beta (ABR)
    - Theta / Alpha (TAR)
    
    returns: ratio_features shape (14 * 3,) = 42 features
    """
    ratio_feats = []
    for ch in CHANNELS:
        powers = band_powers_by_channel[ch]
        theta = powers['theta']
        alpha = powers['alpha']
        beta = powers['beta']

        tbr = theta / (beta + 1e-6)
        abr = alpha / (beta + 1e-6)
        tar = theta / (alpha + 1e-6)

        ratio_feats.extend([tbr, abr, tar])

    return np.array(ratio_feats, dtype=np.float32)


def extract_wpt_features(window_data, wavelet='db4', max_level=2):
    """
    Wavelet Packet Transform (WPT) energy and entropy statistics using 'db4'.
    
    window_data: shape (14, 256)
    returns: wpt_features shape (14 * 2^max_level * 2,)
    """
    n_channels = window_data.shape[0]
    wpt_feats = []

    for ch_idx in range(n_channels):
        wp = pywt.WaveletPacket(data=window_data[ch_idx], wavelet=wavelet, mode='symmetric', maxlevel=max_level)
        nodes = [node.path for node in wp.get_level(max_level, 'freq')]
        
        for node_path in nodes:
            coeffs = wp[node_path].data
            # Energy
            energy = np.sum(coeffs ** 2)
            # Shannon Entropy
            prob = (coeffs ** 2) / (energy + 1e-8)
            prob = prob[prob > 0]
            entropy = -np.sum(prob * np.log2(prob + 1e-12))
            
            wpt_feats.extend([energy, entropy])

    return np.array(wpt_feats, dtype=np.float32)


def extract_window_feature_vector(window_data, fs=128):
    """
    Extracts complete feature vector for a single 2-sec EEG window (14 channels, 256 samples).
    Total features = 70 (PSD) + 42 (Ratios) + 112 (WPT) = 224 features.
    """
    psd_feats, band_powers = extract_welch_psd_features(window_data, fs=fs)
    ratio_feats = extract_stress_ratios(band_powers)
    wpt_feats = extract_wpt_features(window_data, wavelet='db4', max_level=2)

    full_vector = np.concatenate([psd_feats, ratio_feats, wpt_feats], axis=0)
    return full_vector, band_powers


def extract_subject_baseline(resting_windows, fs=128):
    """
    Computes mean resting baseline feature vector μ_baseline from resting segment (subXX_lo.txt).
    
    resting_windows: shape (N_rest, 14, 256)
    returns: mu_baseline shape (224,)
    """
    baseline_feats = []
    for win in resting_windows:
        feat_vec, _ = extract_window_feature_vector(win, fs=fs)
        baseline_feats.append(feat_vec)
    
    baseline_feats = np.array(baseline_feats)
    mu_baseline = np.mean(baseline_feats, axis=0)
    return mu_baseline


def normalize_with_personal_baseline(feature_vector, mu_baseline):
    """
    Adaptive Personal Baseline Normalization:
    z_i = (x_i - μ_baseline) / (|μ_baseline| + 1e-6)
    """
    denom = np.abs(mu_baseline) + 1e-6
    z_i = (feature_vector - mu_baseline) / denom
    return z_i


class LeakageFreePCA:
    """
    Strictly leakage-free PCA dimensionality reduction fitted exclusively on training folds.
    """
    def __init__(self, n_components=64):
        self.pca = PCA(n_components=n_components)
        self.fitted = False

    def fit(self, X_train):
        self.pca.fit(X_train)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("PCA must be fitted on training fold first!")
        return self.pca.transform(X)

    def fit_transform(self, X_train):
        self.fit(X_train)
        return self.transform(X_train)
