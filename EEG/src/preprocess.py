"""
EEG Signal Preprocessing & Quality Index (SQI) Pipeline
STEW Dataset 14-Channel Architecture (128 Hz)
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew

CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
SAMPLING_RATE = 128  # Hz
WINDOW_SEC = 2.0     # seconds
WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_SEC)  # 256 samples
OVERLAP_RATIO = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_RATIO))  # 128 samples


def butter_bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=128, order=4):
    """
    4th-order zero-phase Butterworth bandpass filter (0.5 Hz - 45.0 Hz).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    y = scipy_signal.filtfilt(b, a, data, axis=-1)
    return y


def notch_filter(data, notch_freq=50.0, fs=128, Q=30.0):
    """
    50 Hz IIR Notch filter to eliminate powerline interference.
    """
    nyq = 0.5 * fs
    w0 = notch_freq / nyq
    b, a = scipy_signal.iirnotch(w0, Q)
    y = scipy_signal.filtfilt(b, a, data, axis=-1)
    return y


def amplitude_clipping(data, threshold=100.0):
    """
    Symmetrical amplitude spike clipping at ±100 µV to suppress ocular/myogenic twitches.
    """
    return np.clip(data, -threshold, threshold)


def preprocess_signal(raw_eeg):
    """
    Full Signal Conditioning:
    1. Bandpass filter (0.5 - 45 Hz)
    2. 50 Hz Notch filter
    3. ±100 µV Amplitude clipping

    raw_eeg shape: (samples, n_channels) or (n_channels, samples)
    returns: conditioned eeg of shape (n_channels, samples)
    """
    data = np.asarray(raw_eeg, dtype=np.float32)
    if data.shape[0] > data.shape[1] and data.shape[1] == len(CHANNELS):
        data = data.T  # Shape: (14, samples)

    # 1. Bandpass filter
    filtered = butter_bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=SAMPLING_RATE, order=4)
    # 2. Notch filter
    notched = notch_filter(filtered, notch_freq=50.0, fs=SAMPLING_RATE, Q=30.0)
    # 3. Spike clipping
    clipped = amplitude_clipping(notched, threshold=100.0)

    return clipped


def compute_sqi(window_data):
    """
    Automated Signal Quality Index (SQI) for a 2-second window (14 channels, 256 samples).
    
    Parameters:
    -----------
    window_data : ndarray of shape (14, 256)
    
    Returns:
    --------
    sqi_score : float ∈ [0.0, 1.0]
    is_corrupted : bool (True if sqi_score < 0.50)
    metrics : dict containing kurtosis, skewness, and high_freq_ratio
    """
    # 1. Kurtosis and Skewness across channels
    kurt_vals = kurtosis(window_data, axis=1, fisher=False)  # Normal ~ 3.0
    skew_vals = skew(window_data, axis=1)                    # Normal ~ 0.0

    # 2. High-Frequency Noise Ratio (power > 35 Hz vs total power 0.5 - 45 Hz)
    fft_vals = np.abs(np.fft.rfft(window_data, axis=1)) ** 2
    freqs = np.fft.rfftfreq(WINDOW_SIZE, d=1.0 / SAMPLING_RATE)

    total_mask = (freqs >= 0.5) & (freqs <= 45.0)
    hf_mask = (freqs >= 35.0) & (freqs <= 45.0)

    total_power = np.sum(fft_vals[:, total_mask], axis=1) + 1e-8
    hf_power = np.sum(fft_vals[:, hf_mask], axis=1)
    hf_ratio_channels = hf_power / total_power
    mean_hf_ratio = float(np.mean(hf_ratio_channels))

    mean_kurt = float(np.mean(kurt_vals))
    mean_skew = float(np.mean(np.abs(skew_vals)))

    # SQI Penalties mapping
    # Ideal Kurtosis is ~3. Excess kurtosis indicates sharp spikes/artifacts.
    kurt_penalty = np.clip(abs(mean_kurt - 3.0) / 7.0, 0.0, 1.0)
    # Ideal Skewness is ~0. High skewness indicates asymmetry/blink artifacts.
    skew_penalty = np.clip(mean_skew / 2.5, 0.0, 1.0)
    # High frequency ratio > 0.35 indicates muscle twitch / EMG contamination.
    hf_penalty = np.clip(mean_hf_ratio / 0.35, 0.0, 1.0)

    sqi_score = 1.0 - (0.4 * kurt_penalty + 0.3 * skew_penalty + 0.3 * hf_penalty)
    sqi_score = float(np.clip(sqi_score, 0.0, 1.0))

    is_corrupted = sqi_score < 0.50
    metrics = {
        'kurtosis': mean_kurt,
        'skewness': mean_skew,
        'high_freq_ratio': mean_hf_ratio,
        'sqi': sqi_score
    }
    return sqi_score, is_corrupted, metrics


def segment_windows(eeg_data, window_size=256, step_size=128):
    """
    Segments raw continuous EEG data into 2.0-sec sliding windows with 50% overlap.
    
    eeg_data shape: (14, total_samples)
    returns: windows of shape (N, 14, 256), list of sqi_scores
    """
    n_channels, total_samples = eeg_data.shape
    windows = []
    sqi_scores = []
    sqi_flags = []

    for start_idx in range(0, total_samples - window_size + 1, step_size):
        end_idx = start_idx + window_size
        win = eeg_data[:, start_idx:end_idx]
        sqi, corrupted, _ = compute_sqi(win)
        windows.append(win)
        sqi_scores.append(sqi)
        sqi_flags.append(corrupted)

    return np.array(windows, dtype=np.float32), np.array(sqi_scores, dtype=np.float32), np.array(sqi_flags, dtype=bool)
