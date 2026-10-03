"""
Explainability Engine: SHAP / Gradient Feature Attribution & Physiological Rhythm Deviations
STEW EEG Cognitive Stress System
"""

import numpy as np
import torch

CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']


def compute_rhythm_deviations(active_band_powers, resting_band_powers):
    """
    Computes physiological percentage band power deviations against baseline:
    ΔPower_band = ((Power_active - Power_resting) / (Power_resting + 1e-8)) * 100%
    
    returns: dict of percentage deviations for alpha, beta, theta, delta, gamma
    """
    deviations = {}
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    # Aggregate powers across all 14 channels
    active_agg = {b: 0.0 for b in bands}
    resting_agg = {b: 0.0 for b in bands}

    for ch in CHANNELS:
        for b in bands:
            active_agg[b] += active_band_powers[ch].get(b, 0.0)
            resting_agg[b] += resting_band_powers[ch].get(b, 0.0)

    for b in bands:
        rest_val = resting_agg[b] + 1e-8
        act_val = active_agg[b]
        pct = ((act_val - rest_val) / rest_val) * 100.0
        deviations[b] = float(pct)

    return deviations


def compute_gradient_attributions(model, raw_eeg, feat_vec=None, target_class=1):
    """
    Computes saliency / gradient feature attributions for input raw EEG channels and features.
    
    raw_eeg: (1, 14, 256)
    returns: channel_importance dict (14 channels -> float score), top_features dict
    """
    model.eval()
    device = next(model.parameters()).device

    if not isinstance(raw_eeg, torch.Tensor):
        x = torch.tensor(raw_eeg, dtype=torch.float32, device=device)
    else:
        x = raw_eeg.clone().detach().to(device)

    if x.dim() == 2:
        x = x.unsqueeze(0)

    x.requires_grad = True

    if feat_vec is not None:
        if not isinstance(feat_vec, torch.Tensor):
            f = torch.tensor(feat_vec, dtype=torch.float32, device=device)
        else:
            f = feat_vec.clone().detach().to(device)
        if f.dim() == 1:
            f = f.unsqueeze(0)
        f.requires_grad = True
    else:
        f = None

    binary_logits, _, _ = model(x, f)
    score = binary_logits[0, target_class]
    model.zero_grad()
    score.backward()

    # Gradient magnitude across time per channel
    grads = x.grad.abs().cpu().numpy()[0]  # (14, 256)
    ch_grads = np.mean(grads, axis=1)      # (14,)

    # Normalize to [0, 1]
    norm_grads = ch_grads / (np.sum(ch_grads) + 1e-8)

    channel_importance = {ch: float(norm_grads[i]) for i, ch in enumerate(CHANNELS)}
    return channel_importance


def generate_explanation_report(active_band_powers, resting_band_powers, channel_importance=None):
    """
    Generates structured diagnostic explanation strings for dashboard.
    """
    devs = compute_rhythm_deviations(active_band_powers, resting_band_powers)
    
    alpha_dev = devs['alpha']
    beta_dev = devs['beta']
    theta_dev = devs['theta']

    insights = []
    
    if beta_dev > 15.0:
        insights.append(f"Beta Power Surge: +{beta_dev:.1f}% (Heightened Cortical Arousal / Task Workload)")
    elif beta_dev < -10.0:
        insights.append(f"Beta Power Reduction: {beta_dev:.1f}% (Low Mental Engagement)")

    if alpha_dev < -10.0:
        insights.append(f"Alpha Suppression: {alpha_dev:.1f}% (Active Mental Effort & Sensory Processing)")
    elif alpha_dev > 15.0:
        insights.append(f"Alpha Synchronization: +{alpha_dev:.1f}% (Relaxed / Idling State)")

    if theta_dev > 15.0:
        insights.append(f"Frontal Theta Elevation: +{theta_dev:.1f}% (Working Memory Load)")

    if not insights:
        insights.append("Biosignal rhythms aligned within normal resting baseline variance.")

    top_channels = []
    if channel_importance:
        sorted_chs = sorted(channel_importance.items(), key=lambda x: x[1], reverse=True)
        top_channels = [f"{ch} ({val*100:.1f}%)" for ch, val in sorted_chs[:4]]

    return {
        'deviations': devs,
        'insights': insights,
        'top_channels': top_channels
    }
