"""
Hybrid Deep Neural Network Architecture
Multi-Scale 1D-CNN + BiLSTM + Multi-Head Self-Attention + Two-Stage Heads
STEW EEG Cognitive Stress Decision-Support System
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScale1DCNN(nn.Module):
    """
    Multi-Scale 1D-CNN with parallel receptive fields (k1=5, k2=11).
    Input: (batch_size, 14, 256)
    """
    def __init__(self, in_channels=14, out_channels=32):
        super(MultiScale1DCNN, self).__init__()
        # Branch 1: Fine-grained temporal variations (k1 = 5)
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # Branch 2: Wide-context temporal variations (k2 = 11)
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=11, padding=5),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Merged second conv stage
        self.merged_conv = nn.Sequential(
            nn.Conv1d(out_channels * 2, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

    def forward(self, x):
        # x shape: (B, 14, 256)
        b1 = self.branch1(x)  # (B, 32, 128)
        b2 = self.branch2(x)  # (B, 32, 128)
        concat = torch.cat([b1, b2], dim=1)  # (B, 64, 128)
        out = self.merged_conv(concat)      # (B, 64, 64)
        return out


class TemporalMultiHeadAttention(nn.Module):
    """
    4-Head Temporal Self-Attention over sequence windows.
    Input: (batch_size, seq_len, embed_dim)
    """
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.3):
        super(TemporalMultiHeadAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, attn_weights = self.mha(x, x, x)
        out = self.norm(x + attn_out)
        return out, attn_weights


class HybridNeuroNet(nn.Module):
    """
    Complete Hybrid Network:
    1. Multi-Scale 1D-CNN
    2. 2-Layer BiLSTM
    3. 4-Head Temporal Self-Attention
    4. Two-Stage Hierarchical Heads:
       - Stage 1: Binary (Rest vs Active Stress)
       - Stage 2: Severity Stratification (Low, Moderate, High)
    """
    def __init__(self, in_channels=14, feature_dim=64, lstm_hidden=64, num_heads=4, dropout_rate=0.3):
        super(HybridNeuroNet, self).__init__()
        self.dropout_rate = dropout_rate

        # 1. Multi-Scale 1D-CNN
        self.cnn = MultiScale1DCNN(in_channels=in_channels, out_channels=32)

        # Feature Projection (for tabular features extracted via PCA)
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 2. Bidirectional LSTM (2-layer, hidden=64 -> 128 bidirectional)
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )

        # 3. Temporal Multi-Head Attention
        self.attention = TemporalMultiHeadAttention(embed_dim=lstm_hidden * 2, num_heads=num_heads, dropout=dropout_rate)

        # Unified Representation Fusion
        self.fusion = nn.Sequential(
            nn.Linear(lstm_hidden * 2 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 4. Two-Stage Hierarchical Classification Heads
        # Stage 1: Binary (Normal/Rest = 0 vs Active Stress = 1)
        self.head_stage1_binary = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)
        )

        # Stage 2: Stress Severity (0: Low, 1: Moderate, 2: High)
        self.head_stage2_severity = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 3)
        )

    def forward(self, raw_eeg, feat_vec=None):
        """
        raw_eeg: (B, 14, 256)
        feat_vec: (B, feature_dim) or None
        
        returns: binary_logits (B, 2), severity_logits (B, 3), attn_weights
        """
        # 1. Multi-Scale CNN
        cnn_out = self.cnn(raw_eeg)  # (B, 64, 64)
        cnn_out_t = cnn_out.transpose(1, 2)  # (B, 64, 64) -> (B, seq_len=64, features=64)

        # 2. BiLSTM
        lstm_out, _ = self.bilstm(cnn_out_t)  # (B, 64, 128)

        # 3. Attention
        attn_out, attn_weights = self.attention(lstm_out)  # (B, 64, 128)
        
        # Temporal pooling (mean across 64 sequence points)
        temp_embedding = torch.mean(attn_out, dim=1)  # (B, 128)

        # Feature vector fusion
        if feat_vec is None:
            # Dummy feature projection if none passed
            feat_vec = torch.zeros((raw_eeg.size(0), 64), device=raw_eeg.device)
        
        feat_embedding = self.feature_proj(feat_vec)  # (B, 64)
        unified = torch.cat([temp_embedding, feat_embedding], dim=1)  # (B, 192)
        fused_embedding = self.fusion(unified)  # (B, 128)

        # 4. Heads
        binary_logits = self.head_stage1_binary(fused_embedding)     # (B, 2)
        severity_logits = self.head_stage2_severity(fused_embedding) # (B, 3)

        return binary_logits, severity_logits, attn_weights
