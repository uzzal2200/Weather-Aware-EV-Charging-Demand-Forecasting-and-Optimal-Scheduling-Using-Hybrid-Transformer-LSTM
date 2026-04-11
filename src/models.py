"""
models.py — Transformer-LSTM and all baseline model definitions.
"""
import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from config import (
    D_MODEL, N_HEADS, D_FF, N_LAYERS,
    LSTM_HIDDEN, DROPOUT, LOOKBACK, FEATURES,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RANDOM_SEED
)


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Transformer–LSTM  (proposed model)
# ─────────────────────────────────────────────────────────────────────────────
class TransformerLSTM(nn.Module):
    """
    Paper architecture:
        Input (T×F) → Linear Embedding → Pos. Enc.
        → N stacked Transformer Encoder blocks (Multi-Head SA + FFN + LayerNorm)
        → LSTM (hidden=128)
        → Dropout → Dense output
    """
    def __init__(
        self,
        in_features: int  = FEATURES,
        d_model: int      = D_MODEL,
        n_heads: int      = N_HEADS,
        d_ff: int         = D_FF,
        n_layers: int     = N_LAYERS,
        lstm_hidden: int  = LSTM_HIDDEN,
        dropout: float    = DROPOUT,
        horizon: int      = 1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.lstm    = nn.LSTM(d_model, lstm_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(lstm_hidden, horizon)

    def forward(self, x):                          # x: (B, T, F)
        x = self.input_proj(x)                     # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x)            # (B, T, d_model)
        out, _ = self.lstm(x)                      # (B, T, lstm_hidden)
        out = self.dropout(out[:, -1, :])          # take last step
        return self.fc(out)                        # (B, horizon)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone LSTM baseline
# ─────────────────────────────────────────────────────────────────────────────
class StandaloneLSTM(nn.Module):
    def __init__(
        self,
        in_features: int = FEATURES,
        hidden: int      = LSTM_HIDDEN,
        dropout: float   = DROPOUT,
        horizon: int     = 1,
    ):
        super().__init__()
        self.lstm    = nn.LSTM(in_features, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


# ─────────────────────────────────────────────────────────────────────────────
# Sklearn baselines (flat feature matrices)
# ─────────────────────────────────────────────────────────────────────────────
def get_linear_regression():
    return LinearRegression()


def get_random_forest():
    return RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """Flatten (N, T, F) → (N, T*F) for sklearn models."""
    return X.reshape(len(X), -1)
