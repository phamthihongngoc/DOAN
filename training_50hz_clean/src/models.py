from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(hidden * 2, hidden * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden * 4),
            nn.ReLU(inplace=True),
        )
        self.feature = nn.Linear(hidden * 4, hidden * 4)
        self.classifier = nn.Linear(hidden * 4, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # x: (batch, time, channels)
        x = x.transpose(1, 2)
        feats = self.net(x)
        feats = feats.mean(dim=-1)
        feats = self.feature(feats)
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits


class LSTMModel(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, hidden: int = 128, layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_channels,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.2 if layers > 1 else 0.0,
        )
        self.feature = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        out, _ = self.lstm(x)
        feats = out[:, -1, :]
        feats = self.feature(feats)
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        return x + self.pe[:, :length]


class TransformerModel(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, d_model: int = 128, layers: int = 2) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_channels, d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.feature = nn.Linear(d_model, d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        feats = x.mean(dim=1)
        feats = self.feature(feats)
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits


@dataclass
class ModelConfig:
    name: str
    input_channels: int
    num_classes: int


def create_model(cfg: ModelConfig) -> nn.Module:
    name = cfg.name.lower()
    if name == "cnn":
        return CNN1D(cfg.input_channels, cfg.num_classes)
    if name == "lstm":
        return LSTMModel(cfg.input_channels, cfg.num_classes)
    if name == "transformer":
        return TransformerModel(cfg.input_channels, cfg.num_classes)
    raise ValueError(f"Unknown model: {cfg.name}")
