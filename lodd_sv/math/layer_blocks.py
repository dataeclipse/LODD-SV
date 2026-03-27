
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import math


class ModifiedTransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

    def forward(
        self,
        x: torch.Tensor,
        t_emb: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        if t_emb is not None:
            x = x + t_emb.unsqueeze(1)
        x = self.norm2(x + self.ff(x))
        return x


class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, d_model: int, max_steps: int = 1000) -> None:
        super().__init__()
        self.d_model = d_model
        half = d_model // 2
        pe = torch.zeros(max_steps, d_model)
        position = torch.arange(0, max_steps, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, half, 2).float() * (-math.log(10000.0) / half))
        pe[:, 0:half:2] = torch.sin(position * div_term)
        pe[:, 1:half:2] = torch.cos(position * div_term)
        if d_model % 2:
            pe[:, -1] = torch.cos(position.squeeze(-1) * div_term[-1])
        self.register_buffer("pe", pe)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.pe[t.clamp(0, self.pe.size(0) - 1)]


class DenoisingStack(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int = 4,
        max_seq_len: int = 512,
        max_steps: int = 1000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.time_embed = SinusoidalTimeEmbedding(d_model, max_steps)
        self.time_proj = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList([
            ModifiedTransformerBlock(
                d_model, num_heads, max_seq_len=max_seq_len, dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_hiddens: bool = False,
    ) -> tuple:
        h = self.token_embed(token_ids)
        t_emb = self.time_proj(self.time_embed(t))
        hiddens = [] if return_hiddens else None
        for layer in self.layers:
            h = layer(h, t_emb=t_emb, mask=mask)
            if return_hiddens:
                hiddens.append(h.detach())
        logits = self.head(h)
        return logits, hiddens
