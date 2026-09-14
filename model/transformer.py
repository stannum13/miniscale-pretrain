from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from miniscale.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def _rope_frequencies(config: ModelConfig) -> tuple[Tensor, Tensor]:
    head_dim = config.hidden_size // config.num_heads
    inverse = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(config.max_sequence_length).float()
    angles = torch.outer(positions, inverse)
    return angles.cos(), angles.sin()


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :].to(dtype=x.dtype)
    sin = sin[None, None, :, :].to(dtype=x.dtype)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        kv_size = config.num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = config.dropout

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        repeats = self.num_heads // self.num_kv_heads
        if repeats > 1:
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.o_proj(output.transpose(1, 2).contiguous().view(batch, length, -1))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.feed_forward = FeedForward(config)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x), cos, sin)
        return x + self.feed_forward(self.ffn_norm(x))


@dataclass
class ModelOutput:
    logits: Tensor
    loss: Tensor | None = None


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        cos, sin = _rope_frequencies(config)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> ModelOutput:
        length = input_ids.shape[1]
        if length > self.config.max_sequence_length:
            raise ValueError("input sequence exceeds configured maximum")
        x = self.token_embedding(input_ids)
        cos, sin = self.rope_cos[:length], self.rope_sin[:length]
        for block in self.blocks:
            if self.config.activation_checkpointing and self.training:
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
            )
        return ModelOutput(logits=logits, loss=loss)


def estimate_model_flops(model: nn.Module, tokens: int, sequence_length: int) -> int:
    base = model.module if hasattr(model, "module") else model
    parameters = sum(parameter.numel() for parameter in base.parameters())
    layers = getattr(getattr(base, "config", None), "num_layers", 0)
    hidden = getattr(getattr(base, "config", None), "hidden_size", 0)
    # 6N is the common forward+backward parameter FLOP estimate; the second
    # term accounts for quadratic attention matmuls omitted by 6N.
    return int(6 * parameters * tokens + 12 * layers * hidden * sequence_length * tokens)


def estimate_config_flops(config: ModelConfig, tokens: int, sequence_length: int) -> int:
    """Global training FLOPs, independent of how FSDP shards live parameters."""
    return int(
        6 * config.estimated_parameters() * tokens
        + 12 * config.num_layers * config.hidden_size * sequence_length * tokens
    )
