from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.settings import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * rms * self.weight.float()).to(dtype=x.dtype)


def _build_norm(config: ModelConfig) -> nn.Module:
    if config.norm_type == "layernorm":
        return nn.LayerNorm(config.n_embd, eps=config.norm_eps, bias=config.bias)
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.n_embd, eps=config.norm_eps)
    raise ValueError(f"Unsupported norm_type: {config.norm_type}")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head dimension, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        angles = torch.cat((freqs, freqs), dim=-1)
        cos = angles.cos().to(dtype=dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin

    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self(q.size(-2), device=q.device, dtype=q.dtype)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        return q, k


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else config.n_heads
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_heads
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.qk_norm = config.qk_norm
        self.rotary = None
        self.rotary_dim = 0
        if config.positional_embedding == "rope":
            rotary_dim = min(self.head_dim, int(round(self.head_dim * config.rope_fraction)))
            if rotary_dim % 2 != 0:
                rotary_dim -= 1
            if rotary_dim >= 2:
                self.rotary_dim = rotary_dim
                self.rotary = RotaryEmbedding(self.rotary_dim, base=config.rope_theta)

        # Separate Q, K, V projections (K/V smaller when n_kv_heads < n_heads, for GQA/MQA)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.kv_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.kv_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.rotary is not None:
            q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
            k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
            q_rot, k_rot = self.rotary.apply_rotary(q_rot, k_rot)
            q = torch.cat((q_rot, q_pass), dim=-1)
            k = torch.cat((k_rot, k_pass), dim=-1)
        if self.qk_norm:
            scale = math.sqrt(self.head_dim)
            q = F.normalize(q, dim=-1, eps=1e-6) * scale
            k = F.normalize(k, dim=-1, eps=1e-6) * scale

        # GQA: repeat K/V heads to match Q heads
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_dim = max(1, int(round(config.mlp_hidden_mult * config.n_embd)))
        self.mlp_type = config.mlp_type

        if config.mlp_type == "gelu":
            self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        elif config.mlp_type == "swiglu":
            self.c_fc = nn.Linear(config.n_embd, 2 * hidden_dim, bias=config.bias)
        else:
            raise ValueError(f"Unsupported mlp_type: {config.mlp_type}")

        self.hidden_dim = hidden_dim
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mlp_type == "gelu":
            x = F.gelu(self.c_fc(x))
        else:
            gate, value = self.c_fc(x).chunk(2, dim=-1)
            x = F.silu(gate) * value
        return self.dropout(self.c_proj(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.block_style = config.block_style
        self.ln_1 = _build_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = _build_norm(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.block_style == "parallel":
            return x + self.attn(self.ln_1(x)) + self.mlp(self.ln_2(x))
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        transformer_layers: dict[str, nn.Module] = {
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            "ln_f": _build_norm(config),
        }
        if config.positional_embedding == "learned":
            transformer_layers["wpe"] = nn.Embedding(config.context_len, config.n_embd)
        self.transformer = nn.ModuleDict(transformer_layers)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: share token embedding and output projection weights
        self.transformer["wte"].weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * n_layers) per GPT-2 paper
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()
        assert T <= self.config.context_len, (
            f"Sequence length {T} exceeds context_len {self.config.context_len}"
        )

        x = self.transformer["wte"](idx)
        if self.config.positional_embedding == "learned":
            pos = torch.arange(T, dtype=torch.long, device=idx.device)
            x = x + self.transformer["wpe"](pos)
        x = self.transformer["drop"](x)

        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        eos_token_id = 50256  # <|endoftext|>
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.context_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == eos_token_id:
                break
        return idx

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:
        decay = [p for _, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for _, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        use_fused = (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
            and device_type == "cuda"
        )
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=use_fused)
