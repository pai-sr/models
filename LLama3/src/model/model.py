import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from base.base_model import BaseModel

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim ))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :] # (B, seq_len, n_kv_heads, 1, head_size)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(BaseModel):
    def __init__(self, dim=4096, n_layers=32, n_heads=32, n_kv_heads=None,
                 vocab_size=-1, multiple_of=256, ffn_dim_multiplier=None,
                 norm_eps=1e-5, rope_theta=500000, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len

        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = self.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = self.dim // self.n_heads

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            mask = torch.full((1, 1, self.max_seq_len, self.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(self,
                x: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor
                ):
        bsz, seqlen, _ = x.shape # (batch, seqlen, dim)

        xq = self.wq(x) # (bs, seqlen, dim) -> (bs, seqlen, n_q_heads * head_size)
        xk = self.wk(x) # (bs, seqlen, dim) -> (bs, seqlen, n_kv_heads * head_size)
        xv = self.wv(x) # (bs, seqlen, dim) -> (bs, seqlen, n_kv_heads * head_size)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        xk = repeat_kv(xk, self.n_rep) # (bs, cache_len + seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep) # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2) # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        xv = xv.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)

        if self.flash:
            output = nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None,
                                                                dropout_p=0.0 if self.training else 0.0,
                                                                is_causal=True)
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, "mask")
            scores = scores + self.mask[:, :, :seqlen, :seqlen] # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv) # (bs, n_local_heads, seqlen, head_dim)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        output = self.wo(output)
        return output

class FeedForward(BaseModel):
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 multiple_of: int,
                 ffn_dim_multiplier: Optional[float],
                 ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(BaseModel):
    def __init__(self, layer_id, dim=4096, n_layers=32, n_heads=32, n_kv_heads=None,
                 vocab_size=-1, multiple_of=256, ffn_dim_multiplier=None,
                 norm_eps=1e-5, rope_theta=500000, max_seq_len=2048):
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = self.dim // self.n_heads
        super().__init__()

        self.attention = Attention(dim, n_layers, n_heads,
                                   n_kv_heads, vocab_size, multiple_of,
                                   ffn_dim_multiplier, norm_eps, rope_theta,
                                   max_seq_len)
        self.feed_forward = FeedForward(dim,
                                        hidden_dim=4 * dim,
                                        multiple_of=multiple_of,
                                        ffn_dim_multiplier=ffn_dim_multiplier
                                        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)

    def forward(self,
                x,
                freqs_cos,
                freqs_sin
                ):
        # (B, seqlen, dim) + (B, seqlen, dim) -> (B, seqlen, dim)
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(BaseModel):
    def __init__(self, dim=4096, n_layers=32, n_heads=32, n_kv_heads=None,
                 vocab_size=-1, multiple_of=256, ffn_dim_multiplier=None,
                 norm_eps=1e-5, rope_theta=500000, max_seq_len=2048):
        self.dim = dim
        self.norm_eps = norm_eps
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        super().__init__()

        self.tok_embeddings = nn.Embedding(self.vocab_size, self.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(n_layers):
            self.layers.append(TransformerBlock(layer_id, dim, n_layers, n_heads,
                                   n_kv_heads, vocab_size, multiple_of,
                                   ffn_dim_multiplier, norm_eps, rope_theta,
                                   max_seq_len))
        self.norm = RMSNorm(self.dim, self.norm_eps)
        self.output = nn.Linear(self.dim, self.vocab_size, bias=False)

        freqs_cos, freqs_sin = precompute_freqs_cis(self.dim // self.n_heads, self.max_seq_len, self.rope_theta)
        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0, std=0.02/math.sqrt(2 * self.n_layers))

        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.inference_mode()
    def forward(self, tokens, targets=None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens) # (bs, seq_len) -> (bs, seq_len, dim)

        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin) # (bs, seqlen, dim)
        h = self.norm(h) # (bs, seqlen, dim)

        if targets is not None:
            logits = self.output(h).float()
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                             targets.view(-1), ignore_index=-1)
        else:
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        # (bs, seqlen, vocab_size)
        return logits

    @torch.inference_mode()
    def generate(self, tokens, max_new_tokens, temperature=1.0, top_k=None, eos=None):
        for _ in range(max_new_tokens):
            token_cond = tokens if tokens.size(1) <= self.max_new_tokens else tokens[:, -self.max_seq_len:]
            logits = self(token_cond)
            logits = logits[:, -1, :]
            if temperature == 0.0:
                _, next_token = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
            if next_token == eos:
                break

        return tokens

def print_model_parameters(model):
    param_sum = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_sum += param.numel()
            print(f"Layer: {name}, Parameters: {param.numel()}")
    print(f"Total of parameters: {param_sum}")

