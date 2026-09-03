import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 200015
S, D, DT = 1024, 512, torch.bfloat16


def _rope(t):
    S, E = t.shape
    half = E // 2
    pos = torch.arange(S, device=t.device, dtype=torch.float32).unsqueeze(1)
    freq = torch.exp(torch.arange(0, half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / max(half, 1)))
    ang = pos * freq
    cos, sin = torch.cos(ang), torch.sin(ang)
    t1 = t[..., :half].float(); t2 = t[..., half:half * 2].float()
    out = t.float().clone()
    out[..., :half] = t1 * cos - t2 * sin
    out[..., half:half * 2] = t1 * sin + t2 * cos
    return out.to(t.dtype)


@triton.jit
def _rope_qk_kernel(
    qkv_ptr,            # (S, 3*E) contiguous
    cos_ptr, sin_ptr,   # (S, HALF) fp32 contiguous
    E: tl.constexpr,
    HALF: tl.constexpr,
):
    row = tl.program_id(0)
    which = tl.program_id(1)  # 0 -> q, 1 -> k
    offs = tl.arange(0, HALF)

    c = tl.load(cos_ptr + row * HALF + offs)
    s = tl.load(sin_ptr + row * HALF + offs)

    base = qkv_ptr + row * (3 * E) + which * E
    t1 = tl.load(base + offs).to(tl.float32)
    t2 = tl.load(base + HALF + offs).to(tl.float32)

    o1 = t1 * c - t2 * s
    o2 = t1 * s + t2 * c

    tl.store(base + offs, o1.to(base.dtype.element_ty))
    tl.store(base + HALF + offs, o2.to(base.dtype.element_ty))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _build_cache(self, x):
        device = x.device
        E = self.Wq.shape[1]
        half = E // 2
        # Fused projection weight: one matmul for q, k, v
        Wall = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
        Slen = x.shape[0]
        pos = torch.arange(Slen, device=device, dtype=torch.float32).unsqueeze(1)
        freq = torch.exp(
            torch.arange(0, half, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half, 1))
        )
        ang = pos * freq
        self._Wall = Wall
        self._cos = torch.cos(ang).contiguous()
        self._sin = torch.sin(ang).contiguous()
        self._cache_key = (device, x.shape[0], x.dtype)

    def forward(self, x):
        if not x.is_cuda:
            # Fallback: reference path on CPU
            q = _rope(x @ self.Wq); k = _rope(x @ self.Wk); v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
            a = torch.softmax(scores, dim=-1)
            return a @ v

        key = (x.device, x.shape[0], x.dtype)
        if getattr(self, "_cache_key", None) != key:
            self._build_cache(x)

        Slen, E = x.shape
        half = E // 2

        # Single fused projection: (S, 3E)
        qkv = x @ self._Wall

        # In-place RoPE on q and k halves of qkv
        grid = (Slen, 2)
        _rope_qk_kernel[grid](qkv, self._cos, self._sin, E, half, num_warps=4)

        q = qkv[:, :E]
        k = qkv[:, E:2 * E]
        v = qkv[:, 2 * E:]

        out = F.scaled_dot_product_attention(
            q.unsqueeze(0).unsqueeze(0),
            k.unsqueeze(0).unsqueeze(0),
            v.unsqueeze(0).unsqueeze(0),
            is_causal=True,
        )
        return out[0, 0]
