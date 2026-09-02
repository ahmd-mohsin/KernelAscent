import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 200000
S, D, DT = 512, 512, torch.float16


@triton.jit
def _rope_qk_kernel(
    qkv_ptr, cos_ptr, sin_ptr,
    stride_row,
    E: tl.constexpr,
    HALF: tl.constexpr,
):
    row = tl.program_id(0)
    which = tl.program_id(1)  # 0 -> q, 1 -> k
    cols = tl.arange(0, HALF)
    base = qkv_ptr + row * stride_row + which * E
    t1 = tl.load(base + cols).to(tl.float32)
    t2 = tl.load(base + HALF + cols).to(tl.float32)
    c = tl.load(cos_ptr + row * HALF + cols)
    s = tl.load(sin_ptr + row * HALF + cols)
    o1 = t1 * c - t2 * s
    o2 = t1 * s + t2 * c
    tl.store(base + cols, o1.to(tl.float16))
    tl.store(base + HALF + cols, o2.to(tl.float16))


def _rope_ref(t):
    S_, E = t.shape
    half = E // 2
    pos = torch.arange(S_, device=t.device, dtype=torch.float32).unsqueeze(1)
    freq = torch.exp(torch.arange(0, half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / max(half, 1)))
    ang = pos * freq
    cos, sin = torch.cos(ang), torch.sin(ang)
    t1 = t[..., :half].float(); t2 = t[..., half:half * 2].float()
    out = t.float().clone()
    out[..., :half] = t1 * cos - t2 * sin
    out[..., half:half * 2] = t1 * sin + t2 * cos
    return out.to(t.dtype)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._cache = {}

    def _get_cache(self, seq_len, device):
        key = (seq_len, device)
        c = self._cache.get(key)
        if c is None:
            W = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            E = self.Wq.shape[1]
            half = E // 2
            pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
            freq = torch.exp(torch.arange(0, half, device=device, dtype=torch.float32)
                             * (-math.log(10000.0) / max(half, 1)))
            ang = pos * freq
            cos = torch.cos(ang).contiguous()
            sin = torch.sin(ang).contiguous()
            c = (W, cos, sin)
            self._cache[key] = c
        return c

    def forward(self, x):
        if not x.is_cuda:
            q = _rope_ref(x @ self.Wq); k = _rope_ref(x @ self.Wk); v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            a = torch.softmax(scores, dim=-1)
            return a @ v

        seq_len, E = x.shape
        half = E // 2
        W, cos, sin = self._get_cache(seq_len, x.device)

        # Single fused projection: (S, 3E)
        qkv = x @ W

        # Apply RoPE in-place on q and k slices of qkv via Triton
        _rope_qk_kernel[(seq_len, 2)](
            qkv, cos, sin,
            qkv.stride(0),
            E=E, HALF=half,
            num_warps=4,
        )

        q = qkv[:, :E].unsqueeze(0).unsqueeze(0)
        k = qkv[:, E:2 * E].unsqueeze(0).unsqueeze(0)
        v = qkv[:, 2 * E:].unsqueeze(0).unsqueeze(0)

        out = F.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(E))
        return out.squeeze(0).squeeze(0)
