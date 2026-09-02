import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 200001
S, D, DT = 512, 512, torch.float16


@triton.jit
def _rope_qk_kernel(
    qkv_ptr, qo_ptr, ko_ptr, cos_ptr, sin_ptr,
    stride_qkv,
    HALF: tl.constexpr, E: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < HALF

    c = tl.load(cos_ptr + row * HALF + offs, mask=mask, other=0.0)
    s = tl.load(sin_ptr + row * HALF + offs, mask=mask, other=0.0)

    base = qkv_ptr + row * stride_qkv

    # q part
    q1 = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
    q2 = tl.load(base + HALF + offs, mask=mask, other=0.0).to(tl.float32)
    o_ty = qo_ptr.dtype.element_ty
    tl.store(qo_ptr + row * E + offs, (q1 * c - q2 * s).to(o_ty), mask=mask)
    tl.store(qo_ptr + row * E + HALF + offs, (q1 * s + q2 * c).to(o_ty), mask=mask)

    # k part
    k1 = tl.load(base + E + offs, mask=mask, other=0.0).to(tl.float32)
    k2 = tl.load(base + E + HALF + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(ko_ptr + row * E + offs, (k1 * c - k2 * s).to(o_ty), mask=mask)
    tl.store(ko_ptr + row * E + HALF + offs, (k1 * s + k2 * c).to(o_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_cache(self, seq_len, E, device):
        key = (seq_len, E, str(device))
        if getattr(self, "_cache_key", None) != key:
            half = E // 2
            pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
            freq = torch.exp(
                torch.arange(0, half, device=device, dtype=torch.float32)
                * (-math.log(10000.0) / max(half, 1))
            )
            ang = pos * freq
            self._cos = torch.cos(ang).contiguous()
            self._sin = torch.sin(ang).contiguous()
            self._Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._cache_key = key
        return self._cos, self._sin, self._Wqkv

    def forward(self, x):
        seq_len, E = x.shape
        cos, sin, Wqkv = self._get_cache(seq_len, E, x.device)

        # Single fused QKV projection
        qkv = x @ Wqkv  # (S, 3E)

        half = E // 2
        q = torch.empty((seq_len, E), device=x.device, dtype=x.dtype)
        k = torch.empty((seq_len, E), device=x.device, dtype=x.dtype)

        BLOCK = triton.next_power_of_2(half)
        _rope_qk_kernel[(seq_len,)](
            qkv, q, k, cos, sin,
            qkv.stride(0),
            HALF=half, E=E, BLOCK=BLOCK,
            num_warps=4,
        )

        v = qkv[:, 2 * E:]

        out = F.scaled_dot_product_attention(
            q.unsqueeze(0).unsqueeze(0),
            k.unsqueeze(0).unsqueeze(0),
            v.unsqueeze(0).unsqueeze(0),
            is_causal=True,
        )
        return out[0, 0]
