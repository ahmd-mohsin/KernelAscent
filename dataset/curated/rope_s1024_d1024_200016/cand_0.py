import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 200016
S, D, DT = 1024, 1024, torch.float16


@triton.jit
def _rope_kernel(
    qk_ptr, cos_ptr, sin_ptr,
    row_stride, head_off,
    HALF: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    mask = offs < HALF

    base = qk_ptr + row * row_stride + head * head_off
    t1 = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
    t2 = tl.load(base + HALF + offs, mask=mask, other=0.0).to(tl.float32)

    c = tl.load(cos_ptr + row * HALF + offs, mask=mask, other=0.0)
    s = tl.load(sin_ptr + row * HALF + offs, mask=mask, other=0.0)

    o1 = t1 * c - t2 * s
    o2 = t1 * s + t2 * c

    tl.store(base + offs, o1.to(tl.float16), mask=mask)
    tl.store(base + HALF + offs, o2.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._cache_key = None
        self._W = None
        self._cos = None
        self._sin = None

    def _build_cache(self, s, d, device):
        # Fused projection weight: one GEMM instead of three
        self._W = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
        half = d // 2
        pos = torch.arange(s, device=device, dtype=torch.float32).unsqueeze(1)
        freq = torch.exp(
            torch.arange(0, half, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half, 1))
        )
        ang = pos * freq
        self._cos = torch.cos(ang).contiguous()
        self._sin = torch.sin(ang).contiguous()
        self._cache_key = (s, d, device)

    def forward(self, x):
        s, d = x.shape
        device = x.device
        if self._cache_key != (s, d, device):
            self._build_cache(s, d, device)

        # Single fused QKV projection (tensor-core GEMM)
        qkv = x @ self._W  # (s, 3d), contiguous

        half = d // 2
        BLOCK = triton.next_power_of_2(max(half, 1))
        # Apply RoPE in-place to q (head 0) and k (head 1); v untouched
        _rope_kernel[(s, 2)](
            qkv, self._cos, self._sin,
            qkv.stride(0), d,
            HALF=half, BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 512 else 4,
        )

        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)
        a = torch.softmax(scores, dim=-1)
        return a @ v
