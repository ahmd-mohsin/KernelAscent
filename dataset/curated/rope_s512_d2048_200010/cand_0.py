import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 200010
S, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _rope_inplace_kernel(
    ptr, cos_ptr, sin_ptr,
    row_stride,
    HALF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    which = tl.program_id(1)  # 0 -> q slice, 1 -> k slice
    offs = tl.arange(0, BLOCK)
    mask = offs < HALF

    base = row * row_stride + which * (2 * HALF)

    t1 = tl.load(ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    t2 = tl.load(ptr + base + HALF + offs, mask=mask, other=0.0).to(tl.float32)

    c = tl.load(cos_ptr + row * HALF + offs, mask=mask, other=0.0)
    s = tl.load(sin_ptr + row * HALF + offs, mask=mask, other=0.0)

    o1 = t1 * c - t2 * s
    o2 = t1 * s + t2 * c

    tl.store(ptr + base + offs, o1.to(tl.bfloat16), mask=mask)
    tl.store(ptr + base + HALF + offs, o2.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_cache(self, x):
        d = x.shape[-1]
        s = x.shape[0]
        cache = getattr(self, "_cache", None)
        if (cache is None or cache["dev"] != x.device or cache["S"] != s
                or cache["D"] != d):
            # Fused QKV weight (single GEMM instead of three)
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous().to(x.device)
            half = d // 2
            pos = torch.arange(s, device=x.device, dtype=torch.float32).unsqueeze(1)
            freq = torch.exp(
                torch.arange(0, half, device=x.device, dtype=torch.float32)
                * (-math.log(10000.0) / max(half, 1))
            )
            ang = pos * freq
            cache = {
                "dev": x.device,
                "S": s,
                "D": d,
                "Wqkv": Wqkv,
                "cos": torch.cos(ang).contiguous(),
                "sin": torch.sin(ang).contiguous(),
            }
            self._cache = cache
        return cache

    def forward(self, x):
        cache = self._get_cache(x)
        d = x.shape[-1]
        s = x.shape[0]
        half = d // 2

        # Single fused GEMM for Q, K, V
        qkv = x @ cache["Wqkv"]  # (S, 3D), contiguous

        # Fused in-place RoPE on the Q and K slices (one Triton launch)
        BLOCK = triton.next_power_of_2(half)
        _rope_inplace_kernel[(s, 2)](
            qkv, cache["cos"], cache["sin"],
            qkv.stride(0),
            HALF=half,
            BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 1024 else 4,
        )

        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)
        a = torch.softmax(scores, dim=-1)
        return a @ v
