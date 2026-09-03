import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 200006
S, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _rope_kernel(
    T_ptr, OUT_ptr, COS_ptr, SIN_ptr,
    stride_t, stride_o,
    HALF: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < HALF

    t1 = tl.load(T_ptr + row * stride_t + offs, mask=mask, other=0.0).to(tl.float32)
    t2 = tl.load(T_ptr + row * stride_t + HALF + offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(COS_ptr + row * HALF + offs, mask=mask, other=0.0)
    s = tl.load(SIN_ptr + row * HALF + offs, mask=mask, other=0.0)

    o1 = t1 * c - t2 * s
    o2 = t1 * s + t2 * c

    tl.store(OUT_ptr + row * stride_o + offs, o1.to(OUT_ptr.dtype.element_ty), mask=mask)
    tl.store(OUT_ptr + row * stride_o + HALF + offs, o2.to(OUT_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._cache = {}

    def _get_cache(self, x):
        key = (x.device, x.shape[0], x.dtype)
        c = self._cache.get(key)
        if c is None:
            device = x.device
            d = self.Wq.shape[0]
            # Fused QKV weight
            Wqkv = torch.cat([self.Wq.to(device), self.Wk.to(device), self.Wv.to(device)], dim=1).contiguous()
            # Precompute RoPE tables in fp32 (matches reference math)
            seq = x.shape[0]
            half = d // 2
            pos = torch.arange(seq, device=device, dtype=torch.float32).unsqueeze(1)
            freq = torch.exp(
                torch.arange(0, half, device=device, dtype=torch.float32)
                * (-math.log(10000.0) / max(half, 1))
            )
            ang = pos * freq
            cos = torch.cos(ang).contiguous()
            sin = torch.sin(ang).contiguous()
            c = (Wqkv, cos, sin)
            self._cache[key] = c
        return c

    def forward(self, x):
        seq, d = x.shape
        half = d // 2
        Wqkv, cos, sin = self._get_cache(x)

        # Single fused GEMM for Q, K, V projections
        qkv = x @ Wqkv  # (S, 3D)
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        if x.is_cuda:
            BLOCK = triton.next_power_of_2(half)
            grid = (seq,)
            # In-place RoPE on the q and k slices of the fused buffer
            _rope_kernel[grid](q, q, cos, sin, qkv.stride(0), qkv.stride(0),
                               HALF=half, BLOCK=BLOCK)
            _rope_kernel[grid](k, k, cos, sin, qkv.stride(0), qkv.stride(0),
                               HALF=half, BLOCK=BLOCK)
            qr, kr = q, k
        else:
            def rot(t):
                t1 = t[:, :half].float()
                t2 = t[:, half:half * 2].float()
                return torch.cat([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=1).to(t.dtype)
            qr, kr = rot(q), rot(k)

        scores = (qr @ kr.transpose(-1, -2)) / math.sqrt(d)
        a = torch.softmax(scores, dim=-1)
        return a @ v
