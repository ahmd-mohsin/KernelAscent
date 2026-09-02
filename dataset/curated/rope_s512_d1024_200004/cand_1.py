import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 200004
S, D, DT = 512, 1024, torch.float16


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


@triton.jit
def _rope_kernel(
    qkv_ptr, cos_ptr, sin_ptr,
    stride_row,
    HALF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)   # sequence position
    part = tl.program_id(1)  # 0 -> q, 1 -> k
    offs = tl.arange(0, BLOCK)
    mask = offs < HALF

    base = qkv_ptr + row * stride_row + part * (2 * HALF)
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

    def _get_cache(self, x):
        d = self.Wq.shape[1]
        s = x.shape[0]
        key = (s, x.device, x.dtype)
        if getattr(self, "_cache_key", None) != key:
            # fused QKV weight
            Wcat = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            # rope tables (fp32, matching reference math)
            half = d // 2
            pos = torch.arange(s, device=x.device, dtype=torch.float32).unsqueeze(1)
            freq = torch.exp(
                torch.arange(0, half, device=x.device, dtype=torch.float32)
                * (-math.log(10000.0) / max(half, 1))
            )
            ang = pos * freq
            self._Wcat = Wcat
            self._cos = torch.cos(ang).contiguous()
            self._sin = torch.sin(ang).contiguous()
            self._cache_key = key
        return self._Wcat, self._cos, self._sin

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            # fallback: reference path
            q = _rope_ref(x @ self.Wq); k = _rope_ref(x @ self.Wk); v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            a = torch.softmax(scores, dim=-1)
            return a @ v

        Wcat, cos, sin = self._get_cache(x)
        d = self.Wq.shape[1]
        s = x.shape[0]
        half = d // 2

        # single fused QKV GEMM
        qkv = x @ Wcat  # (s, 3d), contiguous

        # fused in-place RoPE on q and k
        BLOCK = triton.next_power_of_2(half)
        _rope_kernel[(s, 2)](
            qkv, cos, sin,
            qkv.stride(0),
            HALF=half,
            BLOCK=BLOCK,
            num_warps=4,
        )

        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        scores = q @ k.transpose(-1, -2)
        scores *= (1.0 / math.sqrt(d))  # exact: sqrt(1024)=32 is a power of two
        a = torch.softmax(scores, dim=-1)
        return a @ v
