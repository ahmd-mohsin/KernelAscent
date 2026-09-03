import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 200012
S, D, DT = 1024, 512, torch.float16


@triton.jit
def _rope_qk_kernel(qkv_ptr, cos_ptr, sin_ptr, stride, HALF: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, HALF)
    c = tl.load(cos_ptr + row * HALF + offs)
    s = tl.load(sin_ptr + row * HALF + offs)
    base = qkv_ptr + row * stride
    # rotate q (columns [0, 2*HALF))
    q1 = tl.load(base + offs).to(tl.float32)
    q2 = tl.load(base + HALF + offs).to(tl.float32)
    tl.store(base + offs, q1 * c - q2 * s)
    tl.store(base + HALF + offs, q1 * s + q2 * c)
    # rotate k (columns [2*HALF, 4*HALF))
    k1 = tl.load(base + 2 * HALF + offs).to(tl.float32)
    k2 = tl.load(base + 3 * HALF + offs).to(tl.float32)
    tl.store(base + 2 * HALF + offs, k1 * c - k2 * s)
    tl.store(base + 3 * HALF + offs, k1 * s + k2 * c)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _prepare_cache(self, x):
        dev = x.device
        seq = x.shape[0]
        key = (dev, seq, x.dtype)
        if getattr(self, "_cache_key", None) == key:
            return
        # fused QKV weight (columns are independent -> identical results)
        self._Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
        half = self.Wq.shape[1] // 2
        pos = torch.arange(seq, device=dev, dtype=torch.float32).unsqueeze(1)
        freq = torch.exp(
            torch.arange(0, half, device=dev, dtype=torch.float32)
            * (-math.log(10000.0) / max(half, 1))
        )
        ang = pos * freq
        self._cos = torch.cos(ang).contiguous()
        self._sin = torch.sin(ang).contiguous()
        self._half = half
        self._cache_key = key

    def forward(self, x):
        self._prepare_cache(x)
        seq, d = x.shape
        half = self._half

        # single fused GEMM for q, k, v
        qkv = x @ self._Wqkv  # (S, 3D), contiguous

        if x.is_cuda and (half & (half - 1)) == 0:
            _rope_qk_kernel[(seq,)](
                qkv, self._cos, self._sin, qkv.stride(0), HALF=half,
                num_warps=4,
            )
            q = qkv[:, :d]
            k = qkv[:, d:2 * d]
            v = qkv[:, 2 * d:]
        else:
            q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
            c, s = self._cos, self._sin
            for t in (q, k):
                t1 = t[:, :half].float()
                t2 = t[:, half:2 * half].float()
                t[:, :half] = (t1 * c - t2 * s).to(t.dtype)
                t[:, half:2 * half] = (t1 * s + t2 * c).to(t.dtype)

        # fused flash attention (fp32 accumulation, non-causal)
        out = F.scaled_dot_product_attention(
            q.unsqueeze(0).unsqueeze(0),
            k.unsqueeze(0).unsqueeze(0),
            v.unsqueeze(0).unsqueeze(0),
            scale=1.0 / math.sqrt(d),
        )
        return out[0, 0]
