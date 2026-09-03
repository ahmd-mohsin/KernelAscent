import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 200019
S, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _rope_qk_kernel(
    qkv_ptr,          # (S, 3D) bf16, q at cols [0,D), k at cols [D,2D)
    cos_ptr, sin_ptr, # (S, HALF) fp32
    out_ptr,          # (2, S, D) bf16 -> out[0]=q_rope, out[1]=k_rope
    stride_row,       # row stride of qkv (= 3*D)
    SEQ,
    D_DIM: tl.constexpr,
    HALF: tl.constexpr,
):
    row = tl.program_id(0)
    which = tl.program_id(1)  # 0 -> q, 1 -> k

    offs = tl.arange(0, HALF)
    base = row * stride_row + which * D_DIM

    t1 = tl.load(qkv_ptr + base + offs).to(tl.float32)
    t2 = tl.load(qkv_ptr + base + HALF + offs).to(tl.float32)

    c = tl.load(cos_ptr + row * HALF + offs)
    s = tl.load(sin_ptr + row * HALF + offs)

    o1 = t1 * c - t2 * s
    o2 = t1 * s + t2 * c

    out_base = which * SEQ * D_DIM + row * D_DIM
    tl.store(out_ptr + out_base + offs, o1.to(out_ptr.dtype.element_ty))
    tl.store(out_ptr + out_base + HALF + offs, o2.to(out_ptr.dtype.element_ty))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_cache(self, x):
        dev = x.device
        seq, dim = x.shape
        key = (dev, seq, dim, x.dtype)
        cache = getattr(self, "_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1], cache[2], cache[3]
        # fused QKV weight
        Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
        # rope tables
        half = dim // 2
        pos = torch.arange(seq, device=dev, dtype=torch.float32).unsqueeze(1)
        freq = torch.exp(
            torch.arange(0, half, device=dev, dtype=torch.float32)
            * (-math.log(10000.0) / max(half, 1))
        )
        ang = pos * freq
        cos = torch.cos(ang).contiguous()
        sin = torch.sin(ang).contiguous()
        self._cache = (key, Wqkv, cos, sin)
        return Wqkv, cos, sin

    def forward(self, x):
        seq, dim = x.shape
        half = dim // 2
        Wqkv, cos, sin = self._get_cache(x)

        # single fused GEMM for q, k, v
        qkv = x @ Wqkv  # (S, 3D)
        v = qkv[:, 2 * dim:]

        # fused RoPE on q and k in one kernel launch
        qk = torch.empty((2, seq, dim), device=x.device, dtype=x.dtype)
        grid = (seq, 2)
        _rope_qk_kernel[grid](
            qkv, cos, sin, qk,
            qkv.stride(0), seq,
            D_DIM=dim, HALF=half,
            num_warps=4,
        )
        q = qk[0]
        k = qk[1]

        # fused causal flash attention (scale = 1/sqrt(dim) is the default)
        y = F.scaled_dot_product_attention(
            q.unsqueeze(0).unsqueeze(0),
            k.unsqueeze(0).unsqueeze(0),
            v.unsqueeze(0).unsqueeze(0),
            is_causal=True,
        )
        return y[0, 0]
