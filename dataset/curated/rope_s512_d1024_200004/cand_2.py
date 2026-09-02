import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 200004
S, D, DT = 512, 1024, torch.float16


@triton.jit
def _rope_qk_kernel(
    qkv_ptr, cos_ptr, sin_ptr,
    stride_row,
    HALF: tl.constexpr,
):
    row = tl.program_id(0)
    which = tl.program_id(1)  # 0 -> q, 1 -> k

    offs = tl.arange(0, HALF)
    base = qkv_ptr + row * stride_row + which * (2 * HALF)

    t1 = tl.load(base + offs).to(tl.float32)
    t2 = tl.load(base + HALF + offs).to(tl.float32)

    c = tl.load(cos_ptr + row * HALF + offs)
    s = tl.load(sin_ptr + row * HALF + offs)

    o1 = t1 * c - t2 * s
    o2 = t1 * s + t2 * c

    tl.store(base + offs, o1.to(tl.float16))
    tl.store(base + HALF + offs, o2.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None
        self._cos = None
        self._sin = None
        self._cache_key = None

    def _build_cache(self, device, seq_len, dim):
        # Fused projection weight
        self._Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
        # RoPE tables (float32, matching reference math)
        half = dim // 2
        pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        freq = torch.exp(
            torch.arange(0, half, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half, 1))
        )
        ang = pos * freq
        self._cos = torch.cos(ang).contiguous()
        self._sin = torch.sin(ang).contiguous()
        self._cache_key = (device, seq_len, dim)

    def forward(self, x):
        seq_len, dim = x.shape
        device = x.device
        if self._cache_key != (device, seq_len, dim) or self._Wqkv is None \
                or self._Wqkv.device != device:
            self._build_cache(device, seq_len, dim)

        # Single fused GEMM for q, k, v projections
        qkv = torch.matmul(x, self._Wqkv)  # [S, 3D], contiguous

        half = dim // 2
        # In-place RoPE on q and k slices
        grid = (seq_len, 2)
        _rope_qk_kernel[grid](
            qkv, self._cos, self._sin,
            qkv.stride(0),
            HALF=half,
            num_warps=4,
        )

        q = qkv[:, :dim]
        k = qkv[:, dim:2 * dim]
        v = qkv[:, 2 * dim:]

        # scale = 1/32 is an exact power of two -> identical to division
        scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(dim))
        a = torch.softmax(scores, dim=-1)
        return torch.matmul(a, v)
