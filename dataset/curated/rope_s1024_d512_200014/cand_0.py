import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 200014
S, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _rope_qk_kernel(
    QKV,            # (S, 3*D) bf16, q at cols [0,D), k at cols [D,2D)
    COS, SIN,       # (S, HALF) fp32
    stride_row,
    scale,
    HALF: tl.constexpr,
):
    row = tl.program_id(0)
    which = tl.program_id(1)  # 0 -> q, 1 -> k

    offs = tl.arange(0, HALF)
    base = QKV + row * stride_row + which * 2 * HALF

    t1 = tl.load(base + offs).to(tl.float32)
    t2 = tl.load(base + HALF + offs).to(tl.float32)
    c = tl.load(COS + row * HALF + offs)
    s = tl.load(SIN + row * HALF + offs)

    o1 = t1 * c - t2 * s
    o2 = t1 * s + t2 * c

    # fold the 1/sqrt(d) attention scale into q only
    sc = tl.where(which == 0, scale, 1.0)
    o1 = o1 * sc
    o2 = o2 * sc

    tl.store(base + offs, o1.to(QKV.dtype.element_ty))
    tl.store(base + HALF + offs, o2.to(QKV.dtype.element_ty))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

        self._W_fused = None
        self._cos = None
        self._sin = None
        self._cache_key = None

    def _build_cache(self, x):
        device = x.device
        seq, dim = x.shape
        half = dim // 2
        # fused projection weight (D, 3D)
        self._W_fused = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous().to(device)
        pos = torch.arange(seq, device=device, dtype=torch.float32).unsqueeze(1)
        freq = torch.exp(
            torch.arange(0, half, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half, 1))
        )
        ang = pos * freq
        self._cos = torch.cos(ang).contiguous()
        self._sin = torch.sin(ang).contiguous()
        self._cache_key = (device, seq, dim)

    def forward(self, x):
        seq, dim = x.shape
        if self._cache_key != (x.device, seq, dim):
            self._build_cache(x)

        half = dim // 2
        scale = 1.0 / math.sqrt(dim)

        # single fused projection: (S, 3D)
        qkv = x @ self._W_fused

        # in-place RoPE on q and k halves (q additionally scaled by 1/sqrt(d))
        _rope_qk_kernel[(seq, 2)](
            qkv, self._cos, self._sin,
            qkv.stride(0),
            scale,
            HALF=half,
            num_warps=4,
        )

        q = qkv[:, :dim]
        k = qkv[:, dim:2 * dim]
        v = qkv[:, 2 * dim:]

        scores = q @ k.transpose(-1, -2)
        a = torch.softmax(scores, dim=-1)
        return a @ v
