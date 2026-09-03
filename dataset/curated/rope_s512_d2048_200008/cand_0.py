import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 200008
S, D, DT = 512, 2048, torch.float16


@triton.jit
def _rope_kernel(ptr, cos_ptr, sin_ptr, stride,
                 D: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr):
    # grid: (S, 2)  -> program (row, chunk) applies RoPE in-place to
    # columns [chunk*D, chunk*D + 2*HALF) of row `row`
    row = tl.program_id(0)
    c = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    mask = offs < HALF
    base = row * stride + c * D
    x1 = tl.load(ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(ptr + base + HALF + offs, mask=mask, other=0.0).to(tl.float32)
    cs = tl.load(cos_ptr + row * HALF + offs, mask=mask, other=0.0)
    sn = tl.load(sin_ptr + row * HALF + offs, mask=mask, other=0.0)
    o1 = x1 * cs - x2 * sn
    o2 = x1 * sn + x2 * cs
    tl.store(ptr + base + offs, o1.to(ptr.dtype.element_ty), mask=mask)
    tl.store(ptr + base + HALF + offs, o2.to(ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        S_, D_ = x.shape
        dev = x.device
        half = D_ // 2

        # Cache fused QKV weight (one big GEMM instead of three)
        W = getattr(self, "_W_fused", None)
        if W is None or W.device != dev or W.dtype != x.dtype:
            W = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._W_fused = W

        # Cache RoPE cos/sin tables (fp32, matching reference math)
        key = (S_, half, str(dev))
        if getattr(self, "_cs_key", None) != key:
            pos = torch.arange(S_, device=dev, dtype=torch.float32).unsqueeze(1)
            freq = torch.exp(
                torch.arange(0, half, device=dev, dtype=torch.float32)
                * (-math.log(10000.0) / max(half, 1))
            )
            ang = pos * freq
            self._cos = torch.cos(ang).contiguous()
            self._sin = torch.sin(ang).contiguous()
            self._cs_key = key

        # Single fused projection: (S, 3D)
        qkv = x @ W

        # Fused in-place RoPE on the q and k slices (fp32 math, fp16 store,
        # identical rounding to the reference implementation)
        BLOCK = triton.next_power_of_2(max(half, 1))
        _rope_kernel[(S_, 2)](
            qkv, self._cos, self._sin, qkv.stride(0),
            D=D_, HALF=half, BLOCK=BLOCK,
            num_warps=8,
        )

        q = qkv[:, :D_]
        k = qkv[:, D_:2 * D_]
        v = qkv[:, 2 * D_:]

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(D_)
        a = torch.softmax(scores, dim=-1)
        return a @ v
