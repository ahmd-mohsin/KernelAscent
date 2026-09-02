import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 101
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _relu_rmsnorm_kernel(X, W, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X + row * D + offs
    x = tl.load(ptr).to(tl.float32)
    x = tl.maximum(x, 0.0)                      # fused ReLU
    ms = tl.sum(x * x, axis=0) / D              # mean of squares (fp32)
    rs = tl.rsqrt(ms + 1e-6)
    w = tl.load(W + offs)                       # bf16 weight
    y = (x * rs).to(tl.bfloat16) * w            # match: cast to bf16 then * w
    tl.store(Y + row * D + offs, y)


@triton.jit
def _relu_scale_kernel(X, n, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    y = tl.maximum(x, 0.0) * scale
    tl.store(X + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        rows, d = h.shape

        # Fused ReLU + RMSNorm (fp32 accumulation) + weight scale
        y = torch.empty_like(h)
        _relu_rmsnorm_kernel[(rows,)](
            h, self.rms2_w, y,
            D=d, BLOCK=d,
            num_warps=8,
        )

        # GEMM 2 (cuBLAS tensor cores)
        out = y @ self.W3

        # Fused ReLU + scale (in-place)
        n = out.numel()
        BLOCK = 4096
        _relu_scale_kernel[(triton.cdiv(n, BLOCK),)](
            out, n, 1.0805,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
