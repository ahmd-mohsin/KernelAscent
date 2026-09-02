import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 923
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _rms_gelu_softmax_kernel(
    X_ptr, W_ptr, Y_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (computed in fp32, matching reference)
    ms = tl.sum(x * x, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    xn = (x * r).to(tl.float16).to(tl.float32)  # .to(x.dtype) rounding

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    xw = (xn * w).to(tl.float16).to(tl.float32)  # fp16 mul result rounding

    # exact GELU (erf) computed in fp32, stored as fp16 (matches PyTorch half gelu)
    g = 0.5 * xw * (1.0 + tl.math.erf(xw * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # softmax in fp32 accumulation (matches torch.softmax on half)
    g = tl.where(mask, g, float("-inf"))
    mx = tl.max(g, axis=0)
    e = tl.exp(g - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.float16)

    tl.store(Y_ptr + row * N + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS half GEMM
        x = x.contiguous()
        rows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _rms_gelu_softmax_kernel[(rows,)](
            x, self.rms1_w, y,
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        y = y @ self.W4  # cuBLAS half GEMM
        y = F.gelu(y)
        return y
