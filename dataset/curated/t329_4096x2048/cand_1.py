import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 329
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _rms_gelu_kernel(X_ptr, W_ptr, Y_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(X_ptr + row * N + offs).to(tl.float32)

    # RMS norm (fp32 accumulation, matches reference)
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + 1e-6)

    # (xf * rstd).to(fp16)  -- round to half like reference
    y = (x * rstd).to(tl.float16).to(tl.float32)

    # * rms1_w : fp16*fp16 computed in fp32 internally, rounded to fp16
    w = tl.load(W_ptr + offs).to(tl.float32)
    y = (y * w).to(tl.float16).to(tl.float32)

    # exact GELU (erf-based) computed in fp32 (PyTorch opmath for half), rounded to fp16
    g = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # scalar mul in fp32, final store rounds to fp16
    g = g * 1.203
    tl.store(Y_ptr + row * N + offs, g.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS)
        h = torch.mm(x, self.W0)
        h = h.contiguous()

        m, n = h.shape  # (4096, 512)
        out = torch.empty_like(h)
        _rms_gelu_kernel[(m,)](
            h, self.rms1_w, out,
            N=n, BLOCK=n,
            num_warps=4,
        )

        # GEMM 2 (cuBLAS) + in-place ReLU (exact)
        y = torch.mm(out, self.W4)
        return torch.relu_(y)
