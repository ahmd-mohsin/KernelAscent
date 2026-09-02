import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 179
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _fused_norms_kernel(
    X_ptr, RW_ptr, G3_ptr, B3_ptr, G4_ptr, B4_ptr, Out_ptr,
    N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    ptr = X_ptr + row * N + offs

    # ---- RMSNorm (fp32 math, round to fp16 like reference) ----
    xf = tl.load(ptr).to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    y = (xf * r).to(tl.float16)

    # * rms1_w  (fp16 elementwise: compute fp32, round fp16)
    rw = tl.load(RW_ptr + offs).to(tl.float32)
    y = (y.to(tl.float32) * rw).to(tl.float16)

    # * 1.4245 (fp16 elementwise)
    y = (y.to(tl.float32) * 1.4245).to(tl.float16)

    # ---- LayerNorm 3 (fp32 accumulation, fp16 output) ----
    yf = y.to(tl.float32)
    mean = tl.sum(yf, axis=0) / N
    d = yf - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g3 = tl.load(G3_ptr + offs).to(tl.float32)
    b3 = tl.load(B3_ptr + offs).to(tl.float32)
    y = (d * rstd * g3 + b3).to(tl.float16)

    # ---- LayerNorm 4 ----
    yf = y.to(tl.float32)
    mean = tl.sum(yf, axis=0) / N
    d = yf - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g4 = tl.load(G4_ptr + offs).to(tl.float32)
    b4 = tl.load(B4_ptr + offs).to(tl.float32)
    y = (d * rstd * g4 + b4).to(tl.float16)

    tl.store(Out_ptr + row * N + offs, y)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core matmul
        h = torch.matmul(x, self.W0)

        orig_shape = h.shape
        N = orig_shape[-1]
        h = h.contiguous().view(-1, N)
        rows = h.shape[0]
        out = torch.empty_like(h)

        _fused_norms_kernel[(rows,)](
            h, self.rms1_w, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b, out,
            N=N, num_warps=4,
        )
        return out.view(orig_shape)
