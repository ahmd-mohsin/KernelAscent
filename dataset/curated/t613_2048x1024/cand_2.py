import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 613
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    Y_ptr, B1_ptr, RW_ptr, B4_ptr, G5_ptr, BE5_ptr, OUT_ptr,
    stride_y, stride_o,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    # load matmul result (already rounded to bf16), upcast to fp32
    y = tl.load(Y_ptr + row * stride_y + cols).to(tl.float32)

    # x = x + b1  (bf16 add: fp32 compute, round to bf16)
    b1 = tl.load(B1_ptr + cols).to(tl.float32)
    t = (y + b1).to(tl.bfloat16).to(tl.float32)

    # RMSNorm: fp32 reduction, scale in fp32, round to bf16, then bf16 mul by weight
    ms = tl.sum(t * t, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    t = (t * rrms).to(tl.bfloat16).to(tl.float32)
    rw = tl.load(RW_ptr + cols).to(tl.float32)
    t = (t * rw).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf), fp32 opmath, round to bf16
    g = 0.5 * t * (1.0 + tl.math.erf(t * 0.7071067811865476))
    t = g.to(tl.bfloat16).to(tl.float32)

    # x = x + b4
    b4 = tl.load(B4_ptr + cols).to(tl.float32)
    t = (t + b4).to(tl.bfloat16).to(tl.float32)

    # LayerNorm: fp32 stats, affine in fp32, round output to bf16
    mean = tl.sum(t, axis=0) / N
    d = t - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    gg = tl.load(G5_ptr + cols).to(tl.float32)
    bb = tl.load(BE5_ptr + cols).to(tl.float32)
    o = d * rstd * gg + bb

    tl.store(OUT_ptr + row * stride_o + cols, o.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (fp32 accumulate, single round to bf16 — same as reference)
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        _fused_post_kernel[(m,)](
            y, self.b1, self.rms2_w, self.b4, self.ln5_g, self.ln5_b, out,
            y.stride(0), out.stride(0),
            N=n, BLOCK=512,
            num_warps=4,
        )
        return out
