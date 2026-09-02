import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 622
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_gelu_rms_softmax_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    # Load matmul output (bf16) and upcast to fp32
    x = tl.load(X_ptr + row * stride + offs).to(tl.float32)

    # Exact (erf-based) GELU in fp32, rounded back to bf16 to match torch
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMSNorm (fp32 reduction, eps=1e-6)
    ms = tl.sum(g * g, axis=0) / N
    rs = tl.math.rsqrt(ms + 1e-6)
    n = (g * rs).to(tl.bfloat16).to(tl.float32)

    # Multiply by rms weight (bf16 elementwise, fp32 opmath then round)
    w = tl.load(W_ptr + offs).to(tl.float32)
    v = (n * w).to(tl.bfloat16).to(tl.float32)

    # Softmax in fp32 (matches torch's fp32 accumulation for bf16 input)
    mx = tl.max(v, axis=0)
    e = tl.exp(v - mx)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16).to(tl.float32)

    # Final scale, rounded to bf16
    o = (p * 1.0258).to(tl.bfloat16)
    tl.store(Out_ptr + row * stride + offs, o)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 GEMM (fp32 accumulate)
        h = x @ self.W0
        h = h.contiguous()
        rows, cols = h.shape
        out = torch.empty_like(h)
        _fused_gelu_rms_softmax_kernel[(rows,)](
            h, self.rms2_w, out,
            cols, h.stride(0),
            BLOCK=cols,
            num_warps=8,
        )
        return out
