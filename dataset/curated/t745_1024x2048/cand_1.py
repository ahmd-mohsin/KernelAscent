import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 745
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _gelu_ln_kernel(
    X_ptr, G_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # exact (erf) GELU, computed in fp32 then rounded to fp16 to match
    # PyTorch's elementwise half op (opmath float, half output)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # LayerNorm in fp32 accumulation (matches PyTorch half layer_norm)
    mean = tl.sum(g, axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    y = diff * rstd * w + b
    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core GEMM
        h = x @ self.W0
        h = h.contiguous()
        M_, N_ = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N_)
        _gelu_ln_kernel[(M_,)](
            h, self.ln2_g, self.ln2_b, out,
            N_, h.stride(0), out.stride(0), 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
