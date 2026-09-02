import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 222
M, D, DT = 512, 1025, torch.float16


@triton.jit
def _gelu_rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), computed in fp32 then cast back to fp16
    # to match PyTorch's opmath behavior for half tensors
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # RMSNorm: upcast fp16 gelu output to fp32 (matches x.float())
    gf = g16.to(tl.float32)
    ms = tl.sum(gf * gf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    z16 = (gf * r).to(tl.float16)

    # final elementwise multiply with weight (fp16 * fp16 -> fp32 opmath -> fp16)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (z16.to(tl.float32) * w).to(tl.float16)

    tl.store(Y_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 4096, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _gelu_rmsnorm_kernel[(Mrows,)](
            h, self.rms2_w, y, N,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
