import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 731
M, D, DT = 512, 512, torch.float16


@triton.jit
def _softmax_bias_gelu_kernel(
    X_ptr, B_ptr, Y_ptr,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # Load row, compute softmax in fp32 (matches PyTorch half softmax accumulation)
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=-float('inf')).to(tl.float32)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    s = tl.sum(e, 0)
    p = e / s

    # softmax output >= 0, so relu is identity.
    # Cast to fp16 (as PyTorch produces fp16 softmax output)
    p16 = p.to(tl.float16)

    # Bias add in fp16 semantics (exact: fp32 add of two fp16 values, round to fp16)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    t = (p16.to(tl.float32) + b.to(tl.float32)).to(tl.float16)

    # Exact GELU (erf form) computed in fp32, matching PyTorch's opmath for half
    tf = t.to(tl.float32)
    g = 0.5 * tf * (1.0 + tl.math.erf(tf * 0.7071067811865476))

    tl.store(Y_ptr + row * N + offs, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W5 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape

        y = torch.empty_like(h)
        # Fused softmax + relu(no-op) + bias + gelu in one kernel
        _softmax_bias_gelu_kernel[(Mrows,)](
            h, self.b3, y,
            N,
            BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )

        # GEMM 2 (cuBLAS tensor cores)
        return y @ self.W5
