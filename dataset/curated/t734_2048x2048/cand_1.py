import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _ln_relu_kernel(X, G, B, Y, N, eps,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    y = tl.maximum(y, 0.0)
    tl.store(Y + row * N + cols, y.to(Y.dtype.element_ty), mask=mask)


SEED = 734
M, D, DT = 2048, 2048, torch.bfloat16


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        rows, N = h.shape[0] * (h.numel() // (h.shape[-1] * h.shape[0])) if h.dim() > 2 else h.shape[0], h.shape[-1]
        h2d = h.view(-1, N)
        rows = h2d.shape[0]

        out = torch.empty_like(h2d)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _ln_relu_kernel[(rows,)](
            h2d, self.ln1_g, self.ln1_b, out,
            N, 1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )

        # GEMM 2 (cuBLAS tensor cores)
        y = torch.matmul(out.view_as(h), self.W3)
        return y
