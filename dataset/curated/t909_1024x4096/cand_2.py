import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 909
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _ln_softmax_gelu_kernel(
    X_ptr, G_ptr, B_ptr, Y_ptr,
    N, stride_row,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, matching PyTorch fp16 layer_norm)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b

    # Softmax (fp32)
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    s = tl.sum(e, axis=0)
    p = e / s

    # GELU (erf-based, PyTorch default)
    out = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))

    tl.store(Y_ptr + row * stride_row + cols, out.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)

        orig_shape = h.shape
        h2 = h.reshape(-1, orig_shape[-1])
        if not h2.is_contiguous():
            h2 = h2.contiguous()

        rows, N = h2.shape
        y = torch.empty_like(h2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _ln_softmax_gelu_kernel[(rows,)](
            h2, self.ln1_g, self.ln1_b, y,
            N, h2.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.reshape(orig_shape)
