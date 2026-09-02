import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 582
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _ln_relu_gelu_kernel(
    X, G, B, Y,
    N, stride_x, stride_y,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm in fp32 (matches PyTorch's internal accumulation for bf16)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b

    # Round to bf16 (LayerNorm output dtype), as the reference does between ops
    y = y.to(tl.bfloat16)

    # ReLU
    y = tl.maximum(y, 0.0)

    # Exact (erf-based) GELU, computed in fp32
    yf = y.to(tl.float32)
    out = yf * 0.5 * (1.0 + tl.math.erf(yf * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.view(-1, N)
        rows = h2.shape[0]
        y = torch.empty_like(h2)
        BLOCK_N = triton.next_power_of_2(N)
        _ln_relu_gelu_kernel[(rows,)](
            h2, self.ln1_g, self.ln1_b, y,
            N, h2.stride(0), y.stride(0),
            1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return y.view(orig_shape)
