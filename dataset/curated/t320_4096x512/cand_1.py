import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 320
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _ln_relu_kernel(X, G, B, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    y = tl.maximum(y, 0.0)
    tl.store(Y + row * N + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)
        if not h.is_cuda:
            h = F.layer_norm(h, (h.shape[-1],), self.ln1_g, self.ln1_b)
            return torch.relu(h)
        h = h.contiguous()
        rows, N = h.shape[0], h.shape[-1]
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _ln_relu_kernel[(rows,)](
            h, self.ln1_g, self.ln1_b, y, N, 1e-5,
            BLOCK=BLOCK, num_warps=4,
        )
        return y
