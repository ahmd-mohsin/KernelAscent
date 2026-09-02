import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 806
M, D, DT = 4096, 512, torch.bfloat16

@triton.jit
def _fused_bias_gelu_relu_bias_softmax(
    X, B0, B3, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x + b0  (bf16 rounding to match reference)
    x = (x + b0).to(tl.bfloat16).to(tl.float32)

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))), computed in fp32, rounded to bf16
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # relu
    g = tl.maximum(g, 0.0)

    # + b3 with bf16 rounding
    z = (g + b3).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches PyTorch internal upcast for bf16)
    z = tl.where(mask, z, float('-inf'))
    m = tl.max(z, 0)
    e = tl.exp(z - m)
    s = tl.sum(e, 0)
    y = e / s

    tl.store(Y + row * D + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = F.gelu(x)
            x = torch.relu(x)
            x = x + self.b3
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        Mrows, Dcols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_bias_gelu_relu_bias_softmax[(Mrows,)](
            x, self.b0, self.b3, y,
            D=Dcols, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
