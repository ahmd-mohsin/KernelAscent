import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 356
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _gelu_f32(x):
    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    return 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))


@triton.jit
def _fused_kernel(
    X, G, B, Y,
    stride_xm, stride_ym,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # gelu 1 (round to bf16 like PyTorch elementwise op output)
    x = _gelu_f32(x)
    x = x.to(tl.bfloat16).to(tl.float32)

    # layer norm in fp32
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # gelu x3, rounding to bf16 between each (matches PyTorch op boundaries)
    y = _gelu_f32(y)
    y = y.to(tl.bfloat16).to(tl.float32)
    y = _gelu_f32(y)
    y = y.to(tl.bfloat16).to(tl.float32)
    y = _gelu_f32(y)

    tl.store(Y + row * stride_ym + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = F.gelu(x)
            x = F.gelu(x)
            x = F.gelu(x)
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        Mrows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_kernel[(Mrows,)](
            x2d, self.ln1_g, self.ln1_b, y,
            x2d.stride(0), y.stride(0),
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
