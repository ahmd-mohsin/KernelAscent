import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 917
M, D, DT = 8192, 513, torch.bfloat16


@triton.jit
def _fused_ln_gelu_bias_relu_softmax(
    X, G, B, B2, Y,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, matching PyTorch internals)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    y = y.to(tl.bfloat16).to(tl.float32)  # match bf16 rounding of layer_norm output

    # Exact GELU (erf-based, matching F.gelu default)
    y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    y = y.to(tl.bfloat16).to(tl.float32)  # match bf16 rounding

    # bias add
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    y = y + b2
    y = y.to(tl.bfloat16).to(tl.float32)  # match bf16 rounding

    # ReLU
    y = tl.maximum(y, 0.0)

    # Softmax (fp32 internally, as PyTorch upcasts)
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_y + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = F.gelu(y)
            y = y + self.b2
            y = torch.relu(y)
            return torch.softmax(y, dim=-1)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_ln_gelu_bias_relu_softmax[(rows,)](
            x2, self.ln0_g, self.ln0_b, self.b2, out,
            N, x2.stride(0), out.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
