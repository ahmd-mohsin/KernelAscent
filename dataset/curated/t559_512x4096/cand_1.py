import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 559
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _fused_softmax_ln_relu_bias(
    X, OUT, G, B, B3,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch's bf16 softmax)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom

    # round to bf16 (softmax output dtype), then upcast for layernorm stats
    sm_bf = sm.to(tl.bfloat16)
    v = sm_bf.to(tl.float32)

    # layernorm stats in fp32
    mean = tl.sum(tl.where(mask, v, 0.0), axis=0) / N
    d = tl.where(mask, v - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (v - mean) * rstd * g + b
    y_bf = y.to(tl.bfloat16)

    # relu in bf16
    zero = tl.zeros_like(y_bf)
    y_bf = tl.maximum(y_bf, zero)

    # bf16 add: exact fp32 sum of two bf16 values, rounded once to bf16
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y_bf.to(tl.float32) + b3).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = torch.relu(y)
            return y + self.b3

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.reshape(-1, N)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_softmax_ln_relu_bias[(rows,)](
            x2, out, self.ln1_g, self.ln1_b, self.b3,
            N, x2.stride(0), out.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.reshape(orig_shape)
