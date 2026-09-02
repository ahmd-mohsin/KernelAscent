import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 500
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _fused_softmax_bias_ln_kernel(
    X_ptr, B1_ptr, G_ptr, B_ptr, Out_ptr,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulation, matches PyTorch bf16 softmax which upcasts)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # round to bf16 (softmax output dtype), then add bf16 bias, round to bf16
    p_bf = p.to(tl.bfloat16).to(tl.float32)
    b1 = tl.load(B1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = (p_bf + b1).to(tl.bfloat16).to(tl.float32)
    z = tl.where(mask, z, 0.0)

    # layernorm in fp32
    n_f = N.to(tl.float32)
    mean = tl.sum(z, axis=0) / n_f
    d = tl.where(mask, z - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b

    tl.store(Out_ptr + row * stride_o + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_bias_ln_kernel[(Mrows,)](
            x2, self.b1, self.ln2_g, self.ln2_b, out,
            N, x2.stride(0), out.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=16 if BLOCK >= 4096 else 8,
        )
        return out.view(orig_shape)
