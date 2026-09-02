import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 958
M, D, DT = 2048, 4097, torch.bfloat16


@triton.jit
def _fused_softmax_gelu_ln_kernel(
    X, W, B, B4, Out,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, round to bf16 like torch output)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16).to(tl.float32)

    # gelu (exact erf variant), round to bf16
    g = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))
    g = tl.where(mask, g, 0.0)
    g = g.to(tl.bfloat16).to(tl.float32)

    # layernorm with fp32 statistics
    mean = tl.sum(g, axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * w + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # scale, round to bf16 (matches separate bf16 op)
    y = (y * 1.2578).to(tl.bfloat16).to(tl.float32)

    # add bias
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y + b4).to(tl.bfloat16)

    tl.store(Out + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = F.gelu(y)
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            y = y * 1.2578
            y = y + self.b4
            return y

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK <= 4096 else 16

        _fused_softmax_gelu_ln_kernel[(rows,)](
            x2, self.ln2_g, self.ln2_b, self.b4, out,
            N, x2.stride(0), out.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
