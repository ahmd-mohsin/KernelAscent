import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 170
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_softmax_gelu_ln(
    X, BIAS, GAMMA, BETA, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # load matmul output (fp16) and add bias in fp16 (matches half+half add)
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float16)
    b = tl.load(BIAS + cols, mask=mask, other=0.0).to(tl.float16)
    x = x + b

    # softmax in fp32, cast result to fp16 (matches PyTorch half softmax output)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    mx = tl.max(xf, axis=0)
    e = tl.exp(xf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16)

    # exact erf-based GELU in fp32, cast to fp16
    pf = p.to(tl.float32)
    g = pf * 0.5 * (1.0 + tl.math.erf(pf * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # layernorm in fp32
    gf = g16.to(tl.float32)
    gf = tl.where(mask, gf, 0.0)
    mean = tl.sum(gf, axis=0) / N
    diff = tl.where(mask, gf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    gamma = tl.load(GAMMA + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(BETA + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * gamma + beta

    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)

        if not h.is_cuda:
            h = h + self.b1
            h = torch.softmax(h, dim=-1)
            h = F.gelu(h)
            return F.layer_norm(h, (h.shape[-1],), self.ln4_g, self.ln4_b)

        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 512 else 4
        _fused_softmax_gelu_ln[(rows,)](
            h, self.b1, self.ln4_g, self.ln4_b, out,
            h.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
