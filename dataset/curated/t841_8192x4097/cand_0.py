import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 841
M, D, DT = 8192, 4097, torch.float16


@triton.jit
def _fused_ln3_softmax_relu(
    X, G0, B0, G1, B1, G2, B2, Y,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 ----
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    x = d * rstd * g + b
    x = x.to(tl.float16).to(tl.float32)  # round to fp16 between ops (match reference)

    # ---- LayerNorm 1 ----
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    x = d * rstd * g + b
    x = x.to(tl.float16).to(tl.float32)

    # ---- LayerNorm 2 ----
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    x = d * rstd * g + b
    x = x.to(tl.float16).to(tl.float32)

    # ---- Softmax ----
    xm = tl.where(mask, x, float('-inf'))
    row_max = tl.max(xm, axis=0)
    e = tl.exp(xm - row_max)  # exp(-inf) = 0 for masked lanes
    denom = tl.sum(e, axis=0)
    out = e / denom

    # ---- ReLU ----
    out = tl.maximum(out, 0.0)

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            return torch.relu(torch.softmax(x, dim=-1))

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.reshape(-1, N).contiguous()
        rows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        _fused_ln3_softmax_relu[(rows,)](
            x2d, self.ln0_g, self.ln0_b, self.ln1_g, self.ln1_b,
            self.ln2_g, self.ln2_b, y,
            N, x2d.stride(0), y.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return y.reshape(orig_shape)
