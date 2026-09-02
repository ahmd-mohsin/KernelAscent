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
    X, Y,
    G0, B0, G1, B1, G2, B2,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    n_f = N.to(tl.float32)

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 ----
    g = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_f
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + eps)
    x = xc * rstd * g + b
    # match PyTorch: output of each layer_norm is cast back to fp16
    x = x.to(tl.float16).to(tl.float32)

    # ---- LayerNorm 1 ----
    g = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    x_m = tl.where(mask, x, 0.0)
    mean = tl.sum(x_m, axis=0) / n_f
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + eps)
    x = xc * rstd * g + b
    x = x.to(tl.float16).to(tl.float32)

    # ---- LayerNorm 2 ----
    g = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    x_m = tl.where(mask, x, 0.0)
    mean = tl.sum(x_m, axis=0) / n_f
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + eps)
    x = xc * rstd * g + b
    x = x.to(tl.float16).to(tl.float32)

    # ---- Softmax ----
    x_masked = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x_masked, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
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
            x = torch.softmax(x, dim=-1)
            return torch.relu(x)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8
        if BLOCK >= 4096:
            num_warps = 16
        if BLOCK >= 16384:
            num_warps = 32

        _fused_ln3_softmax_relu[(Mrows,)](
            x2, y,
            self.ln0_g, self.ln0_b,
            self.ln1_g, self.ln1_b,
            self.ln2_g, self.ln2_b,
            N, x2.stride(0), y.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
