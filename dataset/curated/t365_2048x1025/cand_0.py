import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 365
M, D, DT = 2048, 1025, torch.float16


@triton.jit
def _fused_row_kernel(
    X_ptr, B1_ptr, G_ptr, B_ptr, Y_ptr,
    N_COLS, x_stride, y_stride,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N_COLS

    # ---- load (fp16 -> fp32 opmath, matching PyTorch half elementwise kernels)
    x = tl.load(X_ptr + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- x * 1.0149 (fp32 opmath, round to fp16 like PyTorch intermediate)
    t = (x * 1.0149).to(tl.float16).to(tl.float32)
    # ---- + b1 (fp32 opmath, round to fp16)
    t = (t + b1).to(tl.float16).to(tl.float32)

    # ---- softmax #1 (computed in fp32, result stored as fp16)
    tneg = tl.where(mask, t, float('-inf'))
    m1 = tl.max(tneg, 0)
    e1 = tl.exp(tneg - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, 0)
    p = (e1 / s1).to(tl.float16).to(tl.float32)

    # ---- layer norm (fp32 stats, biased variance, eps=1e-5, result -> fp16)
    pm = tl.where(mask, p, 0.0)
    mean = tl.sum(pm, 0) / N_COLS
    diff = tl.where(mask, p - mean, 0.0)
    var = tl.sum(diff * diff, 0) / N_COLS
    rstd = 1.0 / tl.sqrt(var + EPS)
    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = ((p - mean) * rstd * g + bb).to(tl.float16).to(tl.float32)

    # ---- softmax #2
    yneg = tl.where(mask, y, float('-inf'))
    m2 = tl.max(yneg, 0)
    e2 = tl.exp(yneg - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, 0)
    out = (e2 / s2).to(tl.float16)

    tl.store(Y_ptr + row * y_stride + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # fallback (reference path)
            x = x * 1.0149
            x = x + self.b1
            x = torch.softmax(x, dim=-1)
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2 = x.contiguous().view(-1, n_cols)
        n_rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_row_kernel[(n_rows,)](
            x2, self.b1, self.ln3_g, self.ln3_b, y,
            n_cols, x2.stride(0), y.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
