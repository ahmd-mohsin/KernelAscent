import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 893
M, D, DT = 8192, 2049, torch.bfloat16


@triton.jit
def _ln_kernel(X, Y, W, B, N, stride_x, stride_y, eps,
               BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * w + b
    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _ln_relu_softmax_kernel(X, Y, W, B, N, stride_x, stride_y, eps,
                            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    # match reference: LN output rounded to bf16 before relu/softmax
    y_bf16 = (d * rstd * w + b).to(tl.bfloat16)
    y_bf16 = tl.maximum(y_bf16, 0.0)
    yf = y_bf16.to(tl.float32)
    yf_masked = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf_masked, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = x @ self.W1
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            x = torch.relu(x)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        rows, N0 = x2.shape

        y0 = torch.empty_like(x2)
        BLOCK0 = triton.next_power_of_2(N0)
        _ln_kernel[(rows,)](
            x2, y0, self.ln0_g, self.ln0_b, N0,
            x2.stride(0), y0.stride(0), 1e-5,
            BLOCK=BLOCK0, num_warps=8,
        )

        h = y0 @ self.W1  # cuBLAS bf16 tensor-core matmul

        N1 = h.shape[-1]
        out = torch.empty_like(h)
        BLOCK1 = triton.next_power_of_2(N1)
        _ln_relu_softmax_kernel[(rows,)](
            h, out, self.ln2_g, self.ln2_b, N1,
            h.stride(0), out.stride(0), 1e-5,
            BLOCK=BLOCK1, num_warps=4,
        )

        return out.view(*orig_shape[:-1], N1)
