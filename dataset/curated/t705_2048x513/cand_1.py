import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 705
M, D, DT = 2048, 513, torch.bfloat16


@triton.jit
def _fused_epilogue_kernel(
    X_ptr, B1_ptr, B2_ptr, G_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # load matmul result row (bf16)
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    b1 = tl.load(B1_ptr + cols, mask=mask, other=0.0)
    b2 = tl.load(B2_ptr + cols, mask=mask, other=0.0)

    # bias adds in bf16 (matches reference elementwise bf16 rounding)
    x = x + b1
    x = x + b2
    # relu in bf16
    zero = tl.zeros_like(x)
    x = tl.where(x > zero, x, zero)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for bf16 softmax)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float("-inf"))
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = e / denom
    # round to bf16 (reference materializes bf16 softmax output)
    p_bf16 = p.to(tl.bfloat16)

    # layernorm in fp32 (matches PyTorch's fp32 compute for bf16 layer_norm)
    pf = p_bf16.to(tl.float32)
    pf = tl.where(mask, pf, 0.0)
    mean = tl.sum(pf, axis=0) / N
    diff = tl.where(mask, pf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b

    tl.store(Y_ptr + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            h = x @ self.W0
            h = h + self.b1
            h = h + self.b2
            h = torch.relu(h)
            h = torch.softmax(h, dim=-1)
            return F.layer_norm(h, (h.shape[-1],), self.ln5_g, self.ln5_b)

        # cuBLAS matmul (bf16 with fp32 accumulate), identical to reference
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_epilogue_kernel[(Mrows,)](
            h, self.b1, self.b2, self.ln5_g, self.ln5_b, y,
            N, h.stride(0), y.stride(0), 1e-5,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
