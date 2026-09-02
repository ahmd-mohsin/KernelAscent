import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 281
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, w0_ptr, w1_ptr, b3_ptr, out_ptr,
    N, stride_row,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # --- RMSNorm 0 ---
    ms0 = tl.sum(xf * xf, axis=0) / N
    r0 = 1.0 / tl.sqrt(ms0 + EPS)
    w0 = tl.load(w0_ptr + offs, mask=mask, other=0.0)
    x_bf = (xf * r0).to(tl.bfloat16) * w0  # bf16 multiply, matches PyTorch

    # --- RMSNorm 1 ---
    xf1 = x_bf.to(tl.float32)
    ms1 = tl.sum(xf1 * xf1, axis=0) / N
    r1 = 1.0 / tl.sqrt(ms1 + EPS)
    w1 = tl.load(w1_ptr + offs, mask=mask, other=0.0)
    x_bf2 = (xf1 * r1).to(tl.bfloat16) * w1

    # --- ReLU + bias (bf16, same as PyTorch elementwise) ---
    zero = tl.zeros_like(x_bf2)
    x_bf2 = tl.maximum(x_bf2, zero)
    b3 = tl.load(b3_ptr + offs, mask=mask, other=0.0)
    x_bf2 = x_bf2 + b3

    # --- Softmax (float32 accumulation like PyTorch, bf16 output) ---
    sf = x_bf2.to(tl.float32)
    sf = tl.where(mask, sf, float('-inf'))
    m = tl.max(sf, axis=0)
    e = tl.exp(sf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_row + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = torch.relu(x)
            x = x + self.b3
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(rows,)](
            x2d, self.rms0_w, self.rms1_w, self.b3, out,
            N, x2d.stride(0),
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
