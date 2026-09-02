import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 398
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _fused_bias_rms2_relu(
    Y_ptr, B_ptr, W2_ptr, W3_ptr, Out_ptr,
    N, stride_y, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    y = tl.load(Y_ptr + row * stride_y + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # bias add (fp32 opmath, rounded to bf16 like PyTorch)
    x = (y + b).to(tl.bfloat16)

    # first RMSNorm
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.rsqrt(ms + 1e-6)
    x = (xf * r).to(tl.bfloat16)
    w2 = tl.load(W2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x.to(tl.float32) * w2).to(tl.bfloat16)

    # second RMSNorm
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.rsqrt(ms + 1e-6)
    x = (xf * r).to(tl.bfloat16)
    w3 = tl.load(W3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    xf = x.to(tl.float32) * w3

    # relu, final cast to bf16
    xf = tl.maximum(xf, 0.0)
    out = xf.to(tl.bfloat16)
    tl.store(Out_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        y = torch.matmul(x, self.W0)
        if not y.is_cuda:
            # CPU fallback: reference path
            y = y + self.b1
            _xf = y.float()
            y = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms2_w
            _xf = y.float()
            y = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms3_w
            return torch.relu(y)

        y = y.contiguous()
        orig_shape = y.shape
        N = orig_shape[-1]
        y2 = y.view(-1, N)
        Mrows = y2.shape[0]
        out = torch.empty_like(y2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_bias_rms2_relu[(Mrows,)](
            y2, self.b1, self.rms2_w, self.rms3_w, out,
            N, y2.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
