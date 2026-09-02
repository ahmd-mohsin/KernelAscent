import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 607
M, D, DT = 512, 513, torch.float16


@triton.jit
def _ln_fwd_kernel(X, G, B, Y, N, stride_x, stride_y, eps,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


@triton.jit
def _rms_fwd_kernel(X, W, Y, N, stride_x, stride_y, eps,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + eps)
    w = tl.load(W + cols, mask=mask, other=0.0)
    # match reference: cast normalized value to fp16 first, then multiply by fp16 weight
    y = (x * rrms).to(tl.float16) * w
    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = x @ self.W1
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return x

        orig_shape = x.shape
        N_in = orig_shape[-1]
        x2 = x.contiguous().view(-1, N_in)
        rows = x2.shape[0]

        # Fused LayerNorm (fp32 accumulation, fp16 output)
        ln_out = torch.empty_like(x2)
        BLOCK_IN = triton.next_power_of_2(N_in)
        _ln_fwd_kernel[(rows,)](
            x2, self.ln0_g, self.ln0_b, ln_out,
            N_in, x2.stride(0), ln_out.stride(0), 1e-5,
            BLOCK=BLOCK_IN, num_warps=4,
        )

        # cuBLAS fp16 tensor-core GEMM
        h = torch.matmul(ln_out, self.W1)

        # Fused RMSNorm
        N_out = h.shape[-1]
        out = torch.empty_like(h)
        BLOCK_OUT = triton.next_power_of_2(N_out)
        _rms_fwd_kernel[(rows,)](
            h, self.rms2_w, out,
            N_out, h.stride(0), out.stride(0), 1e-6,
            BLOCK=BLOCK_OUT, num_warps=4,
        )

        return out.view(*orig_shape[:-1], N_out)
