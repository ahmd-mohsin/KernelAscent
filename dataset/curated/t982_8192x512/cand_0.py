import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 982
M, D, DT = 8192, 512, torch.float16


@triton.jit
def _fused_kernel(
    x_ptr, out_ptr,
    w1_ptr, w2_ptr, b3_ptr, g4_ptr, be4_ptr,
    N, D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0)
    # relu (fp16)
    x = tl.maximum(x, 0.0)

    # RMSNorm 1
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    rstd = 1.0 / tl.sqrt(ms + 1e-6)
    w1 = tl.load(w1_ptr + cols, mask=mask, other=0.0)
    x = (xf * rstd).to(tl.float16) * w1

    # RMSNorm 2
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    rstd = 1.0 / tl.sqrt(ms + 1e-6)
    w2 = tl.load(w2_ptr + cols, mask=mask, other=0.0)
    x = (xf * rstd).to(tl.float16) * w2

    # bias add (fp16)
    b3 = tl.load(b3_ptr + cols, mask=mask, other=0.0)
    x = x + b3

    # LayerNorm (fp32 internal, like PyTorch half layer_norm)
    xf = x.to(tl.float32)
    mean = tl.sum(xf, axis=0) / D
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(g4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(be4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * inv * g + b

    tl.store(out_ptr + row * D + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = x + self.b3
            return F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)

        orig_shape = x.shape
        Dd = orig_shape[-1]
        x2 = x.contiguous().view(-1, Dd)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)
        _fused_kernel[(n_rows,)](
            x2, out,
            self.rms1_w, self.rms2_w, self.b3, self.ln4_g, self.ln4_b,
            n_rows, D=Dd, BLOCK=triton.next_power_of_2(Dd),
            num_warps=4,
        )
        return out.view(orig_shape)
