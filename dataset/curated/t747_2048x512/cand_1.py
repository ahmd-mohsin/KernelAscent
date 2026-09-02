import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 747
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_ln_rms_kernel(
    X, G, B, B1, W, Y,
    stride_x, stride_y,
    D: tl.constexpr,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, D)

    x = tl.load(X + row * stride_x + offs).to(tl.float32)

    # LayerNorm (fp32 accumulation like PyTorch, then round to bf16)
    mean = tl.sum(x, axis=0) / D
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G + offs).to(tl.float32)
    b = tl.load(B + offs).to(tl.float32)
    ln = (xc * rstd * g + b).to(tl.bfloat16).to(tl.float32)

    # x + b1 in bf16 semantics (fp32 compute, bf16 round)
    b1 = tl.load(B1 + offs).to(tl.float32)
    t = (ln + b1).to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32, round to bf16
    ms = tl.sum(t * t, axis=0) / D
    r = 1.0 / tl.sqrt(ms + EPS_RMS)
    y = (t * r).to(tl.bfloat16).to(tl.float32)

    # multiply by rms2_w (bf16 op semantics)
    w = tl.load(W + offs).to(tl.float32)
    y = (y * w).to(tl.bfloat16).to(tl.float32)

    # scalar scale (bf16 op semantics)
    y = (y * SCALE).to(tl.bfloat16)

    tl.store(Y + row * stride_y + offs, y)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # fallback: reference path
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = x + self.b1
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = x * 1.1319
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]
        y = torch.empty_like(x2)

        _fused_ln_rms_kernel[(n_rows,)](
            x2, self.ln0_g, self.ln0_b, self.b1, self.rms2_w, y,
            x2.stride(0), y.stride(0),
            D=d,
            EPS_LN=1e-5,
            EPS_RMS=1e-6,
            SCALE=1.1319,
            num_warps=4,
        )
        return y.view(orig_shape)
