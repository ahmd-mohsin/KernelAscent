import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 849
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_kernel(X, B0, G1, B1, W2, Y, N, eps_ln, eps_rms, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0)

    # x = x + b0 (bf16 arithmetic -> round to bf16)
    x = (x.to(tl.float32) + b0.to(tl.float32)).to(tl.bfloat16).to(tl.float32)

    # LayerNorm (fp32 compute, output rounded to bf16)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    inv = 1.0 / tl.sqrt(var + eps_ln)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (xc * inv * g1 + b1).to(tl.bfloat16).to(tl.float32)

    # RMSNorm: fp32 compute, cast to bf16, then bf16 multiply with rms2_w
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / N
    rinv = 1.0 / tl.sqrt(ms + eps_rms)
    xr = (x * rinv).to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (xr * w2).to(tl.bfloat16).to(tl.float32)

    # GELU (exact, fp32 compute, round to bf16)
    y = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return F.gelu(x)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](
            x2, self.b0, self.ln1_g, self.ln1_b, self.rms2_w, y,
            N, 1e-5, 1e-6, BLOCK=BLOCK,
            num_warps=4,
        )
        return y.view(orig_shape)
