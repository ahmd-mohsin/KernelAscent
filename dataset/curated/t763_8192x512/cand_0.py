import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 763
M, D, DT = 8192, 512, torch.float16


@triton.jit
def _fused_ln_rms_kernel(
    X, G, B, W, Y,
    N,
    eps_ln, eps_rms,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, matching PyTorch's mixed-precision path)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + eps_ln)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * inv * g + b

    # Round to fp16 (layer_norm output dtype), then upcast (matches x.float())
    y16 = y.to(tl.float16)
    yf = y16.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)

    # RMSNorm in fp32
    ms = tl.sum(yf * yf, axis=0) / N
    r = yf * tl.math.rsqrt(ms + eps_rms)

    # Cast to fp16, then multiply by weight in fp16 (matches reference ordering)
    r16 = r.to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    out = r16 * w

    tl.store(Y + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        Mrows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_ln_rms_kernel[(Mrows,)](
            x2d, self.ln0_g, self.ln0_b, self.rms1_w, y,
            N,
            1e-5, 1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
