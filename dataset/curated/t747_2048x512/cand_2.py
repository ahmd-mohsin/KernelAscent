import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 747
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_ln_bias_rms_kernel(
    X, G, B, B1, W2, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, like PyTorch's bf16 layer_norm)
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xc * rstd * g + b).to(tl.bfloat16).to(tl.float32)  # round to bf16 like reference

    # + bias (bf16 rounding as in reference elementwise add)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    z = (y + b1).to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32, cast to bf16, then multiply by weight, then scalar scale
    zm = tl.where(mask, z, 0.0)
    ms = tl.sum(zm * zm, axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)

    w2 = tl.load(W2 + offs, mask=mask, other=0.0).to(tl.float32)
    o = (z * r).to(tl.bfloat16).to(tl.float32)
    o = (o * w2).to(tl.bfloat16).to(tl.float32)
    o = (o * 1.1319).to(tl.bfloat16)

    tl.store(Y + row * D + offs, o, mask=mask)


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
            # CPU fallback (reference path)
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = x + self.b1
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return x * 1.1319

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_ln_bias_rms_kernel[(n_rows,)](
            x2, self.ln0_g, self.ln0_b, self.b1, self.rms2_w, out,
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
