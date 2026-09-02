import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 849
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_ln_rms_gelu(
    x_ptr, b0_ptr, g1_ptr, b1_ptr, w2_ptr, out_ptr,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(b0_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x + b0 (bf16 add -> round to bf16)
    x = (x + b0).to(tl.bfloat16).to(tl.float32)

    # LayerNorm (fp32 accumulation, biased var, eps=1e-5), output rounded to bf16
    mean = tl.sum(x, axis=0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g1 = tl.load(g1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(b1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = ((x - mean) * rstd * g1 + b1).to(tl.bfloat16).to(tl.float32)

    # RMSNorm: fp32 mean of squares of bf16 values, eps=1e-6
    yy = tl.where(mask, y * y, 0.0)
    ms = tl.sum(yy, axis=0) / D
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    w2 = tl.load(w2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # (yf * rsqrt).to(bf16) * w2  (bf16 mul -> fp32 compute, round to bf16)
    z = ((y * rrms).to(tl.bfloat16).to(tl.float32) * w2).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf), computed in fp32, rounded to bf16
    out = 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))
    tl.store(out_ptr + row * D + offs, out.to(tl.bfloat16), mask=mask)


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
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        n_rows = xc.shape[0]
        out = torch.empty_like(xc)
        BLOCK = triton.next_power_of_2(d)
        _fused_ln_rms_gelu[(n_rows,)](
            xc, self.b0, self.ln1_g, self.ln1_b, self.rms2_w, out,
            D=d, BLOCK=BLOCK,
            num_warps=4,
        )
        return out.view(orig_shape)
