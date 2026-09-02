import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 201
M, D, DT = 512, 2048, torch.bfloat16

_SQRT1_2 = 0.7071067811865476


@triton.jit
def _fused_kernel(
    x_ptr, w_ptr, out_ptr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0)

    # gelu (exact) computed in fp32, cast back to bf16 (matches PyTorch opmath)
    xf = x.to(tl.float32)
    g1 = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g1_b = g1.to(x_ptr.dtype.element_ty)

    # relu in bf16 (exact)
    r = tl.maximum(g1_b, 0.0)

    # rmsnorm: fp32 accumulation, cast normalized to bf16
    rf = r.to(tl.float32)
    ms = tl.sum(tl.where(mask, rf * rf, 0.0), axis=0) / D
    inv = tl.math.rsqrt(ms + 1e-6)
    normed_b = (rf * inv).to(x_ptr.dtype.element_ty)

    # multiply by weight: bf16*bf16 -> fp32 opmath -> bf16
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    scaled = (normed_b.to(tl.float32) * w.to(tl.float32)).to(x_ptr.dtype.element_ty)

    # final gelu: fp32 opmath -> bf16
    sf = scaled.to(tl.float32)
    g2 = 0.5 * sf * (1.0 + tl.math.erf(sf * 0.7071067811865476))
    out = g2.to(x_ptr.dtype.element_ty)

    tl.store(out_ptr + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            xg = F.gelu(x)
            xg = torch.relu(xg)
            _xf = xg.float()
            xg = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(xg.dtype) * self.rms2_w
            return F.gelu(xg)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x2, self.rms2_w, out,
            D=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
