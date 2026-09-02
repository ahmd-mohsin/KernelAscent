import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 895
M, D, DT = 512, 4097, torch.float16


@triton.jit
def _fused_ln_rms_gelu_kernel(
    X_ptr, G_ptr, B_ptr, W_ptr, Y_ptr,
    N, stride_x, stride_y,
    LN_EPS: tl.constexpr, RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 accumulation, like PyTorch) ----
    n = N.to(tl.float32)
    mean = tl.sum(x, axis=0) / n
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n
    rstd = tl.math.rsqrt(var + LN_EPS)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    ln = (x - mean) * rstd * g + b
    # PyTorch layer_norm outputs fp16 -> round to fp16
    ln16 = ln.to(tl.float16)

    # ---- RMSNorm: cast fp16 result back to fp32 ----
    xf = ln16.to(tl.float32)
    xf_masked = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf_masked * xf_masked, axis=0) / n
    rrms = tl.math.rsqrt(ms + RMS_EPS)
    r16 = (xf * rrms).to(tl.float16)  # .to(x.dtype)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)  # fp16
    y16 = r16 * w  # fp16 multiply, matching reference

    # ---- GELU (exact, fp32 opmath like PyTorch CUDA half kernel) ----
    yf = y16.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    out = 0.5 * yf * (1.0 + tl.math.erf(yf * INV_SQRT2))
    out16 = out.to(tl.float16)

    tl.store(Y_ptr + row * stride_y + cols, out16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return F.gelu(x)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 4096 else 4

        _fused_ln_rms_gelu_kernel[(Mrows,)](
            x2, self.ln0_g, self.ln0_b, self.rms1_w, y,
            N, x2.stride(0), y.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
