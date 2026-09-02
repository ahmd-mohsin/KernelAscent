import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 502
M, D, DT = 8192, 2049, torch.bfloat16


@triton.jit
def _ln_rms_fused_kernel(
    X_ptr, G_ptr, B_ptr, W_ptr, Y_ptr,
    N,
    eps_ln, eps_rms, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 math, like PyTorch's mixed-dtype layer_norm) ----
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps_ln)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    h = diff * rstd * g + b
    # round to bf16 exactly like the intermediate tensor in the reference
    h = h.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm on the bf16-rounded LN output (fp32 accumulation) ----
    h2 = tl.where(mask, h * h, 0.0)
    ms = tl.sum(h2, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + eps_rms)

    y = (h * rrms).to(tl.bfloat16).to(tl.float32)  # .to(x.dtype) rounding

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y * w).to(tl.bfloat16).to(tl.float32)     # bf16 mul (fp32 opmath) rounding
    y = (y * scale).to(tl.bfloat16)                # final scalar mul rounding

    tl.store(Y_ptr + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return x * 1.1595

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        _ln_rms_fused_kernel[(rows,)](
            x2, self.ln0_g, self.ln0_b, self.rms1_w, y,
            N,
            1e-5, 1e-6, 1.1595,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
