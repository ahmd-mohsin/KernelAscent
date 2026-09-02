import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 498
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _fused_rms_ln_kernel(
    X, RMS_W, LN_G, LN_B, Y,
    N,
    eps_rms, eps_ln,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x_ptr = X + row * N + cols
    x16 = tl.load(x_ptr, mask=mask, other=0.0)
    xf = x16.to(tl.float32)

    # RMSNorm (stats in fp32, like reference)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps_rms)
    xn16 = (xf * r).to(tl.float16)          # cast to fp16 (matches .to(x.dtype))

    w = tl.load(RMS_W + cols, mask=mask, other=0.0)  # fp16
    y16 = xn16 * w                            # fp16 * fp16 -> fp16 (matches ref)
    yf = y16.to(tl.float32)

    # LayerNorm (stats in fp32, like PyTorch's native half layer_norm)
    mean = tl.sum(yf, axis=0) / N
    var = tl.sum(yf * yf, axis=0) / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps_ln)

    g = tl.load(LN_G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN_B + cols, mask=mask, other=0.0).to(tl.float32)

    out = (yf - mean) * rstd * g + b
    tl.store(Y + row * N + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Two GEMMs via cuBLAS tensor cores (fp32 accumulate) - same as reference
        x = torch.matmul(torch.matmul(x, self.W0), self.W1)

        orig_shape = x.shape
        Nn = orig_shape[-1]
        x2 = x.contiguous().view(-1, Nn)
        Mm = x2.shape[0]

        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(Nn)
        _fused_rms_ln_kernel[(Mm,)](
            x2, self.rms2_w, self.ln3_g, self.ln3_b, out,
            Nn, 1e-6, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
