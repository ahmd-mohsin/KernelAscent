import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 155
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_relu_ln_bias_rms_kernel(
    X, G, B, B2, W, Y,
    N,
    eps_ln, eps_rms, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # Load row, upcast to fp32
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # ReLU (exact, order with upcast doesn't matter)
    x = tl.maximum(x, 0.0)

    # LayerNorm (fp32 accumulation, like PyTorch's bf16 layer_norm)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps_ln)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    # round to bf16 (layer_norm output dtype)
    y = y.to(tl.bfloat16).to(tl.float32)

    # x + b2 (fp32 compute, bf16 round like PyTorch elementwise)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b2).to(tl.bfloat16).to(tl.float32)

    # RMSNorm: fp32 mean of squares, rsqrt, multiply, cast to bf16
    yz = tl.where(mask, y, 0.0)
    ms = tl.sum(yz * yz, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps_rms)
    y = (y * r).to(tl.bfloat16).to(tl.float32)

    # * rms3_w (bf16 elementwise -> fp32 compute, bf16 round)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y * w).to(tl.bfloat16).to(tl.float32)

    # * scalar (fp32 compute, bf16 round on store)
    y = y * scale

    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = x + self.b2
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            x = x * 1.0873
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        xc = x.contiguous().view(-1, N)
        Mrows = xc.shape[0]
        y = torch.empty_like(xc)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_relu_ln_bias_rms_kernel[(Mrows,)](
            xc, self.ln1_g, self.ln1_b, self.b2, self.rms3_w, y,
            N,
            1e-5, 1e-6, 1.0873,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
