import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 879
M, D, DT = 512, 2049, torch.float16


@triton.jit
def _relu_rms_ln_kernel(
    X, W_rms, G, B, Y,
    N, stride_x, stride_y,
    eps_rms, eps_ln,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # ReLU (in fp16, same as torch.relu)
    x = tl.maximum(x, 0.0)

    # RMSNorm in fp32
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + eps_rms)
    # round to fp16 before multiplying by rms weight (matches reference)
    h = (xf * r).to(tl.float16)
    w = tl.load(W_rms + cols, mask=mask, other=0.0)
    h = h * w  # fp16 multiply, matching reference

    # LayerNorm: fp32 accumulation (matches F.layer_norm on half inputs)
    hf = h.to(tl.float32)
    mean = tl.sum(tl.where(mask, hf, 0.0), axis=0) / N
    d = tl.where(mask, hf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + eps_ln)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * inv * g + b

    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)

        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.reshape(-1, N)
        if not h2.is_contiguous():
            h2 = h2.contiguous()
        rows = h2.shape[0]
        out = torch.empty_like(h2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _relu_rms_ln_kernel[(rows,)](
            h2, self.rms2_w, self.ln3_g, self.ln3_b, out,
            N, h2.stride(0), out.stride(0),
            1e-6, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.reshape(orig_shape)
