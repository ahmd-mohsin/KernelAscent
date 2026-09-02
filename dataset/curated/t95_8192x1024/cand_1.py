import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 95
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _ln_scale_softmax_kernel(
    X, G, B, Y,
    stride_xm, stride_ym,
    N, eps, scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm statistics in fp32 (matches PyTorch's fp32 accumulation for fp16 input)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * g + b
    # Round to fp16 to match PyTorch's intermediate output dtype
    y = y.to(tl.float16).to(tl.float32)
    # Scale in fp16 arithmetic semantics: fp32 mul then round to fp16
    y = (y * scale).to(tl.float16).to(tl.float32)

    # Softmax in fp32 (matches PyTorch internal fp32 compute for fp16 softmax)
    y = tl.where(mask, y, float('-inf'))
    y_max = tl.max(y, axis=0)
    num = tl.exp(y - y_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Y + row * stride_ym + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = y * 1.0224
            return torch.softmax(y, dim=-1)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 1024 else 4

        _ln_scale_softmax_kernel[(Mrows,)](
            x2, self.ln0_g, self.ln0_b, out,
            x2.stride(0), out.stride(0),
            N, 1e-5, 1.0224,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
