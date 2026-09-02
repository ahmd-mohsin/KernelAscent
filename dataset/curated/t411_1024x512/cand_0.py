import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 411
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _fused_scale_ln_softmax(
    X, G, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # scale in fp16 (round to fp16 like eager op), then upcast for reductions
    xs = (x * SCALE).to(tl.float16)
    xf = xs.to(tl.float32)

    # LayerNorm (fp32 accumulation, like PyTorch on half inputs)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    d = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    ln = (xf - mean) * rstd * g + b
    # round to fp16 (layer_norm output dtype) before softmax, like eager
    lnh = ln.to(tl.float16)
    lf = lnh.to(tl.float32)

    # Softmax in fp32
    lf = tl.where(mask, lf, float('-inf'))
    m = tl.max(lf, axis=0)
    e = tl.exp(lf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.4183
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_scale_ln_softmax[(rows,)](
            x2, self.ln1_g, self.ln1_b, y,
            x2.stride(0), y.stride(0),
            N=N, SCALE=1.4183, EPS=1e-5, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
