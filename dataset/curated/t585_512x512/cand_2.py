import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 585
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_rms_act_kernel(
    X, W, Y,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    xn = (xf * inv).to(tl.float16)  # round to fp16 like .to(x.dtype)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float16)

    # fp16 multiplies (each rounds to fp16, matching PyTorch half arithmetic)
    h = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    h = (h.to(tl.float32) * 1.4123).to(tl.float16)
    h = tl.maximum(h, 0.0)
    h = (h.to(tl.float32) * 1.0481).to(tl.float16)

    # exact GELU computed in fp32 (matches PyTorch CUDA opmath for half)
    hf = h.to(tl.float32)
    g = hf * 0.5 * (1.0 + tl.math.erf(hf * 0.7071067811865476))
    out = g.to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            h = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            h = h * 1.4123
            h = torch.relu(h)
            h = h * 1.0481
            return F.gelu(h)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        _fused_rms_act_kernel[(rows,)](
            x2, self.rms0_w, y,
            N, x2.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=4 if BLOCK <= 1024 else 8,
        )
        return y.view(orig_shape)
