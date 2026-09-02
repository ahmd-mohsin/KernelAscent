import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 171
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_softmax_rms_relu_rms(
    X, W2, W4, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    # ---- softmax (fp32 compute, matches torch half softmax accumulation) ----
    x = tl.load(X + base + offs, mask=mask, other=float('-inf')).to(tl.float32)
    mx = tl.max(x, axis=0)
    ex = tl.exp(x - mx)
    ssum = tl.sum(tl.where(mask, ex, 0.0), axis=0)
    x16 = (ex / ssum).to(tl.float16)

    # ---- scale by 1.0869 (torch half elementwise computes in fp32 opmath) ----
    x16 = (x16.to(tl.float32) * 1.0869).to(tl.float16)

    # ---- RMSNorm #1 ----
    xf = x16.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0)
    x16 = ((xf * r).to(tl.float16).to(tl.float32) * w2.to(tl.float32)).to(tl.float16)

    # ---- ReLU ----
    x16 = tl.maximum(x16, tl.full((), 0.0, tl.float16))

    # ---- RMSNorm #2 ----
    xf = x16.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    w4 = tl.load(W4 + offs, mask=mask, other=0.0)
    y = ((xf * r).to(tl.float16).to(tl.float32) * w4.to(tl.float32)).to(tl.float16)

    tl.store(Y + base + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.is_cuda and x.dtype == torch.float16
        x = x.contiguous()
        rows, d = x.shape[0] if x.dim() == 2 else x.numel() // x.shape[-1], x.shape[-1]
        orig_shape = x.shape
        x2 = x.view(-1, d)
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_softmax_rms_relu_rms[(x2.shape[0],)](
            x2, self.rms2_w, self.rms4_w, y,
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
