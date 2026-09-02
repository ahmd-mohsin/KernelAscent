import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 397
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_kernel(X, W, G, B, Y, N, stride_x, stride_y,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (fp32 math), cast to fp16, multiply weight in fp16
    ms = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    w = tl.load(W + cols, mask=mask, other=0.0)
    h = (x * rrms).to(tl.float16) * w

    # ReLU (fp16)
    h = tl.maximum(h, tl.zeros_like(h))

    # LayerNorm: fp32 opmath, output fp16
    hf = h.to(tl.float32)
    mean = tl.sum(hf, axis=0) / N
    d = tl.where(mask, hf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    z = ((hf - mean) * rstd * g + b).to(tl.float16)

    # GELU (erf-based, fp32 opmath), output fp16
    zf = z.to(tl.float32)
    out = 0.5 * zf * (1.0 + tl.math.erf(zf * 0.7071067811865476))
    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            y = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            y = torch.relu(y)
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            return F.gelu(y)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(rows,)](
            x2, self.rms0_w, self.ln2_g, self.ln2_b, y,
            N, x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
