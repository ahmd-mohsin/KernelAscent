import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 144
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _fused_rms_relu_softmax(X, W, Y, N,
                            C1, C2, EPS,
                            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output (fp16)
    x = tl.load(X + row * N + offs, mask=mask, other=0.0)

    # x = x * 1.2328 (fp16 arithmetic; exact product fits in fp32, then round to fp16)
    xf = x.to(tl.float32) * C1
    xs = xf.to(tl.float16)

    # RMSNorm in fp32
    xf32 = xs.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf32 * xf32, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS)
    xhat = (xf32 * r).to(tl.float16)

    # * rms2_w (fp16 mul)
    w = tl.load(W + offs, mask=mask, other=0.0)
    xw = (xhat.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # * 1.4636 (fp16 mul)
    xa = (xw.to(tl.float32) * C2).to(tl.float16)

    # relu
    xa = tl.maximum(xa, 0.0)

    # softmax in fp32 accumulation (matches PyTorch half softmax semantics)
    xsm = tl.where(mask, xa.to(tl.float32), float('-inf'))
    mx = tl.max(xsm, axis=0)
    e = tl.exp(xsm - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.float16)

    tl.store(Y + row * N + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        # constants pre-rounded to fp16 so fp32 mul + cast == fp16 mul exactly
        self._c1 = float(torch.tensor(1.2328, dtype=torch.float16).item())
        self._c2 = float(torch.tensor(1.4636, dtype=torch.float16).item())

    def forward(self, x):
        if not x.is_cuda:
            x = x @ self.W0
            x = x * 1.2328
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = x * 1.4636
            x = torch.relu(x)
            return torch.softmax(x, dim=-1)

        h = torch.matmul(x, self.W0)  # cuBLAS fp16 GEMM
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_relu_softmax[(rows,)](
            h, self.rms2_w, out, N,
            self._c1, self._c2, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
