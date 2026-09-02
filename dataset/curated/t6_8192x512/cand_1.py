import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 6
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_ln_rms_kernel(
    X, G, B, W, Y,
    N,
    eps_ln, eps_rms,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (stats in fp32, like PyTorch's bf16 layer_norm)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    inv = 1.0 / tl.sqrt(var + eps_ln)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * inv * g + b

    # Round to bf16 (matches intermediate materialization in reference)
    yb = y.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32 on the bf16-rounded values
    ms = tl.sum(tl.where(mask, yb * yb, 0.0), axis=0) / N
    rinv = 1.0 / tl.sqrt(ms + eps_rms)

    # (._xf * rsqrt).to(bf16) then bf16*bf16 mul (opmath fp32 -> bf16)
    z = (yb * rinv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z * w).to(tl.bfloat16)

    tl.store(Y + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x @ self.W0
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return x

        # Matmul via cuBLAS (tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        orig_shape = h.shape
        N = orig_shape[-1]
        h2d = h.view(-1, N)
        rows = h2d.shape[0]

        out = torch.empty_like(h2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_ln_rms_kernel[(rows,)](
            h2d, self.ln1_g, self.ln1_b, self.rms2_w, out,
            N,
            1e-5, 1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return out.view(orig_shape)
