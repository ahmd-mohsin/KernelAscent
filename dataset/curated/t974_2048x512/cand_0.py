import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 974
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_rms_ln_kernel(
    X_ptr, W1_ptr, G_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y,
    EPS_RMS: tl.constexpr, EPS_LN: tl.constexpr, SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (fp32 math, round to bf16 as reference does)
    ms = tl.sum(x * x, axis=0) / N
    rs = 1.0 / tl.sqrt(ms + EPS_RMS)
    y = (x * rs).to(tl.bfloat16).to(tl.float32)

    # multiply by rms weight: bf16*bf16 -> fp32 opmath, round to bf16
    w1 = tl.load(W1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    t = (y * w1).to(tl.bfloat16).to(tl.float32)

    # LayerNorm in fp32
    mean = tl.sum(t, axis=0) / N
    d = tl.where(mask, t - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    o = (d * inv * g + b).to(tl.bfloat16).to(tl.float32)
    # final scalar multiply: bf16 tensor * scalar -> fp32 opmath, round to bf16
    o = (o * SCALE).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + offs, o, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS (already optimal on A100 tensor cores)
        x = x @ self.W0

        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            return x * 1.3244

        x = x.contiguous()
        rows, N = x.shape[0], x.shape[-1]
        x2d = x.view(-1, N)
        rows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        _fused_rms_ln_kernel[(rows,)](
            x2d, self.rms1_w, self.ln2_g, self.ln2_b, y,
            N, x2d.stride(0), y.stride(0),
            EPS_RMS=1e-6, EPS_LN=1e-5, SCALE=1.3244,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view_as(x)
