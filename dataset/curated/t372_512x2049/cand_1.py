import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 372
M, D, DT = 512, 2049, torch.float16


@triton.jit
def _fused_softmax_relu_softmax_rms_kernel(
    X_ptr, W_ptr, OUT_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax 1 (fp32 accumulate, round to fp16 like PyTorch) ----
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y = (e1 / s1).to(tl.float16).to(tl.float32)

    # ---- relu ----
    y = tl.maximum(y, 0.0)

    # ---- softmax 2 ----
    y2 = tl.where(mask, y, float('-inf'))
    m2 = tl.max(y2, axis=0)
    e2 = tl.exp(y2 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    z = (e2 / s2).to(tl.float16).to(tl.float32)

    # ---- scale by 1.2112 (opmath fp32, round to fp16) ----
    z = (z * 1.2112).to(tl.float16).to(tl.float32)

    # ---- RMSNorm in fp32 ----
    zz = tl.where(mask, z * z, 0.0)
    ms = tl.sum(zz, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    zn = (z * inv).to(tl.float16).to(tl.float32)

    # ---- multiply by weight (opmath fp32, round to fp16) ----
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (zn * w).to(tl.float16)

    tl.store(OUT_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_softmax_relu_softmax_rms_kernel[(rows,)](
            h, self.rms5_w, out,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
