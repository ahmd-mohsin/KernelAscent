import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 904
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_epilogue(
    X_ptr, W_ptr, O_ptr,
    stride_x, stride_o,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    x = tl.load(X_ptr + row * stride_x + offs).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu (exact, erf) computed in fp32, rounded to fp16 like PyTorch
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # relu
    g = tl.maximum(g, 0.0)

    # rmsnorm in fp32, cast to fp16
    ms = tl.sum(g * g, axis=0) / N
    r = (g * tl.math.rsqrt(ms + 1e-6)).to(tl.float16).to(tl.float32)

    # weight multiply (opmath fp32, rounded fp16)
    w = tl.load(W_ptr + offs).to(tl.float32)
    y = (r * w).to(tl.float16).to(tl.float32)

    # second gelu
    g2 = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    g2 = g2.to(tl.float16).to(tl.float32)

    # softmax in fp32, output fp16
    m = tl.max(g2, axis=0)
    e = tl.exp(g2 - m)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(O_ptr + row * stride_o + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            h = x @ self.W0
            h = F.gelu(h)
            h = torch.relu(h)
            hf = h.float()
            h = (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(h.dtype) * self.rms3_w
            h = F.gelu(h)
            return torch.softmax(h, dim=-1)

        # GEMM via cuBLAS (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        out = torch.empty_like(h)
        n_rows, n_cols = h.shape

        grid = (n_rows,)
        _fused_epilogue[grid](
            h, self.rms3_w, out,
            h.stride(0), out.stride(0),
            N=n_cols, BLOCK=n_cols,
            num_warps=4,
        )
        return out
