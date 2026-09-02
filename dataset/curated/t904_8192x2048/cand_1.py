import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 904
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_gelu_relu_rms_gelu_softmax(
    X_ptr, W_ptr, Out_ptr,
    stride_row,
    N_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, N_COLS)
    ptr = X_ptr + row * stride_row + offs

    # load (fp16) -> fp32
    x = tl.load(ptr).to(tl.float32)

    # gelu (exact, erf), rounded back to fp16 like reference intermediate
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # relu
    g = tl.maximum(g, 0.0)

    # rmsnorm in fp32 (matches _xf = x.float() path)
    ms = tl.sum(g * g, axis=0) / N_COLS
    r = tl.math.rsqrt(ms + 1e-6)
    y = (g * r).to(tl.float16).to(tl.float32)

    # scale by rms weight (fp16 params, fp32 compute -> fp16 round, like PyTorch half mul)
    w = tl.load(W_ptr + offs).to(tl.float32)
    y = (y * w).to(tl.float16).to(tl.float32)

    # gelu again
    y = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    y = y.to(tl.float16).to(tl.float32)

    # softmax with fp32 accumulation (matches PyTorch half softmax)
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Out_ptr + row * stride_row + offs, out.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        n_rows, n_cols = h.shape
        out = torch.empty_like(h)
        grid = (n_rows,)
        _fused_gelu_relu_rms_gelu_softmax[grid](
            h, self.rms3_w, out,
            h.stride(0),
            N_COLS=n_cols,
            num_warps=4,
        )
        return out
