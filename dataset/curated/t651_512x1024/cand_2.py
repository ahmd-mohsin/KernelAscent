import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 651
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _rms_relu_softmax_kernel(
    X_ptr, W_ptr, Out_ptr,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # load row (fp16 -> fp32 for RMS stats, matching x.float())
    x = tl.load(X_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm: mean of squares in fp32, rsqrt, then cast product to fp16
    ms = tl.sum(x * x, axis=0) / D
    rs = tl.math.rsqrt(ms + 1e-6)
    y = (x * rs).to(tl.float16)

    # multiply by weight in fp16 (matches .to(dtype) * rms1_w)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    y = y * w

    # ReLU in fp16
    y = tl.maximum(y, 0.0)

    # softmax with fp32 accumulation (matches PyTorch half softmax on CUDA)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    mmax = tl.max(yf, axis=0)
    e = tl.exp(yf - mmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Out_ptr + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM (tensor cores on A100)
        x = x @ self.W0

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]

        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8
        _rms_relu_softmax_kernel[(n_rows,)](
            x2, self.rms1_w, out,
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
