import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 144
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _fused_rms_relu_softmax(
    X_ptr, W_ptr, Y_ptr,
    N, stride,
    EPS: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    ptr = X_ptr + row * stride + cols

    # load matmul output (fp16)
    x = tl.load(ptr).to(tl.float32)

    # x = x * 1.2328  (fp16 tensor * scalar -> opmath fp32, result fp16)
    x = (x * S1).to(tl.float16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)
    xh = (x * inv).to(tl.float16)  # .to(x.dtype)

    # * rms2_w (fp16 * fp16 -> opmath fp32 -> fp16)
    w = tl.load(W_ptr + cols).to(tl.float32)
    x = (xh.to(tl.float32) * w).to(tl.float16)

    # * 1.4636
    x = (x.to(tl.float32) * S2).to(tl.float16)

    # relu
    x = tl.maximum(x, 0.0)

    # softmax in fp32, output fp16
    xf = x.to(tl.float32)
    mx = tl.max(xf, axis=0)
    e = tl.exp(xf - mx)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.float16)

    tl.store(Y_ptr + row * stride + cols, y)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_relu_softmax[(rows,)](
            h, self.rms2_w, out,
            N, h.stride(0),
            EPS=1e-6,
            S1=1.2328,
            S2=1.4636,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out
