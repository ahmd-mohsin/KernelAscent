import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 988
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _rmsnorm_softmax_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (fp32 math, matching reference)
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    xn = (x * inv).to(tl.float16)

    # scale by weight in fp16 (matching reference dtype behavior)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float16)
    v = xn * w

    # softmax with fp32 accumulation (matching PyTorch half softmax)
    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, float('-inf'))
    mx = tl.max(vf, axis=0)
    e = tl.exp(vf - mx)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (fp16 tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _rmsnorm_softmax_kernel[(Mrows,)](
            h, self.rms1_w, y,
            N, h.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
