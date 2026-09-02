import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 519
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _softmax_scale_rms_kernel(
    X_ptr, W_ptr, Out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax in fp32 (matches PyTorch bf16 softmax internal accumulation)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s

    # cast to bf16 (softmax output dtype), then scale by 1.0722 with fp32 opmath
    p_bf = p.to(tl.bfloat16)
    y_bf = (p_bf.to(tl.float32) * 1.0722).to(tl.bfloat16)

    # RMS norm in fp32
    yf = y_bf.to(tl.float32)
    ms = tl.sum(yf * yf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    z_bf = (yf * r).to(tl.bfloat16)

    # multiply by weight (bf16 * bf16 with fp32 opmath)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (z_bf.to(tl.float32) * w).to(tl.bfloat16)

    tl.store(Out_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores)
        h = x @ self.W0
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _softmax_scale_rms_kernel[(Mrows,)](
            h, self.rms3_w, out,
            N,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
