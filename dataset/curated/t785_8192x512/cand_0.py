import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 785
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _softmax_rms_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, like torch.softmax on bf16 CUDA)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # round to bf16 as torch.softmax outputs bf16, then re-upcast (matches _xf = x.float())
    pb = p.to(tl.bfloat16)
    pf = pb.to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(pf * pf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)

    a = (pf * r).to(tl.bfloat16)
    w = tl.load(W_ptr + offs, mask=mask, other=0).to(tl.bfloat16)
    y = a * w  # bf16 multiply, matching PyTorch bf16 elementwise mul

    tl.store(Y_ptr + row * stride_y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x @ self.W0
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return x

        # matmul via cuBLAS (tensor cores)
        h = x @ self.W0  # (M, 4096) bf16
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)

        _softmax_rms_kernel[(Mrows,)](
            h, self.rms2_w, out,
            h.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
