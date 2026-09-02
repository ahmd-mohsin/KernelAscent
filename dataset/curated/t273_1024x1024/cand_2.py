import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 273
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _fused_softmax_double_rms(
    X_ptr, W2_ptr, W3_ptr, Y_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (computed in fp32, rounded to bf16 like PyTorch)
    m = tl.max(x, axis=0)
    e = tl.math.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p_bf = (e / s).to(tl.bfloat16)

    # RMSNorm 1
    pf = p_bf.to(tl.float32)
    ms1 = tl.sum(pf * pf, axis=0) / N
    r1 = tl.math.rsqrt(ms1 + 1e-6)
    w2 = tl.load(W2_ptr + offs, mask=mask, other=0.0)
    a_bf = (pf * r1).to(tl.bfloat16) * w2  # bf16 multiply, matches reference

    # RMSNorm 2
    af = a_bf.to(tl.float32)
    ms2 = tl.sum(af * af, axis=0) / N
    r2 = tl.math.rsqrt(ms2 + 1e-6)
    w3 = tl.load(W3_ptr + offs, mask=mask, other=0.0)
    b_bf = (af * r2).to(tl.bfloat16) * w3

    tl.store(Y_ptr + row * N + offs, b_bf, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x @ self.W0
            x = torch.softmax(x, dim=-1)
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return x

        h = torch.matmul(x, self.W0)  # cuBLAS bf16 GEMM (tensor cores)
        h = h.contiguous()
        rows, N = h.shape[0] * (h.numel() // (h.shape[-1] * h.shape[0])) if h.dim() > 2 else h.shape[0], h.shape[-1]
        h2 = h.view(-1, N)
        rows = h2.shape[0]
        out = torch.empty_like(h2)

        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_double_rms[(rows,)](
            h2, self.rms2_w, self.rms3_w, out,
            N=N, BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 512 else 4,
        )
        return out.view(h.shape)
