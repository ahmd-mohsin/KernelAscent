import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 981
M, D, DT = 1024, 513, torch.bfloat16


@triton.jit
def _fused_softmax_bias_rms_kernel(
    X_ptr, B_ptr, W_ptr, Out_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X_ptr + row * N + offs

    # load logits (bf16) as fp32
    x = tl.load(ptr).to(tl.float32)

    # softmax in fp32 (matching PyTorch's fp32 accumulation for bf16 softmax)
    m = tl.max(x, axis=0)
    e = tl.math.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s
    # round to bf16 (softmax output dtype), back to fp32 for next ops
    p = p.to(tl.bfloat16).to(tl.float32)

    # x * 1.4911  (bf16 op with fp32 opmath -> bf16 round)
    t = (p * 1.4911).to(tl.bfloat16).to(tl.float32)

    # x + b3  (bf16 op with fp32 opmath -> bf16 round)
    b = tl.load(B_ptr + offs).to(tl.float32)
    t = (t + b).to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32 exactly as reference: xf * rsqrt(mean(xf^2) + 1e-6)
    ms = tl.sum(t * t, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    y = (t * r).to(tl.bfloat16).to(tl.float32)

    # * rms4_w (bf16 op -> bf16 round)
    w = tl.load(W_ptr + offs).to(tl.float32)
    y = (y * w).to(tl.bfloat16).to(tl.float32)

    # * 1.1793 (bf16 op -> bf16 round)
    y = (y * 1.1793).to(tl.bfloat16)

    tl.store(Out_ptr + row * N + offs, y)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS/tensor cores (same as reference)
        h = x @ self.W0
        h = h.contiguous()

        rows, N = h.shape
        if h.is_cuda:
            out = torch.empty_like(h)
            _fused_softmax_bias_rms_kernel[(rows,)](
                h, self.b3, self.rms4_w, out,
                N=N, BLOCK=N,
                num_warps=8,
            )
            return out
        else:
            # CPU fallback (reference path)
            h = torch.softmax(h, dim=-1)
            h = h * 1.4911
            h = h + self.b3
            _xf = h.float()
            h = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(h.dtype) * self.rms4_w
            h = h * 1.1793
            return h


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
