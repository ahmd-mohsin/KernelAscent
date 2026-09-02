import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100008
S, D, DT = 512, 2048, torch.float16


@triton.jit
def _scaled_softmax_kernel(
    X_ptr, Y_ptr,
    sqrt_d,
    N,
    stride_xm, stride_ym,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_xm + offs, mask=mask, other=float('-inf'))
    # replicate: scores(fp16) / sqrt(d) -> fp16 rounding, then softmax in fp32
    x = (x.to(tl.float32) / sqrt_d).to(tl.float16).to(tl.float32)

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y_ptr + row * stride_ym + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = x.shape[-1]

        # Cache fused QKV weight (single big GEMM instead of three)
        W = getattr(self, '_Wqkv', None)
        if W is None or W.device != x.device or W.dtype != x.dtype:
            W = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._Wqkv = W

        qkv = x @ W
        q, k, v = qkv.split(d, dim=-1)

        # scores in fp16 (fp32 accumulate inside GEMM), same as reference q @ k^T
        scores = q @ k.transpose(-1, -2)

        s = scores.shape[-1]
        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(s)
        num_warps = 4 if BLOCK <= 1024 else 8
        _scaled_softmax_kernel[(scores.shape[0],)](
            scores, a,
            math.sqrt(d),
            s,
            scores.stride(0), a.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return a @ v
