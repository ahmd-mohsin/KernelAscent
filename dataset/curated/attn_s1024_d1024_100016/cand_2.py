import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100016
S, D, DT = 1024, 1024, torch.float16


@triton.jit
def _softmax_scale_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    inv_scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(X + row * stride_xm + offs, mask=mask, other=float('-inf'))
    # emulate fp16 scaling (scores / sqrt(d)) exactly: 1/32 is a power of two -> exact
    x = (x.to(tl.float32) * inv_scale)
    x = x.to(tl.float16).to(tl.float32)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride_ym + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_fused_weight(self):
        w = getattr(self, "_Wqkv", None)
        if w is None or w.device != self.Wq.device or w.dtype != self.Wq.dtype:
            w = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._Wqkv = w
        return w

    def forward(self, x):
        d = self.Wq.shape[0]
        Wqkv = self._get_fused_weight()

        # single fused GEMM for the three projections
        qkv = x @ Wqkv
        q = qkv[..., :d]
        k = qkv[..., d:2 * d]
        v = qkv[..., 2 * d:]

        # attention scores (cuBLAS handles the strided views)
        scores = q @ k.transpose(-1, -2)

        # fused scale + softmax in one Triton kernel
        s2 = scores.reshape(-1, scores.shape[-1])
        M, N = s2.shape
        a = torch.empty_like(s2)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK_N >= 2048:
            num_warps = 8
        if BLOCK_N >= 8192:
            num_warps = 16
        _softmax_scale_kernel[(M,)](
            s2, a,
            s2.stride(0), a.stride(0),
            N,
            1.0 / math.sqrt(d),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        a = a.view(scores.shape)

        return a @ v
