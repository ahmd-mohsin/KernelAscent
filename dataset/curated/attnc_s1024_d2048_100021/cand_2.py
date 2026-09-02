import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100021
S, D, DT = 1024, 2048, torch.float16


@triton.jit
def _causal_scale_softmax_kernel(
    S_ptr, O_ptr,
    n_cols, inv_scale,
    stride_s, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    # causal mask: only columns <= row are valid
    valid = (cols <= row) & (cols < n_cols)

    x = tl.load(S_ptr + row * stride_s + cols, mask=valid, other=float('-inf'))
    # replicate: (scores_fp16 / sqrt(d)) computed in fp32 opmath, rounded to fp16
    x = x.to(tl.float32) / inv_scale
    x = x.to(tl.float16).to(tl.float32)

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(valid, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(O_ptr + row * stride_o + cols, out, mask=cols < n_cols)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None

    def forward(self, x):
        # Cache fused QKV weight (single large GEMM instead of three)
        if (self._Wqkv is None or self._Wqkv.device != x.device
                or self._Wqkv.dtype != self.Wq.dtype):
            self._Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()

        d = self.Wq.shape[1]
        qkv = x @ self._Wqkv
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        # raw scores (fp16 GEMM on tensor cores)
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        n = scores.shape[0]
        n_cols = scores.shape[1]
        a = torch.empty_like(scores)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 1024:
            num_warps = 8
        if BLOCK >= 4096:
            num_warps = 16

        _causal_scale_softmax_kernel[(n,)](
            scores, a,
            n_cols, float(math.sqrt(q.shape[-1])),
            scores.stride(0), a.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return a @ v
