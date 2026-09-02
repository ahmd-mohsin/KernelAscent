import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100001
S, D, DT = 512, 512, torch.float16


@triton.jit
def _causal_softmax_kernel(
    X_ptr, O_ptr,
    n_cols,
    stride_x, stride_o,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    col_mask = cols < n_cols

    x = tl.load(X_ptr + row * stride_x + cols, mask=col_mask,
                other=float('-inf')).to(tl.float32)
    x = x * scale
    # causal mask: columns strictly greater than row -> -inf
    x = tl.where(cols <= row, x, float('-inf'))

    row_max = tl.max(x, axis=0)
    num = tl.exp(x - row_max)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(O_ptr + row * stride_o + cols, out.to(tl.float16), mask=col_mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Lazily build fused QKV weight (single GEMM instead of three)
        Wqkv = getattr(self, "_Wqkv", None)
        if Wqkv is None or Wqkv.device != x.device:
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._Wqkv = Wqkv

        d = self.Wq.shape[0]
        qkv = x @ Wqkv                      # (S, 3D) in one GEMM
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        scores = q @ k.transpose(-1, -2)    # (S, S), unscaled fp16 GEMM

        n = scores.shape[-1]
        m = scores.shape[0]
        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _causal_softmax_kernel[(m,)](
            scores, a,
            n,
            scores.stride(0), a.stride(0),
            1.0 / math.sqrt(d),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return a @ v


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]
