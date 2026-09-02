import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100034
S, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _scale_softmax_kernel(
    X_ptr, Out_ptr,
    n_cols,
    stride_x, stride_o,
    inv_scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf'))
    # match reference: division performed in bf16, then softmax in fp32
    xs = (x.to(tl.float32) * inv_scale).to(tl.bfloat16).to(tl.float32)
    xs = tl.where(mask, xs, float('-inf'))

    row_max = tl.max(xs, axis=0)
    num = tl.exp(xs - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    y = num / denom

    tl.store(Out_ptr + row * stride_o + cols, y.to(tl.bfloat16), mask=mask)


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
        qkv = x @ Wqkv  # (S, 3D)
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        scores = q @ k.transpose(-1, -2)  # (S, S), bf16

        n_rows, n_cols = scores.shape
        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _scale_softmax_kernel[(n_rows,)](
            scores, a,
            n_cols,
            scores.stride(0), a.stride(0),
            1.0 / math.sqrt(d),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return a @ v
