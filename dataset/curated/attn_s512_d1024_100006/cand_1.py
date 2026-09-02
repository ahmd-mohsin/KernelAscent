import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100006
S, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _scaled_softmax_kernel(
    X_ptr, Y_ptr,
    n_cols,
    stride_x, stride_y,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf'))
    # match reference: divide in bf16 first, then softmax
    x = (x * scale).to(tl.bfloat16)
    xf = x.to(tl.float32)

    row_max = tl.max(xf, axis=0)
    num = tl.exp(xf - row_max)
    denom = tl.sum(num, axis=0)
    y = num / denom

    tl.store(Y_ptr + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Lazily build & cache fused QKV weight (single big GEMM instead of 3)
        Wqkv = getattr(self, "_Wqkv", None)
        if Wqkv is None or Wqkv.device != x.device:
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(x.device).contiguous()
            self._Wqkv = Wqkv

        d = self.Wq.shape[0]
        qkv = x @ Wqkv                     # [S, 3D] one GEMM
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        scores = q @ k.transpose(-1, -2)   # [S, S] bf16 GEMM (tensor cores)

        if scores.is_cuda:
            n_rows, n_cols = scores.shape
            a = torch.empty_like(scores)
            BLOCK = triton.next_power_of_2(n_cols)
            num_warps = 8 if BLOCK >= 512 else 4
            _scaled_softmax_kernel[(n_rows,)](
                scores, a,
                n_cols,
                scores.stride(0), a.stride(0),
                1.0 / math.sqrt(d),
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
        else:
            a = torch.softmax(scores / math.sqrt(d), dim=-1)

        return a @ v
