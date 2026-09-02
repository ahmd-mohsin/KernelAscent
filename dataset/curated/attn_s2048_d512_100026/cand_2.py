import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100026
S, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _scaled_softmax_kernel(
    S_ptr, O_ptr,
    n_cols,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(S_ptr + row * n_cols + offs, mask=mask, other=float('-inf'))
    # Match reference: scores (bf16) divided by sqrt(D) -> rounded to bf16,
    # then softmax computed in fp32, output rounded to bf16.
    x = (x.to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(O_ptr + row * n_cols + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None  # lazily cached fused projection weight

    def forward(self, x):
        d = self.Wq.shape[0]

        if x.is_cuda:
            # Fuse the three projections into a single GEMM (identical per-element
            # accumulation order over the K dimension -> identical results).
            if (self._Wqkv is None or self._Wqkv.device != x.device
                    or self._Wqkv.dtype != x.dtype):
                self._Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()

            qkv = x @ self._Wqkv
            q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]

            scores = q @ k.transpose(-1, -2)  # bf16 GEMM w/ fp32 accumulate (cuBLAS)
            scores = scores.contiguous()

            n_rows, n_cols = scores.shape
            a = torch.empty_like(scores)
            BLOCK = triton.next_power_of_2(n_cols)
            num_warps = 8 if BLOCK >= 2048 else 4
            _scaled_softmax_kernel[(n_rows,)](
                scores, a, n_cols,
                1.0 / math.sqrt(d),
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
            return a @ v

        # CPU / fallback path (reference implementation)
        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        a = torch.softmax(scores, dim=-1)
        return a @ v
