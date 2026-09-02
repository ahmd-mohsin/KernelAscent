import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100028
S, D, DT = 2048, 1024, torch.float16


@triton.jit
def _softmax_scale_kernel(
    X_ptr, Y_ptr,
    n_cols,
    scale,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y_ptr + row * stride_y + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_fused_w(self, device):
        w = getattr(self, "_Wqkv", None)
        if w is None or w.device != device:
            w = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous().to(device)
            self._Wqkv = w
        return w

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            q = x @ self.Wq
            k = x @ self.Wk
            v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            a = torch.softmax(scores, dim=-1)
            return a @ v

        d = self.Wq.shape[0]
        Wqkv = self._get_fused_w(x.device)

        # Single fused GEMM for Q, K, V projections
        qkv = x @ Wqkv
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        # Attention scores (scale is fused into the softmax kernel)
        scores = q @ k.transpose(-1, -2)

        n_rows, n_cols = scores.shape
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _softmax_scale_kernel[(n_rows,)](
            scores, scores,
            n_cols,
            1.0 / math.sqrt(d),
            scores.stride(0), scores.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return scores @ v
