import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100003
S, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _causal_softmax_kernel(
    S_ptr, O_ptr,
    n_cols, scale,
    stride_s, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    col_mask = cols < n_cols
    x = tl.load(S_ptr + row * stride_s + cols, mask=col_mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    # causal mask: columns > row -> -inf
    x = tl.where(cols <= row, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(O_ptr + row * stride_o + cols, y.to(O_ptr.dtype.element_ty), mask=col_mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_wqkv(self):
        w = getattr(self, "_Wqkv_cache", None)
        if w is None or w.device != self.Wq.device or w.dtype != self.Wq.dtype:
            w = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._Wqkv_cache = w
        return w

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference math)
            q = x @ self.Wq
            k = x @ self.Wk
            v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
            a = torch.softmax(scores, dim=-1)
            return a @ v

        d = self.Wq.shape[0]
        # Fused QKV projection: one GEMM instead of three
        qkv = x @ self._get_wqkv()
        q = qkv[..., 0 * d:1 * d]
        k = qkv[..., 1 * d:2 * d]
        v = qkv[..., 2 * d:3 * d]

        # Attention scores (unscaled); scaling fused into the softmax kernel
        scores = q @ k.transpose(-1, -2)

        n = scores.shape[-1]
        m = scores.shape[-2]
        scores_2d = scores.reshape(-1, n)
        if not scores_2d.is_contiguous():
            scores_2d = scores_2d.contiguous()

        a = torch.empty_like(scores_2d)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 1024:
            num_warps = 8
        if BLOCK >= 4096:
            num_warps = 16

        _causal_softmax_kernel[(scores_2d.shape[0],)](
            scores_2d, a,
            n, 1.0 / math.sqrt(d),
            scores_2d.stride(0), a.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        a = a.reshape(scores.shape)
        return a @ v
