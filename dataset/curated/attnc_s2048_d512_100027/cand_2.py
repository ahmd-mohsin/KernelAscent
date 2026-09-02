import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100027
S, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _causal_softmax_kernel(
    S_ptr,          # [N, N] bf16 scores, modified in-place
    N,              # number of columns (== rows)
    scale,          # 1/sqrt(head_dim)
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    ptrs = S_ptr + row * N + cols

    s = tl.load(ptrs, mask=mask, other=float('-inf')).to(tl.float32)
    s = s * scale
    # causal mask: cols > row -> -inf
    s = tl.where(cols <= row, s, float('-inf'))

    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(ptrs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_fused_w(self, device, dtype):
        w = getattr(self, "_Wqkv", None)
        if w is None or w.device != device or w.dtype != dtype:
            w = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=device, dtype=dtype).contiguous()
            self._Wqkv = w
        return w

    def forward(self, x):
        d = self.Wq.shape[0]
        scale = 1.0 / math.sqrt(d)

        if not x.is_cuda:
            # CPU fallback (reference path)
            q = x @ self.Wq
            k = x @ self.Wk
            v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) * scale
            scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
            a = torch.softmax(scores, dim=-1)
            return a @ v

        # Fused single QKV matmul (one cuBLAS call instead of three)
        W = self._get_fused_w(x.device, x.dtype)
        qkv = x @ W  # [n, 3d]
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        n = q.shape[0]
        scores = q @ k.transpose(-1, -2)  # [n, n] bf16, tensor-core matmul

        scores_c = scores.contiguous()
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _causal_softmax_kernel[(n,)](
            scores_c, n, scale,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return scores_c @ v
