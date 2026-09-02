import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100019
S, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _causal_softmax_kernel(
    S_ptr, O_ptr,
    N,              # number of columns (keys)
    SEQ,            # sequence length (rows per batch item)
    scale,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_in_seq = pid % SEQ
    cols = tl.arange(0, BLOCK)
    col_mask = cols < N

    x = tl.load(S_ptr + pid * N + cols, mask=col_mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    # causal mask: keys with index > query index get -inf
    x = tl.where(cols <= row_in_seq, x, float('-inf'))

    m = tl.max(x, 0)
    e = tl.exp(x - m)
    s = tl.sum(e, 0)
    y = e / s

    tl.store(O_ptr + pid * N + cols, y.to(O_ptr.dtype.element_ty), mask=col_mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = x.shape[-1]

        if not x.is_cuda:
            # reference fallback for CPU
            q = x @ self.Wq
            k = x @ self.Wk
            v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)
            scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
            a = torch.softmax(scores, dim=-1)
            return a @ v

        # Fused QKV projection (single GEMM instead of three)
        W = getattr(self, "_Wqkv", None)
        if W is None or W.device != x.device or W.dtype != x.dtype:
            W = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous().to(x.device, x.dtype)
            self._Wqkv = W

        qkv = x @ W
        q, k, v = qkv.split(d, dim=-1)

        # Raw attention scores (scale folded into softmax kernel; /32 is exact anyway)
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        seq = scores.shape[-2]
        n = scores.shape[-1]
        rows = scores.numel() // n

        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _causal_softmax_kernel[(rows,)](
            scores, a, n, seq, 1.0 / math.sqrt(d),
            BLOCK=BLOCK, num_warps=num_warps,
        )

        return a @ v
