import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100035
S, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _causal_softmax_kernel(
    S_ptr,                 # (M, N) scores, modified in-place
    N,                     # number of columns (= sequence length)
    stride_row,
    scale,                 # 1/sqrt(d)
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    col_mask = offs < N
    ptrs = S_ptr + row * stride_row + offs

    x = tl.load(ptrs, mask=col_mask, other=float('-inf')).to(tl.float32)
    # scale, and round through bf16 to match reference (scores / sqrt(d) in bf16)
    x = x * scale
    x = x.to(tl.bfloat16).to(tl.float32)
    # causal mask: only cols <= row are valid
    x = tl.where(offs <= row, x, float('-inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(offs <= row, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(ptrs, y.to(tl.bfloat16), mask=col_mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = self.Wq.shape[0]

        # Lazily build fused QKV weight (single big GEMM instead of three)
        w_cat = getattr(self, "_w_cat", None)
        if w_cat is None or w_cat.device != x.device or w_cat.dtype != x.dtype:
            w_cat = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=x.device, dtype=x.dtype).contiguous()
            self._w_cat = w_cat

        qkv = x @ w_cat                       # (S, 3D) fused projection
        q, k, v = qkv.split(d, dim=-1)

        # attention scores (bf16 tensor-core GEMM)
        scores = torch.matmul(q, k.transpose(-1, -2)).contiguous()  # (S, S)

        n = scores.shape[-1]
        m_rows = scores.numel() // n
        scale = 1.0 / math.sqrt(q.shape[-1])

        if scores.is_cuda and scores.dtype == torch.bfloat16 and scores.dim() == 2:
            BLOCK = triton.next_power_of_2(n)
            num_warps = 8 if BLOCK >= 2048 else 4
            _causal_softmax_kernel[(m_rows,)](
                scores, n, scores.stride(0), scale,
                BLOCK=BLOCK, num_warps=num_warps,
            )
            a = scores
        else:
            s2 = scores * scale
            s2 = s2 + torch.triu(torch.full_like(s2, float('-inf')), diagonal=1)
            a = torch.softmax(s2, dim=-1)

        return torch.matmul(a, v)
