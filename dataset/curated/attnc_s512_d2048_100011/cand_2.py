import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100011
S, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _causal_softmax_kernel(
    S_ptr,          # scores, shape (R, N) flattened, R = batch * N
    N,              # number of key columns (= seq len)
    scale,          # 1/sqrt(d)
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    qpos = row % N  # query position within the sequence (handles batching)
    cols = tl.arange(0, BLOCK)
    col_mask = cols < N
    ptr = S_ptr + row.to(tl.int64) * N
    x = tl.load(ptr + cols, mask=col_mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    causal = cols <= qpos
    x = tl.where(causal, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(causal, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(ptr + cols, y.to(S_ptr.dtype.element_ty), mask=col_mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    @torch.no_grad()
    def forward(self, x):
        d = x.shape[-1]

        # Lazily build (and cache) a fused QKV weight so all three projections
        # run as a single large GEMM.
        Wqkv = getattr(self, "_Wqkv_cache", None)
        if (Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype):
            Wqkv = torch.cat(
                [self.Wq, self.Wk, self.Wv], dim=1
            ).to(device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv_cache = Wqkv

        if x.is_cuda:
            qkv = x @ Wqkv
            q, k, v = qkv.split(d, dim=-1)

            # Attention scores (bf16 tensor-core GEMM)
            scores = q @ k.transpose(-1, -2)
            scores = scores.contiguous()

            n = scores.shape[-1]
            rows = scores.numel() // n
            BLOCK = triton.next_power_of_2(n)
            num_warps = 4
            if BLOCK >= 2048:
                num_warps = 8
            if BLOCK >= 8192:
                num_warps = 16

            # Fused: scale + causal mask + softmax, in-place
            _causal_softmax_kernel[(rows,)](
                scores, n, 1.0 / math.sqrt(d), BLOCK=BLOCK, num_warps=num_warps
            )

            return scores @ v
        else:
            q = x @ self.Wq
            k = x @ self.Wk
            v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
            a = torch.softmax(scores, dim=-1)
            return a @ v
