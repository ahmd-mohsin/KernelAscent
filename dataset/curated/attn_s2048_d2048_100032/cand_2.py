import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100032
S, D, DT = 2048, 2048, torch.float16


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
    base = row.to(tl.int64) * n_cols
    s = tl.load(S_ptr + base + offs, mask=mask, other=float('-inf')).to(tl.float32)
    s = s * scale
    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    denom = tl.sum(e, axis=0)
    out = (e / denom).to(tl.float16)
    tl.store(O_ptr + base + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None

    def forward(self, x):
        if self._Wqkv is None or self._Wqkv.device != x.device:
            # Fuse the three projection matrices into one for a single large GEMM
            self._Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()

        x = x.contiguous()
        d = x.shape[-1]

        # One big GEMM for Q, K, V
        qkv = x @ self._Wqkv
        q, k, v = qkv.split(d, dim=-1)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Attention scores (unscaled); scale is fused into the softmax kernel
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        n_rows = scores.numel() // scores.shape[-1]
        n_cols = scores.shape[-1]
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        # Fused scale + softmax (fp32 math internally, fp16 output), in-place
        _scaled_softmax_kernel[(n_rows,)](
            scores, scores,
            n_cols,
            1.0 / math.sqrt(d),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return scores @ v
