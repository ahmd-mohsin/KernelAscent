import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100034
S, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _scaled_softmax_kernel(
    S_ptr, O_ptr,
    n_cols,
    scale,
    BLOCK: tl.constexpr,
):
    # One program per row: fused (scores / scale) -> softmax, all in one pass.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    s = tl.load(S_ptr + row * n_cols + cols, mask=mask, other=float('-inf')).to(tl.float32)
    # Match reference: bf16 scores divided by python float -> rounded to bf16,
    # then softmax accumulated in fp32.
    s = s / scale
    s = s.to(tl.bfloat16).to(tl.float32)
    s = tl.where(mask, s, float('-inf'))

    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(O_ptr + row * n_cols + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = self.Wq.shape[0]
        scale = math.sqrt(float(d))

        # Fuse the three projections into a single GEMM (weights cached after first call).
        W = getattr(self, "_Wqkv", None)
        if W is None or W.device != x.device or W.dtype != x.dtype:
            W = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv = W

        qkv = x @ W
        q, k, v = qkv.chunk(3, dim=-1)

        # Scores GEMM (bf16 tensor cores via cuBLAS)
        scores = q @ k.transpose(-1, -2)

        if scores.is_cuda and scores.dtype == torch.bfloat16:
            scores = scores.contiguous()
            n_rows, n_cols = scores.shape[-2], scores.shape[-1]
            total_rows = scores.numel() // n_cols
            a = torch.empty_like(scores)
            BLOCK = triton.next_power_of_2(n_cols)
            num_warps = 4
            if BLOCK >= 2048:
                num_warps = 8
            if BLOCK >= 8192:
                num_warps = 16
            _scaled_softmax_kernel[(total_rows,)](
                scores, a, n_cols, scale,
                BLOCK=BLOCK, num_warps=num_warps,
            )
        else:
            a = torch.softmax(scores / scale, dim=-1)

        return a @ v


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]
