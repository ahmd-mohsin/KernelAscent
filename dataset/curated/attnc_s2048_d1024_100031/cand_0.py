import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100031
S, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _causal_softmax_kernel(
    X_ptr, O_ptr,
    n_cols,
    scale,
    stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    # causal mask: keep cols <= row
    x = tl.where(cols <= row, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(O_ptr + row * stride_o + cols, y.to(O_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Lazily build fused QKV weight (single GEMM instead of three)
        wqkv = getattr(self, "_Wqkv", None)
        if wqkv is None or wqkv.device != x.device or wqkv.dtype != x.dtype:
            wqkv = torch.cat(
                [self.Wq, self.Wk, self.Wv], dim=1
            ).to(device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv = wqkv

        d = self.Wq.shape[0]
        qkv = x @ wqkv                      # (S, 3D) single GEMM
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        # Raw attention scores via cuBLAS
        scores = q @ k.transpose(-1, -2)    # (S, S) bf16

        n_rows, n_cols = scores.shape
        BLOCK = triton.next_power_of_2(n_cols)
        scale = 1.0 / math.sqrt(d)

        # Fused: scale + causal mask + softmax (fp32 internally), in-place
        _causal_softmax_kernel[(n_rows,)](
            scores, scores,
            n_cols,
            scale,
            scores.stride(0), scores.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )

        return scores @ v
