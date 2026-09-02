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
    S_ptr, O_ptr,
    seq_len, scale,
    stride_s, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    col_mask = cols < seq_len

    x = tl.load(S_ptr + row * stride_s + cols, mask=col_mask,
                other=float('-inf')).to(tl.float32)
    x = x * scale
    # causal mask: only cols <= row are valid
    x = tl.where(cols <= row, x, float('-inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(O_ptr + row * stride_o + cols, y.to(tl.bfloat16), mask=col_mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = self.Wq.shape[0]

        # Lazily build & cache fused QKV weight (single big GEMM instead of 3)
        Wqkv = getattr(self, '_Wqkv', None)
        if Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype:
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(
                device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv = Wqkv

        qkv = x @ Wqkv                      # (S, 3D) single GEMM
        q, k, v = qkv.split(d, dim=-1)

        # raw attention scores (S, S) via cuBLAS
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        seq = scores.shape[-1]
        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(seq)
        num_warps = 4 if BLOCK <= 1024 else 8
        _causal_softmax_kernel[(seq,)](
            scores, a,
            seq, 1.0 / math.sqrt(d),
            scores.stride(-2), a.stride(-2),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return a @ v
