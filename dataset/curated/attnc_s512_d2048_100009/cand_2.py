import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100009
S, D, DT = 512, 2048, torch.float16


@triton.jit
def _causal_softmax_kernel(
    S_ptr, O_ptr,
    n_cols, scale,
    stride_s, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    in_row = cols < n_cols
    causal = cols <= row

    s = tl.load(S_ptr + row * stride_s + cols, mask=causal, other=float('-inf'))
    # replicate reference: (q@k^T) in fp16, scaled in fp16, then softmax in fp32
    s = (s * scale).to(tl.float16).to(tl.float32)

    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    e = tl.where(causal, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(O_ptr + row * stride_o + cols, out.to(tl.float16), mask=in_row)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Lazily build fused QKV weight (single big GEMM instead of three)
        Wqkv = getattr(self, "_Wqkv", None)
        if Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype:
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv = Wqkv

        d = self.Wq.shape[0]
        qkv = x @ Wqkv                       # (S, 3D) one GEMM
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        scores = q @ k.transpose(-1, -2)     # (S, S) fp16 GEMM (unscaled)
        scores = scores.contiguous()

        n = scores.shape[0]
        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(scores.shape[1])
        _causal_softmax_kernel[(n,)](
            scores, a,
            scores.shape[1], 1.0 / math.sqrt(d),
            scores.stride(0), a.stride(0),
            BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 1024 else 4,
        )

        return a @ v
