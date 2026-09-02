import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100030
S, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _scaled_softmax_kernel(
    S_ptr, O_ptr,
    n_cols,
    stride_s, stride_o,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    s = tl.load(S_ptr + row * stride_s + offs, mask=mask, other=float('-inf')).to(tl.float32)
    # scale is an exact power of two (1/32), so fp32 multiply matches bf16 division exactly
    s = s * scale

    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(O_ptr + row * stride_o + offs, out.to(O_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = self.Wq.shape[0]

        # Fused QKV projection: one large GEMM instead of three
        Wqkv = getattr(self, "_Wqkv", None)
        if Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != self.Wq.dtype:
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._Wqkv = Wqkv

        qkv = x @ Wqkv
        q, k, v = qkv.split(d, dim=-1)

        scores = q @ k.transpose(-1, -2)

        scale = 1.0 / math.sqrt(q.shape[-1])

        if scores.is_cuda:
            n_cols = scores.shape[-1]
            scores_2d = scores.reshape(-1, n_cols)
            if not scores_2d.is_contiguous():
                scores_2d = scores_2d.contiguous()
            a = torch.empty_like(scores_2d)
            n_rows = scores_2d.shape[0]
            BLOCK = triton.next_power_of_2(n_cols)
            num_warps = 4
            if BLOCK >= 2048:
                num_warps = 8
            if BLOCK >= 8192:
                num_warps = 16
            _scaled_softmax_kernel[(n_rows,)](
                scores_2d, a,
                n_cols,
                scores_2d.stride(0), a.stride(0),
                scale,
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
            a = a.reshape(scores.shape)
        else:
            a = torch.softmax(scores * scale, dim=-1)

        return a @ v
