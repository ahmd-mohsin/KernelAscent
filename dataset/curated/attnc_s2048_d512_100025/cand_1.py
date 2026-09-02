import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100025
S, D, DT = 2048, 512, torch.float16


@triton.jit
def _causal_softmax_kernel(
    S_ptr, O_ptr,
    n_cols, stride_s, stride_o,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    in_row = cols < n_cols
    causal = cols <= row

    s = tl.load(S_ptr + row * stride_s + cols, mask=causal & in_row,
                other=float('-inf')).to(tl.float32)
    s = s * scale

    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    e = tl.where(causal & in_row, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(O_ptr + row * stride_o + cols, out.to(O_ptr.dtype.element_ty),
             mask=in_row)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Cache fused QKV weight (single GEMM instead of three)
        Wqkv = getattr(self, "_Wqkv", None)
        if Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype:
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(x.device, x.dtype).contiguous()
            self._Wqkv = Wqkv

        d = self.Wq.shape[0]
        qkv = x @ Wqkv                       # (S, 3D) one big tensor-core GEMM
        q, k, v = qkv.split(d, dim=-1)       # views, no copies

        # raw scores via cuBLAS tensor cores
        scores = q @ k.transpose(-1, -2)     # (S, S) fp16
        n = scores.shape[0]

        if scores.is_cuda:
            scores_c = scores if scores.stride(1) == 1 else scores.contiguous()
            BLOCK = triton.next_power_of_2(n)
            num_warps = 4
            if BLOCK >= 1024:
                num_warps = 8
            if BLOCK >= 4096:
                num_warps = 16
            _causal_softmax_kernel[(n,)](
                scores_c, scores_c,
                n, scores_c.stride(0), scores_c.stride(0),
                1.0 / math.sqrt(d),
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
            a = scores_c
        else:
            s = scores.float() / math.sqrt(d)
            s = s + torch.triu(torch.full_like(s, float('-inf')), diagonal=1)
            a = torch.softmax(s, dim=-1).to(scores.dtype)

        return a @ v
