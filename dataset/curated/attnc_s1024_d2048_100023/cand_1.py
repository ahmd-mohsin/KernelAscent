import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100023
S, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _causal_softmax_kernel(
    S_ptr, O_ptr,
    n_cols, stride_s, stride_o,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    in_bounds = cols < n_cols
    causal = cols <= row

    s = tl.load(S_ptr + row * stride_s + cols, mask=causal & in_bounds,
                other=float('-inf')).to(tl.float32)
    s = s * scale

    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    e = tl.where(causal & in_bounds, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = e / denom

    tl.store(O_ptr + row * stride_o + cols, p.to(O_ptr.dtype.element_ty),
             mask=in_bounds)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = self.Wq.shape[0]

        # Lazily cache fused QKV weight (single big GEMM instead of three)
        Wqkv = getattr(self, "_Wqkv", None)
        if Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype:
            Wqkv = torch.cat(
                [self.Wq, self.Wk, self.Wv], dim=1
            ).to(device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv = Wqkv

        qkv = x @ Wqkv
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        if not x.is_cuda:
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)
            scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
            a = torch.softmax(scores, dim=-1)
            return a @ v

        # Raw (unscaled) attention scores
        scores = q @ k.transpose(-1, -2)  # (n, n) bf16, contiguous
        n = scores.shape[0]

        # Fused: scale + causal mask + softmax in fp32, output bf16
        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _causal_softmax_kernel[(n,)](
            scores, a,
            n, scores.stride(0), a.stride(0),
            1.0 / math.sqrt(d),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return a @ v


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]
