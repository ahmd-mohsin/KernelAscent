import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100005
S, D, DT = 512, 1024, torch.float16


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
    load_mask = in_bounds & causal

    s = tl.load(S_ptr + row * stride_s + cols, mask=load_mask,
                other=float('-inf')).to(tl.float32)
    s = s * scale

    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    e = tl.where(load_mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(O_ptr + row * stride_o + cols, out.to(tl.float16), mask=in_bounds)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = self.Wq.shape[1]

        # Cache fused QKV weight (single GEMM instead of three)
        wqkv = getattr(self, '_wqkv_cache', None)
        if wqkv is None or wqkv.device != x.device or wqkv.dtype != x.dtype:
            wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(
                device=x.device, dtype=x.dtype).contiguous()
            self._wqkv_cache = wqkv

        qkv = x @ wqkv
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        # Raw attention scores (fp16 GEMM, same as reference q @ k.T)
        scores = torch.matmul(q, k.transpose(-1, -2))

        n = scores.shape[-1]
        scale = 1.0 / math.sqrt(q.shape[-1])
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        a = torch.empty_like(scores)
        _causal_softmax_kernel[(scores.shape[0],)](
            scores, a,
            n, scores.stride(0), a.stride(0),
            scale,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return a @ v
