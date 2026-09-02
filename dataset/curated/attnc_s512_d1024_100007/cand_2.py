import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100007
S, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _causal_softmax_kernel(
    S_ptr, O_ptr,
    N, stride_s, stride_o,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    in_bounds = cols < N

    x = tl.load(S_ptr + row * stride_s + cols, mask=in_bounds, other=float('-inf')).to(tl.float32)
    x = x * scale
    # causal mask: columns strictly greater than row are -inf
    x = tl.where(cols <= row, x, float('-inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(O_ptr + row * stride_o + cols, y.to(O_ptr.dtype.element_ty), mask=in_bounds)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = x.shape[-1]
        n = x.shape[-2]

        # Lazily cache fused QKV weight (single big GEMM instead of three)
        wqkv = getattr(self, "_wqkv", None)
        if wqkv is None or wqkv.device != x.device or wqkv.dtype != x.dtype:
            wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=x.device, dtype=x.dtype).contiguous()
            self._wqkv = wqkv

        qkv = x @ wqkv
        q = qkv[..., :d]
        k = qkv[..., d:2 * d]
        v = qkv[..., 2 * d:]

        # attention scores (bf16 tensor-core GEMM)
        scores = torch.matmul(q, k.transpose(-1, -2))

        if scores.is_cuda:
            a = torch.empty_like(scores)
            BLOCK = triton.next_power_of_2(n)
            num_warps = 4 if BLOCK <= 1024 else 8
            _causal_softmax_kernel[(n,)](
                scores, a,
                n, scores.stride(-2), a.stride(-2),
                1.0 / math.sqrt(d),
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
        else:
            s = scores / math.sqrt(d)
            s = s + torch.triu(torch.full_like(s, float('-inf')), diagonal=1)
            a = torch.softmax(s, dim=-1)

        return a @ v


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]
