import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100022
S, D, DT = 1024, 2048, torch.bfloat16


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
    s = tl.load(S_ptr + row * n_cols + offs, mask=mask, other=float('-inf')).to(tl.float32)
    # Match reference: bf16 division by sqrt(d) (rounded to bf16), then fp32 softmax
    s = (s / scale).to(tl.bfloat16).to(tl.float32)
    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom
    tl.store(O_ptr + row * n_cols + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = x.shape[-1]

        # Cache fused QKV weight (single large GEMM instead of three)
        Wqkv = getattr(self, "_Wqkv", None)
        if Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype:
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv = Wqkv

        qkv = x @ Wqkv
        q, k, v = qkv.split(d, dim=-1)

        # scores in bf16 via cuBLAS tensor-core GEMM
        scores = q @ k.transpose(-1, -2)

        if scores.is_cuda:
            orig_shape = scores.shape
            n_cols = orig_shape[-1]
            scores2d = scores.reshape(-1, n_cols).contiguous()
            n_rows = scores2d.shape[0]
            a = torch.empty_like(scores2d)
            BLOCK = triton.next_power_of_2(n_cols)
            num_warps = 4
            if BLOCK >= 2048:
                num_warps = 8
            if BLOCK >= 8192:
                num_warps = 16
            _scaled_softmax_kernel[(n_rows,)](
                scores2d, a, n_cols, math.sqrt(d),
                BLOCK=BLOCK, num_warps=num_warps,
            )
            a = a.reshape(orig_shape)
        else:
            a = torch.softmax(scores / math.sqrt(d), dim=-1)

        return a @ v


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]
