import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100022
S, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _scale_softmax_kernel(
    S_ptr, O_ptr,
    scale,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(S_ptr + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(O_ptr + row * N + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = self.Wq.shape[0]

        # Lazily build a fused QKV weight (single big GEMM instead of three)
        Wqkv = getattr(self, "_Wqkv", None)
        if Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype:
            Wqkv = torch.cat(
                [self.Wq, self.Wk, self.Wv], dim=1
            ).to(device=x.device, dtype=x.dtype).contiguous()
            self._Wqkv = Wqkv

        # Fused QKV projection: one GEMM
        qkv = x @ Wqkv
        q = qkv[..., :d]
        k = qkv[..., d:2 * d]
        v = qkv[..., 2 * d:]

        scale = 1.0 / math.sqrt(d)

        if x.is_cuda:
            # Attention scores (cuBLAS GEMM on strided views)
            scores = q @ k.transpose(-1, -2)
            scores = scores.contiguous()
            n = scores.shape[-1]
            rows = scores.numel() // n
            scores_2d = scores.view(rows, n)
            a = torch.empty_like(scores_2d)
            BLOCK = triton.next_power_of_2(n)
            num_warps = 4
            if BLOCK >= 2048:
                num_warps = 8
            if BLOCK >= 8192:
                num_warps = 16
            _scale_softmax_kernel[(rows,)](
                scores_2d, a, scale, n,
                BLOCK=BLOCK, num_warps=num_warps,
            )
            a = a.view(scores.shape)
            return a @ v
        else:
            scores = (q @ k.transpose(-1, -2)) * scale
            a = torch.softmax(scores, dim=-1)
            return a @ v
