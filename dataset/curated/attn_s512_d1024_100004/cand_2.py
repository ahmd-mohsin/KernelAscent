import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100004
S, D, DT = 512, 1024, torch.float16


@triton.jit
def _softmax_scale_kernel(
    X_ptr, Y_ptr,
    N,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y_ptr + row * N + offs, y.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_fused_weight(self, device, dtype):
        w = getattr(self, "_Wqkv", None)
        if w is None or w.device != device or w.dtype != dtype:
            w = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=device, dtype=dtype).contiguous()
            self._Wqkv = w
        return w

    def forward(self, x):
        d = x.shape[-1]
        scale = 1.0 / math.sqrt(d)

        if not x.is_cuda:
            # fallback: reference path
            q = x @ self.Wq
            k = x @ self.Wk
            v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) * scale
            a = torch.softmax(scores, dim=-1)
            return a @ v

        # Fused QKV projection: one large GEMM instead of three
        W = self._get_fused_weight(x.device, x.dtype)
        qkv = x @ W
        q, k, v = qkv.split(d, dim=-1)

        # Attention scores (cuBLAS GEMM, contiguous output)
        scores = q @ k.transpose(-1, -2)

        # Fused scale + softmax in one Triton kernel (fp32 accumulation)
        n = scores.shape[-1]
        rows = scores.numel() // n
        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _softmax_scale_kernel[(rows,)](
            scores, a, n, scale,
            BLOCK=BLOCK, num_warps=num_warps,
        )

        return a @ v
