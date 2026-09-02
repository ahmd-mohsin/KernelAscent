import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100008
S, D, DT = 512, 2048, torch.float16


@triton.jit
def _scaled_softmax_kernel(
    X_ptr, Y_ptr,
    N, scale,
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
    tl.store(Y_ptr + row * N + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None

    def forward(self, x):
        # Lazily build fused QKV weight (single big GEMM instead of three)
        if (self._Wqkv is None or self._Wqkv.device != x.device
                or self._Wqkv.dtype != x.dtype):
            self._Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(
                device=x.device, dtype=x.dtype).contiguous()

        d = x.shape[-1]
        qkv = x @ self._Wqkv
        q, k, v = qkv.chunk(3, dim=-1)

        # scores in fp16 via tensor-core GEMM
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        n = scores.shape[-1]
        rows = scores.numel() // n
        a = torch.empty_like(scores)

        if scores.is_cuda:
            BLOCK = triton.next_power_of_2(n)
            num_warps = 4
            if BLOCK >= 2048:
                num_warps = 8
            if BLOCK >= 8192:
                num_warps = 16
            _scaled_softmax_kernel[(rows,)](
                scores, a, n, 1.0 / math.sqrt(d),
                BLOCK=BLOCK, num_warps=num_warps,
            )
        else:
            a = torch.softmax(scores / math.sqrt(d), dim=-1)

        return a @ v


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]
