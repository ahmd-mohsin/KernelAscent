import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100003
S, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _causal_softmax_kernel(
    S_ptr, O_ptr,
    N, stride_s, stride_o,
    SQRT_D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(S_ptr + row * stride_s + cols, mask=mask, other=0.0)
    # match reference: bf16 scores divided by sqrt(D) (fp32 opmath, rounded to bf16)
    x = (x.to(tl.float32) / SQRT_D).to(tl.bfloat16).to(tl.float32)
    # causal mask: add -inf above the diagonal
    x = tl.where((cols <= row) & mask, x, float('-inf'))

    m = tl.max(x, 0)
    e = tl.exp(x - m)
    s = tl.sum(e, 0)
    y = e / s

    tl.store(O_ptr + row * stride_o + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        d = self.Wq.shape[0]

        # Cache fused QKV weight (single GEMM instead of three)
        Wqkv = getattr(self, '_Wqkv', None)
        if Wqkv is None or Wqkv.device != x.device:
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._Wqkv = Wqkv

        qkv = x @ Wqkv
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        # scores in bf16 (same as reference matmul)
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        n = scores.shape[-1]
        a = torch.empty_like(scores)

        BLOCK = triton.next_power_of_2(n)
        _causal_softmax_kernel[(scores.shape[0],)](
            scores, a,
            n, scores.stride(0), a.stride(0),
            SQRT_D=math.sqrt(d),
            BLOCK=BLOCK,
            num_warps=8,
        )

        return a @ v


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]
