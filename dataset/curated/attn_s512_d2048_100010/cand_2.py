import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100010
S, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _scale_softmax_kernel(
    S_ptr, O_ptr,
    n_cols,
    stride_s, stride_o,
    inv_scale,  # 1.0 not used; we pass the divisor
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(S_ptr + row * stride_s + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # Replicate: scores(bf16) / sqrt(d)  -> computed in fp32, rounded to bf16
    x = x / inv_scale
    x = x.to(tl.bfloat16).to(tl.float32)

    # Softmax in fp32 (matches PyTorch's bf16 softmax accumulation), output bf16
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(O_ptr + row * stride_o + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_wcat(self):
        w = getattr(self, "_wcat", None)
        if (w is None or w.device != self.Wq.device or w.dtype != self.Wq.dtype):
            w = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._wcat = w
        return w

    def forward(self, x):
        d = x.shape[-1]
        orig_shape = x.shape
        x2 = x.reshape(-1, d)

        # Fused QKV projection: single large GEMM instead of three
        wcat = self._get_wcat()
        qkv = x2 @ wcat
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]

        # Attention scores (bf16 GEMM via cuBLAS/tensor cores)
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        n_rows, n_cols = scores.shape
        a = torch.empty_like(scores)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _scale_softmax_kernel[(n_rows,)](
            scores, a,
            n_cols,
            scores.stride(0), a.stride(0),
            math.sqrt(d),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        out = a @ v
        return out.reshape(orig_shape)
