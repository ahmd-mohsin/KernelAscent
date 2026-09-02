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
    n_cols,
    stride_row,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    col_mask = cols < n_cols

    s = tl.load(S_ptr + row * stride_row + cols, mask=col_mask,
                other=float('-inf')).to(tl.float32)
    s = s * scale
    # causal mask: columns > row get -inf
    s = tl.where(cols <= row, s, float('-inf'))

    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    e = tl.where(cols <= row, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(O_ptr + row * stride_row + cols, out.to(O_ptr.dtype.element_ty),
             mask=col_mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None

    def _get_fused_weight(self, device, dtype):
        w = self._Wqkv
        if w is None or w.device != device or w.dtype != dtype:
            w = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=device, dtype=dtype).contiguous()
            self._Wqkv = w
        return w

    def forward(self, x):
        d = x.shape[-1]
        scale = 1.0 / math.sqrt(d)

        if not x.is_cuda:
            # CPU fallback (reference math)
            q = x @ self.Wq
            k = x @ self.Wk
            v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) * scale
            scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
            a = torch.softmax(scores, dim=-1)
            return a @ v

        # Fused QKV projection: one big GEMM instead of three
        Wqkv = self._get_fused_weight(x.device, x.dtype)
        qkv = x @ Wqkv  # (S, 3D)
        q, k, v = qkv.split(d, dim=-1)

        # Raw scores (tensor-core GEMM, fp32 accumulate)
        scores = torch.matmul(q, k.transpose(-1, -2))  # (S, S)
        scores = scores.contiguous()

        n = scores.shape[-1]
        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _causal_softmax_kernel[(scores.shape[0],)](
            scores, a,
            n,
            scores.stride(0),
            scale,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return a @ v
