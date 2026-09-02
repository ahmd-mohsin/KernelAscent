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
    n_cols,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X_ptr + row * n_cols + cols, mask=mask, other=float('-inf'))
    # scale (division), rounded through fp16 to match reference fp16 scores tensor
    x = (x.to(tl.float32) / scale).to(tl.float16).to(tl.float32)

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y_ptr + row * n_cols + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_wqkv(self, device):
        w = getattr(self, "_Wqkv_cache", None)
        if w is None or w.device != device or w.data_ptr() == 0:
            w = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous().to(device)
            self._Wqkv_cache = w
        return w

    def forward(self, x):
        d = x.shape[-1]
        x2 = x.contiguous()

        if x2.is_cuda:
            Wqkv = self._get_wqkv(x2.device)
            # one big GEMM instead of three
            qkv = x2 @ Wqkv
            q, k, v = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]

            # raw scores (scaling fused into softmax kernel)
            scores = q @ k.transpose(-1, -2)
            scores = scores.contiguous()

            orig_shape = scores.shape
            n_cols = orig_shape[-1]
            scores_2d = scores.view(-1, n_cols)
            n_rows = scores_2d.shape[0]

            a = torch.empty_like(scores_2d)
            BLOCK = triton.next_power_of_2(n_cols)
            num_warps = 4
            if BLOCK >= 2048:
                num_warps = 8
            if BLOCK >= 8192:
                num_warps = 16

            _scaled_softmax_kernel[(n_rows,)](
                scores_2d, a,
                n_cols,
                math.sqrt(d),
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
            a = a.view(orig_shape)
            return a @ v.contiguous()
        else:
            # CPU fallback: reference path
            q = x @ self.Wq
            k = x @ self.Wk
            v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            a = torch.softmax(scores, dim=-1)
            return a @ v
