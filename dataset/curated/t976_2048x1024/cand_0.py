import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 976
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_double_rms_kernel(
    x_ptr, w0_ptr, w2_ptr, out_ptr,
    D: tl.constexpr, SCALE: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- first RMSNorm (fp32 math, cast to fp16, weight mult in fp16) ----
    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    ms0 = tl.sum(x * x, axis=0) / D
    inv0 = tl.rsqrt(ms0 + EPS)
    h = (x * inv0).to(tl.float16)
    w0 = tl.load(w0_ptr + offs, mask=mask, other=0.0)
    h = h * w0

    # ---- scalar multiply (PyTorch computes half*scalar in fp32, rounds to fp16) ----
    h = (h.to(tl.float32) * SCALE).to(tl.float16)

    # ---- second RMSNorm ----
    hf = h.to(tl.float32)
    ms2 = tl.sum(hf * hf, axis=0) / D
    inv2 = tl.rsqrt(ms2 + EPS)
    h2 = (hf * inv2).to(tl.float16)
    w2 = tl.load(w2_ptr + offs, mask=mask, other=0.0)
    h2 = h2 * w2

    tl.store(out_ptr + row * D + offs, h2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        d = x.shape[-1]
        n_rows = x.numel() // d
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_double_rms_kernel[(n_rows,)](
            x, self.rms0_w, self.rms2_w, out,
            D=d, SCALE=1.4509, EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        # cuBLAS fp16 GEMM with fp32 accumulation (matches PyTorch semantics)
        return out @ self.W3
