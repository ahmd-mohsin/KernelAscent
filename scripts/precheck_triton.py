#!/usr/bin/env python3
"""Known-good-input gate for the KA_PROMPT=kernel arm.

The kernel-prompt arm returned 0/29 tasks solved against 25/29 for the published prompt. Before
that can be reported as "the model cannot write a working kernel", the harness has to be shown
capable of PASSING one -- otherwise it is the sixth defect again, with a different mask.

So: hand-write a Triton ModelNew that is correct by construction, push it through the real
grader, and require it to verify. If this fails, the 0/29 says nothing about the model.

    python3 scripts/precheck_triton.py            # run on a GPU allocation
"""
import os, sys, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# A deliberately simple fused kernel: RMSNorm-free, elementwise, no reductions. If the harness
# cannot verify THIS, nothing more ambitious will fare better.
TRITON_MODELNEW = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

DT = torch.float16


@triton.jit
def _scale_add(X, OUT, N, ALPHA, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(X + off, mask=m, other=0.0)
    tl.store(OUT + off, x * ALPHA + 1.0, mask=m)


class ModelNew(nn.Module):
    def __init__(self, dt=DT):
        super().__init__()
        self.dt = dt

    def forward(self, x):
        xc = x.contiguous()
        out = torch.empty_like(xc)
        n = xc.numel()
        BLOCK = 1024
        _scale_add[(triton.cdiv(n, BLOCK),)](xc, out, n, 2.0, BLOCK=BLOCK)
        return out
'''

REFERENCE = '''
import torch
import torch.nn as nn

DT = torch.float16


class Model(nn.Module):
    def __init__(self, dt=DT):
        super().__init__()
        self.dt = dt

    def forward(self, x):
        return x * 2.0 + 1.0


def get_inputs():
    return [torch.randn(4096, 1024, device="cuda", dtype=DT)]


def get_init_inputs():
    return []
'''


def main():
    print("=" * 78)
    print("PRECHECK: can this harness verify a KNOWN-GOOD triton kernel?")
    print("=" * 78)

    import torch
    print("  torch %s  cuda=%s" % (torch.__version__, torch.cuda.is_available()))
    if not torch.cuda.is_available():
        print("  FAIL: no GPU visible -- run this on an allocation"); return 2
    try:
        import triton
        print("  triton %s" % triton.__version__)
    except Exception as e:
        print("  FAIL: triton not importable: %s" % e); return 2

    # 1) does @triton.jit survive the grader's module-from-file loading path at all?
    from kernelascent import agent_bench as AB
    try:
        mod = AB.load_module(TRITON_MODELNEW) if hasattr(AB, "load_module") else None
    except Exception:
        mod = None
    if mod is None:
        import re
        fn = [n for n in dir(AB) if "load" in n.lower() and "mod" in n.lower()]
        print("  (module loader probed: %s)" % (fn or "none found"))

    # 2) the real thing: grade it exactly as a candidate is graded
    from kernelascent.v3 import lab_weight_rsi as W
    res = W._grade_isolated(REFERENCE, [TRITON_MODELNEW])
    print("\n  grader returned: %r" % (res,))
    if res and res[0] and res[0][0]:
        print("\n  PASS -- a hand-written triton kernel VERIFIES through this harness.")
        print("  => a 0/29 from the kernel-prompt arm is about the MODEL, not the instrument.")
        return 0
    print("\n  FAIL -- the harness cannot verify a correct triton kernel.")
    print("  => the kernel-prompt arm's 0/29 is an INSTRUMENT artifact and must not be")
    print("     reported as a finding about model capability.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc(); sys.exit(2)
