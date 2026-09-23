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


# Same computation, no triton. If THIS fails, the precheck itself is wrong (bad reference
# contract, wrong dtype, wrong grader call) and says nothing about triton.
TORCH_MODELNEW = """
import torch
import torch.nn as nn

DT = torch.float16


class ModelNew(nn.Module):
    def __init__(self, dt=DT):
        super().__init__()
        self.dt = dt

    def forward(self, x):
        return x * 2.0 + 1.0
"""

# Deliberately wrong. Must FAIL, otherwise the grader passes everything and a PASS above
# would be meaningless.
WRONG_MODELNEW = """
import torch
import torch.nn as nn

DT = torch.float16


class ModelNew(nn.Module):
    def __init__(self, dt=DT):
        super().__init__()
        self.dt = dt

    def forward(self, x):
        return x * 3.0 - 7.0
"""


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

    from kernelascent.v3 import lab_weight_rsi as W

    # Three-point calibration. A single triton FAIL is ambiguous: it could be triton, or my
    # own reference/contract being wrong, or a grader that rejects everything. Only the
    # pattern across all three is diagnostic.
    cases = [("plain-torch  (must PASS)", TORCH_MODELNEW, True),
             ("triton       (the question)", TRITON_MODELNEW, None),
             ("wrong answer (must FAIL)", WRONG_MODELNEW, False)]
    got = {}
    print()
    for label, code, expect in cases:
        try:
            r = W._grade_isolated(REFERENCE, [code])
            ok = bool(r and r[0] and r[0][0])
            sp = (r[0][1] if r and r[0] and len(r[0]) > 1 else 0.0)
        except Exception as e:
            ok, sp, r = False, 0.0, "EXCEPTION %s" % e
        got[label.split()[0]] = ok
        flag = "" if expect is None else ("  <-- UNEXPECTED" if ok != expect else "  ok")
        print("  %-28s -> ok=%-5s speedup=%.2fx%s" % (label, ok, sp, flag))
        if isinstance(r, str) or not r:
            print("       raw: %r" % (r,))

    torch_ok, tri_ok, wrong_ok = got["plain-torch"], got["triton"], got["wrong"]
    print("\n" + "=" * 78)
    if not torch_ok or wrong_ok:
        print("  INCONCLUSIVE -- this precheck is not trustworthy.")
        if not torch_ok:
            print("    a plain-torch kernel that IS the reference did not verify, so the")
            print("    reference/contract in this script is wrong, not triton.")
        if wrong_ok:
            print("    a deliberately wrong kernel verified, so the grader accepts anything.")
        print("    Fix the precheck before drawing any conclusion about the kernel arm.")
        return 2
    if tri_ok:
        print("  PASS -- a hand-written triton kernel VERIFIES through this harness,")
        print("  while a wrong kernel does not. A 0/29 from the kernel-prompt arm is")
        print("  therefore about the MODEL, not the instrument.")
        return 0
    print("  FAIL -- plain-torch verifies and a wrong kernel is rejected, but a CORRECT")
    print("  triton kernel does not verify. The harness cannot grade triton, so the")
    print("  kernel-prompt arm's 0/29 is an INSTRUMENT ARTIFACT and must not be reported")
    print("  as a finding about model capability.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc(); sys.exit(2)
