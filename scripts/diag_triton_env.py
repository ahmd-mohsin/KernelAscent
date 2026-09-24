#!/usr/bin/env python3
"""Why does triton raise FileNotFoundError inside the grader?

The 3-point precheck established the harness rejects a CORRECT triton kernel with
FileNotFoundError(2) while accepting plain-torch and rejecting a wrong answer. That is an
environment fault, not a model result. The usual causes, in order of likelihood:

  1. ptxas / cuobjdump / nvdisasm missing -- triton shells out to the CUDA toolchain
  2. TRITON_CACHE_DIR unwritable (read-only container $HOME, or a quota)
  3. inspect.getsource failing because the module was exec'd from a string rather than a file

This prints the FULL traceback plus the state of each, so the fix is identified rather than
guessed at.
"""
import os, sys, traceback, shutil, tempfile

print("=" * 78)
print("TRITON ENVIRONMENT DIAGNOSTIC")
print("=" * 78)

import torch
print("torch %s  cuda=%s  device=%s" % (torch.__version__, torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"))
import triton
print("triton %s  at %s" % (triton.__version__, os.path.dirname(triton.__file__)))

print("\n--- 1. CUDA toolchain binaries triton shells out to ---")
for tool in ("ptxas", "cuobjdump", "nvdisasm", "nvcc"):
    p = shutil.which(tool)
    print("  %-10s PATH: %s" % (tool, p or "NOT FOUND"))
# triton bundles its own copy under triton/backends/nvidia/bin in 3.x
for root, _dirs, files in os.walk(os.path.dirname(triton.__file__)):
    for f in files:
        if f in ("ptxas", "cuobjdump", "nvdisasm"):
            fp = os.path.join(root, f)
            print("  bundled: %s (executable=%s)" % (fp, os.access(fp, os.X_OK)))

print("\n--- 2. cache dirs ---")
for var, default in (("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache")),
                     ("HOME", None), ("TMPDIR", "/tmp")):
    v = os.environ.get(var, default)
    print("  %-18s = %s" % (var, v))
    if v and var != "HOME":
        try:
            os.makedirs(v, exist_ok=True)
            fd, t = tempfile.mkstemp(dir=v); os.close(fd); os.remove(t)
            print("                     writable: yes")
        except Exception as e:
            print("                     writable: NO -- %r" % (e,))
home = os.path.expanduser("~")
try:
    fd, t = tempfile.mkstemp(dir=home); os.close(fd); os.remove(t)
    print("  $HOME writable: yes")
except Exception as e:
    print("  $HOME writable: NO -- %r" % (e,))

print("\n--- 3. compile a triton kernel, full traceback on failure ---")
import triton.language as tl


@triton.jit
def _double(X, Y, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    tl.store(Y + off, tl.load(X + off, mask=m, other=0.0) * 2.0, mask=m)


try:
    x = torch.randn(4096, device="cuda", dtype=torch.float16)
    y = torch.empty_like(x)
    _double[(16,)](x, y, 4096, BLOCK=256)
    torch.cuda.synchronize()
    print("  RESULT: triton kernel compiled and ran; correct=%s" % torch.allclose(y, x * 2))
    print("\n  => triton itself WORKS here. The FileNotFoundError comes from how the grader")
    print("     loads the candidate, not from the toolchain.")
except Exception:
    print("  RESULT: FAILED\n")
    traceback.print_exc()
    print("\n  => the frame above names the missing file. If it is ptxas, the container needs")
    print("     the CUDA toolchain on PATH; if it is a cache path, set TRITON_CACHE_DIR to a")
    print("     writable location in the job environment.")
