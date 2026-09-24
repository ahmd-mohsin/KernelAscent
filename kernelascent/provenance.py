"""Record the environment an artifact was produced in, inside the artifact.

Motivation, from a real and nearly-costly incident: Triton kernels could not verify on this
harness for the whole project because the cluster's `CC` pointed at a host compiler absent in
the container. When that was fixed, every result file produced beforehand became invalid for
any claim about kernels -- and *nothing in those files said so*. A stale local copy of one was
one sentence away from being reported as a fresh result.

JSON has no field for "the grader could not compile Triton when this was written", so we add
one. The cost is a few hundred bytes per artifact; the benefit is that "is this file still
valid?" becomes answerable from the file instead of from memory.

Keep this dependency-light and never let it raise: a provenance helper that crashes a 40-minute
run is worse than no provenance.
"""
import json
import os
import platform
import socket
import subprocess
import time

# Environment variables that change what a number MEANS, not merely where it was written.
_SEMANTIC_ENV = ("KA_SCORE", "KA_PROMPT", "KA_ROOT", "KA_DATA_DIR", "KA_GRADE_GPU",
                 "KA_MAX_STRATEGIES", "KA_ROOF_ARCH", "CC", "CXX")


def _git_commit():
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        sha = out.stdout.strip()
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return (sha + ("-dirty" if dirty else "")) if sha else None
    except Exception:
        return None


def _triton_status():
    """Can this process actually COMPILE with triton? Import success is not enough -- the CC
    defect let `import triton` succeed while every kernel failed at build time."""
    try:
        import triton  # noqa: F401
    except Exception as e:
        return {"import": False, "why": repr(e)[:80]}
    info = {"import": True, "version": getattr(triton, "__version__", "?")}
    try:
        import shutil
        cc = os.environ.get("CC")
        info["cc"] = cc or "(unset -> gcc/clang lookup)"
        # the exact failure mode we hit: CC names a compiler that does not exist here
        info["cc_exists"] = bool(shutil.which(cc)) if cc else bool(shutil.which("gcc") or shutil.which("clang"))
    except Exception:
        pass
    return info


def stamp(extra=None):
    """A dict describing this run's environment. Never raises."""
    p = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "host": socket.gethostname(),
         "git": _git_commit(),
         "python": platform.python_version(),
         "env": {k: os.environ[k] for k in _SEMANTIC_ENV if k in os.environ},
         "triton": _triton_status()}
    try:
        # the RESOLVED roofline arch, not just the env var: a headroom number is meaningless
        # without knowing which card's peak FLOPs normalised it
        from kernelascent import agent_bench as _AB
        p["roof_arch"] = _AB._detect_arch()
        p["roof_peaks"] = _AB._ROOF_SPECS.get(p["roof_arch"])
    except Exception:
        pass
    try:
        import torch
        p["torch"] = torch.__version__
        p["cuda"] = torch.version.cuda
        p["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    if extra:
        try:
            p.update(extra)
        except Exception:
            pass
    return p


def dump(obj, dest, extra=None, **kw):
    """Drop-in for `json.dump(obj, <path-or-file>, ...)` with `_provenance` attached.

    `dest` may be a path OR an already-open file object, because the call sites were converted
    by replacing `json.dump(` with `PROV.dump(` in place and they use both shapes:

        PROV.dump(obj, open(path, "w"), indent=2)     # lab_track_c, lab_compounding
        PROV.dump(obj, path + ".prov.json")           # difficulty_filter

    This shim has now broken the batch twice -- once by not accepting `indent`, once by
    assuming `dest` was a string -- and both times because the test exercised a call I wrote
    rather than the calls that exist. tests/test_provenance_shim.py now extracts the real call
    sites from source and runs each one.
    """
    kw.setdefault("indent", 2)
    is_handle = hasattr(dest, "write")
    try:
        if isinstance(obj, dict):
            obj = dict(obj)
            obj["_provenance"] = stamp(extra)
        elif not is_handle:                     # list payload with a known path -> sidecar
            json.dump({"_provenance": stamp(extra)}, open(str(dest) + ".prov.json", "w"), indent=2)
        else:                                   # list payload, only a handle: use its name
            name = getattr(dest, "name", None)
            if isinstance(name, str):
                json.dump({"_provenance": stamp(extra)}, open(name + ".prov.json", "w"), indent=2)
    except Exception:
        pass
    if is_handle:
        json.dump(obj, dest, **kw)
    else:
        with open(dest, "w") as fh:
            json.dump(obj, fh, **kw)


def is_valid_for_kernels(path):
    """True iff this artifact was produced where triton could actually build.

    Use before reading any kernel-related number out of a stored result.
    """
    try:
        d = json.load(open(path))
    except Exception:
        return None
    p = d.get("_provenance") if isinstance(d, dict) else None
    if not p:
        return None                              # unknown provenance -- treat as suspect
    t = p.get("triton") or {}
    return bool(t.get("import")) and bool(t.get("cc_exists"))
