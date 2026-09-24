#!/usr/bin/env python3
"""Catch use-before-definition in the probe scripts without needing a GPU.

`probe_extraction.py` died on a GPU node with UnboundLocalError 20 minutes into the queue,
because an edit inserted `from ... import extract_modelnew as _ex` AFTER the loop that used it.
Python does not complain until the line executes, and that line only executes on hardware.

`compile()` does not catch this either -- it is a runtime binding error. But the *static* check
is cheap: walk each function body, and flag a Name that is loaded before any binding for it in
that scope. That turns a 20-minute cluster round-trip into a local test.
"""
import ast, glob, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def local_use_before_def(fn):
    """Names bound somewhere in this function but LOADED at an earlier line.

    Comprehensions are skipped: `{k: v for k, v in items}` loads `k` before binding it in SOURCE
    order, but a comprehension has its own scope in Python 3, so that is not a use-before-def.
    Six such false positives came out of the first version -- a checker that cries wolf gets
    ignored, which is the failure this repo has already recorded twice.
    """
    # Every construct with its OWN scope is skipped: comprehensions, lambdas and nested
    # functions. A closure parameter (`def gate(x, y)`) or a lambda parameter (`lambda m: ...`)
    # is not a use of the enclosing function's variable, and treating it as one produced six
    # false positives in the first version and four in the second. Scope is the whole question
    # here, so the walker has to respect it.
    OWN_SCOPE = (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
                 ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)
    skip = set()
    for node in ast.walk(fn):
        if node is fn:
            continue
        if isinstance(node, OWN_SCOPE):
            for sub in ast.walk(node):
                skip.add(id(sub))
    bound = {}          # name -> earliest binding line
    loaded = {}         # name -> earliest load line
    for node in ast.walk(fn):
        if id(node) in skip:
            continue
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                bound[node.id] = min(bound.get(node.id, 10**9), node.lineno)
            elif isinstance(node.ctx, ast.Load):
                loaded[node.id] = min(loaded.get(node.id, 10**9), node.lineno)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                nm = (al.asname or al.name).split(".")[0]
                bound[nm] = min(bound.get(nm, 10**9), node.lineno)
    return [(n, loaded[n], bound[n]) for n in bound
            if n in loaded and loaded[n] < bound[n]]


def main():
    bad = []
    for f in sorted(glob.glob(os.path.join(ROOT, "scripts", "*.py"))):
        try:
            tree = ast.parse(open(f).read(), filename=f)
        except SyntaxError as e:
            bad.append((os.path.relpath(f, ROOT), "SyntaxError", e.lineno, 0)); continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for nm, use, dfn in local_use_before_def(node):
                    bad.append((os.path.relpath(f, ROOT), "%s in %s()" % (nm, node.name), use, dfn))
    print("scanned %d scripts" % len(glob.glob(os.path.join(ROOT, "scripts", "*.py"))))
    if bad:
        print("\n  use-before-definition:")
        for f, what, use, dfn in bad:
            print("    %-40s %-28s used line %d, bound line %d" % (f, what, use, dfn))
        return 1
    print("  none -- no local name is loaded before it is bound")
    return 0


if __name__ == "__main__":
    sys.exit(main())
