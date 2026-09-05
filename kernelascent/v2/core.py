"""KernelAscent v2 core: the executed, mutable improver (U) architecture.

Implements the transition the v2 spec (section 4.2) requires and the current repo lacked:

    (S_{k+1}, U_{k+1}) = Execute_{theta, U_k}(S_k, U_k, H_k, P_k; B_k)

U is a real artifact (Python source + params), content-addressed and snapshotted separately
from S. Each round the controller loads the ACTUAL bytes of the accepted U and executes them;
the executed U may propose edits to both S and U; an accepted U edit is what runs next round.
This is what makes "the improver improved its own improvement procedure" an executable,
auditable claim rather than a saved-but-unused file.

Trust boundaries: the official evaluator, the resource ledger, and the model gateway are NOT
part of the mutable state. U proposes; the controller validates and admits; the ledger charges.
Full OS/Docker sandboxing of executed U is Batch C; Batch A executes our own + fixture U in a
fresh namespace with an interface + public-probe gate and records provenance.
"""
from __future__ import annotations
import os, json, hashlib, time, copy, types, dataclasses
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional


# --------------------------------------------------------------------------------------
# Resource ledger (immutable trust boundary): U can spend, never refund or expand.
# --------------------------------------------------------------------------------------
class BudgetExceeded(Exception):
    pass


@dataclass
class Ledger:
    model_calls: int = 0
    input_tokens: int = 0
    generated_tokens: int = 0
    tool_seconds: float = 0.0
    caps: dict = field(default_factory=lambda: {"model_calls": 10**9, "tool_seconds": 10**9})
    events: list = field(default_factory=list)

    def charge_call(self, in_tok: int = 0, out_tok: int = 0, tag: str = ""):
        self.model_calls += 1
        self.input_tokens += int(in_tok or 0)
        self.generated_tokens += int(out_tok or 0)
        self.events.append({"t": time.time(), "kind": "model_call", "tag": tag, "in": in_tok, "out": out_tok})
        if self.model_calls > self.caps.get("model_calls", 10**9):
            raise BudgetExceeded("model_calls cap %d exceeded" % self.caps["model_calls"])

    def charge_tool(self, seconds: float, tag: str = ""):
        self.tool_seconds += float(seconds)
        self.events.append({"t": time.time(), "kind": "tool", "tag": tag, "sec": seconds})
        if self.tool_seconds > self.caps.get("tool_seconds", 10**9):
            raise BudgetExceeded("tool_seconds cap exceeded")

    def snapshot(self):
        return {"model_calls": self.model_calls, "input_tokens": self.input_tokens,
                "generated_tokens": self.generated_tokens, "tool_seconds": round(self.tool_seconds, 3)}


# --------------------------------------------------------------------------------------
# State: S (solver) and U (improver). Both content-addressed. U carries executable source.
# --------------------------------------------------------------------------------------
@dataclass
class SolverState:
    prompt_policy: str = "base"
    retrieval_k: int = 3
    skills: list = field(default_factory=list)      # verified reusable blocks (S memory)
    params: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "SolverState":
        return SolverState(**json.loads(s))


@dataclass
class ImproverState:
    """The improver is executable: `source` defines improve_step(ctx)->StateUpdate, and
    `params` are tunable knobs the source reads. The model may edit either or both."""
    source: str                                     # python source defining improve_step
    params: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"source": self.source, "params": self.params}, sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "ImproverState":
        d = json.loads(s); return ImproverState(source=d["source"], params=d.get("params", {}))


@dataclass
class StateUpdate:
    """What an executed U returns: proposed edits to S and/or U, plus provenance."""
    s_skills_add: list = field(default_factory=list)
    s_param_edits: dict = field(default_factory=dict)
    u_param_edits: dict = field(default_factory=dict)
    u_new_source: Optional[str] = None
    notes: str = ""


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# Content-addressed store + provenance ledger (snapshot / load / fork).
# --------------------------------------------------------------------------------------
class StateStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(os.path.join(root, "objects"), exist_ok=True)
        self.prov_path = os.path.join(root, "provenance.jsonl")

    def put(self, text: str) -> str:
        h = _sha(text.encode())
        p = os.path.join(self.root, "objects", h + ".txt")
        if not os.path.exists(p):
            open(p, "w").write(text)
        return h

    def get(self, h: str) -> str:
        return open(os.path.join(self.root, "objects", h + ".txt")).read()

    def snapshot(self, S: SolverState, U: ImproverState) -> dict:
        return {"S": self.put(S.to_json()), "U": self.put(U.to_json())}

    def load_S(self, h: str) -> SolverState:
        return SolverState.from_json(self.get(h))

    def load_U(self, h: str) -> ImproverState:
        return ImproverState.from_json(self.get(h))

    def record(self, ev: dict):
        ev = dict(ev, ts=time.time())
        open(self.prov_path, "a").write(json.dumps(ev) + "\n")

    def fork(self, checkpoint: dict, intervention: Optional[Callable[["SolverState", "ImproverState"], tuple]] = None):
        """Return (S, U) copies from a checkpoint, optionally transformed by an intervention
        (used by the ancestry keep/revert/rescue branches). Pure: does not mutate the store."""
        S = self.load_S(checkpoint["S"]); U = self.load_U(checkpoint["U"])
        if intervention is not None:
            S, U = intervention(copy.deepcopy(S), copy.deepcopy(U))
        return S, U


# --------------------------------------------------------------------------------------
# Executing U safely enough for Batch A: import its source in a fresh namespace, require the
# improve_step interface. (OS/Docker sandbox is Batch C.)
# --------------------------------------------------------------------------------------
class ImproverInterfaceError(Exception):
    pass


def load_improver_callable(source: str, inject: Optional[dict] = None) -> Callable:
    ns: dict = {"StateUpdate": StateUpdate, "ImproveContext": ImproveContext}
    if inject:
        ns.update(inject)
    try:
        exec(compile(source, "<improver>", "exec"), ns)
    except Exception as e:
        raise ImproverInterfaceError("U source failed to compile/exec: %r" % e)
    fn = ns.get("improve_step")
    if not callable(fn):
        raise ImproverInterfaceError("U source defines no callable improve_step")
    return fn


@dataclass
class ImproveContext:
    """Everything an executed U is allowed to see. No private grader, no final scores."""
    S: SolverState
    U_params: dict
    history: list
    practice_tasks: list
    model_rpc: Callable          # (prompt)->text; charges the ledger internally
    dev_tools: dict              # e.g. {"solve": fn, "grade_public": fn, "bank_skill": fn}
    ledger: Ledger
    round: int


class Controller:
    """Owns the Execute transition and the trust boundaries. U proposes; controller admits."""

    def __init__(self, store: StateStore, u_probe: Callable[[ImproverState], bool]):
        self.store = store
        self.u_probe = u_probe                 # fixed public probe an updated U must pass

    def execute_round(self, S: SolverState, U: ImproverState, ctx_factory: Callable[[SolverState, ImproverState], ImproveContext],
                      k: int) -> tuple:
        """Load and EXECUTE the actual bytes of U, apply admitted S/U edits, return (S', U',
        update, meta). The returned U' is what the NEXT round executes."""
        u_hash_executed = self.store.put(U.to_json())
        fn = load_improver_callable(U.source)                 # execute the accepted U's bytes
        ctx = ctx_factory(S, U)
        update: StateUpdate = fn(ctx)
        if not isinstance(update, StateUpdate):
            raise ImproverInterfaceError("improve_step must return StateUpdate, got %r" % type(update))

        # ---- admit S edits ----
        S2 = copy.deepcopy(S)
        S2.skills = (S2.skills + list(update.s_skills_add))[:64]
        S2.params.update(update.s_param_edits or {})
        if "retrieval_k" in (update.s_param_edits or {}):
            S2.retrieval_k = int(update.s_param_edits["retrieval_k"])

        # ---- admit U edits (params and/or full new source), gated by the fixed public probe ----
        U2 = ImproverState(source=U.source, params=dict(U.params))
        U2.params.update(update.u_param_edits or {})
        u_edit_kind = "params" if update.u_param_edits else "none"
        if update.u_new_source is not None:
            candidate = ImproverState(source=update.u_new_source, params=U2.params)
            ok = False
            try:
                load_improver_callable(candidate.source)      # must define the interface
                ok = self.u_probe(candidate)                  # must pass the fixed public probe
            except Exception:
                ok = False
            if ok:
                U2 = candidate; u_edit_kind = "source"
            else:
                u_edit_kind = "source_rejected"

        cp_before = {"S": self.store.put(S.to_json()), "U": u_hash_executed}
        cp_after = self.store.snapshot(S2, U2)
        meta = {"round": k, "u_hash_executed": u_hash_executed,
                "u_edit_kind": u_edit_kind, "u_changed": cp_after["U"] != u_hash_executed,
                "s_changed": cp_after["S"] != cp_before["S"],
                "skills_added": len(update.s_skills_add), "notes": update.notes[:200]}
        self.store.record({"kind": "execute_round", **meta, "cp_before": cp_before, "cp_after": cp_after})
        return S2, U2, update, meta
