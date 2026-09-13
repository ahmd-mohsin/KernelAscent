#!/usr/bin/env python3
"""Aggregate every collected result JSON in results/raw/ into the website board JSONs (docs/data/).
Idempotent — run repeatedly by the orchestrator. Handles the definitive re-run (rerun_*), the earlier runs,
combined (T4), track-c (T3), baselines, roofline. Applies the pre-registered decision rules.
"""
import json, glob, os, re, datetime, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")
OUT = os.path.join(ROOT, "docs", "data")
today = datetime.date.today().isoformat()

def last(h, k, d=None):
    for r in reversed(h):
        if r.get(k) is not None: return r[k]
    return d
def peak(h, k):
    vs = [r.get(k) for r in h if r.get(k) is not None]; return max(vs) if vs else None

RES = {"fable": "Fable-5.1", "nova": "Nova-Pro", "llama70": "Llama-3.3-70B", "gpt56": "GPT-5.6",
       "kimi": "Kimi-K2.5", "opus5": "Claude-Opus-5", "deepseekv32": "DeepSeek-v3.2"}
TRN = {"qwen1p5": "Qwen2.5-Coder-1.5B", "deepseek1p3": "DeepSeek-Coder-1.3B", "qwen0p5": "Qwen2.5-Coder-0.5B",
       "opencoder1p5": "OpenCoder-1.5B", "qwen7b": "Qwen2.5-Coder-7B", "qwen14b": "Qwen2.5-Coder-14B"}

def tier_of(m):
    ml = m.lower()
    if any(s in ml for s in ["14b", "15b", "13b"]): return "large"
    if any(s in ml for s in ["7b", "6.7b", "8b", "9b"]): return "mid"
    return "small"

def load(pat):
    out = []
    for f in glob.glob(os.path.join(RAW, pat)):
        try: out.append((os.path.basename(f), json.load(open(f))))
        except Exception: pass
    return out

# ---- T2 weight-RSI per-scale (prefer rerun_*, fall back to wrsi_*) ----
def weight_rsi():
    groups = {}  # base model name -> list of (seed, history, C0)
    for fn, d in load("rerun_*.json") + load("mech_*.json") + load("wrsi_*.json"):
        if "C0_frozen" not in d: continue          # weight-RSI only (excludes comb/trackc which are also rerun_*)
        h = d.get("history", [])
        if not h: continue
        m = d.get("model", fn); short = m.split("/")[-1].replace("-Instruct", "")
        groups.setdefault(short, []).append((d.get("seed", 0), h, d.get("C0_frozen") or 0))
    rows = []
    for short, runs in groups.items():
        finals = [last(h, "delta_self_minus_fresh", 0) or 0 for _, h, _ in runs]
        last2 = [statistics.mean([x for x in [r.get("delta_self_minus_fresh") for r in h[-2:]] if x is not None] or [0]) for _, h, _ in runs]
        csf = statistics.mean([last(h, "C_self", 0) or 0 for _, h, _ in runs])
        c0 = statistics.mean([c for _, _, c in runs])
        mean_final = statistics.mean(finals); n = len(runs)
        sign_consistent = all(x > 0 for x in finals) or all(x < 0 for x in finals)
        holds = (statistics.mean(last2) > 0.05) and sign_consistent
        verdict = "compounds" if holds else ("overfits" if mean_final <= -0.02 else ("gains" if csf > c0 + 0.05 else "flat"))
        rows.append(dict(model=short, tier=tier_of(short), n_seeds=n, C0=round(c0, 3), C_self=round(csf, 3),
                         self_minus_fresh=round(mean_final, 3), seeds=[round(x, 3) for x in finals],
                         sign_consistent=sign_consistent, verdict=verdict))
    order = {"small": 0, "mid": 1, "large": 2}
    rows.sort(key=lambda x: (order[x["tier"]], -x["self_minus_fresh"]))
    if rows:
        json.dump(dict(updated=today, metric="speed vs torch.compile, size-matched SOTA bank, headroom-normalized, self vs fresh-frozen (multi-seed)", models=rows),
                  open(os.path.join(OUT, "tier_speed_rsi.json"), "w"), indent=2)
        json.dump(dict(updated=today, note="cross-seed self-vs-fresh; holds = mean(last2)>0.05 AND sign-consistent.", models=rows),
                  open(os.path.join(OUT, "rigor.json"), "w"), indent=2)
    return len(rows)

# ---- T4 combined (prefer rerun_comb_*, fall back to comb_*) ----
def combined():
    rows = []
    seen = set()
    for fn, d in load("rerun_comb_*.json") + load("comb_*.json"):
        h = d.get("history", [])
        if not h: continue
        tag = re.sub(r"^(rerun_)?comb_", "", fn)[:-5]
        rk = tag.split("2")[0].rstrip("_"); tk = tag.split("2")[-1]
        key = (rk, tk)
        if key in seen: continue
        seen.add(key)
        dl = last(h, "delta_improved_minus_frozen", 0) or 0; pk = peak(h, "delta_improved_minus_frozen") or 0
        mean_dl = statistics.mean([r.get("delta_improved_minus_frozen") or 0 for r in h])
        v = "harness helps" if (max(dl, pk) >= 0.08 and mean_dl > 0) else ("hurts" if dl <= -0.05 else "flat")
        rows.append(dict(trainee=TRN.get(tk, tk), researcher=RES.get(rk, rk), rounds=len(h),
                         C0=round(d.get("C0") or 0, 3), C_improved=round(last(h, "C_improved") or 0, 3),
                         C_frozen=round(last(h, "C_frozen") or 0, 3), impr_minus_frozen=round(dl, 3),
                         peak=round(pk, 3), verdict=v))
    rows.sort(key=lambda x: -x["peak"])
    if rows:
        json.dump(dict(updated=today, note="closed researcher rewrites the training harness; improved-vs-frozen on the open trainee (headroom score, SOTA bank).", models=rows),
                  open(os.path.join(OUT, "combined_rsi.json"), "w"), indent=2)
    return len(rows)

# ---- T3 track-c ----
def trackc():
    rows = []; seen = set()
    for fn, d in load("rerun_trackc_*.json") + load("trackc_*.json"):
        h = d.get("history", [])
        if not h: continue
        tag = re.sub(r"^(rerun_)?trackc_", "", fn)[:-5]; mk = tag.split("_")[0]
        if mk in seen: continue
        seen.add(mk)
        dv = last(h, "delta_vs_base", 0) or 0
        rows.append(dict(model=RES.get(mk, d.get("model", mk)), mode=d.get("mode", "self-modify"), rounds=len(h),
                         Q0=round(d.get("Q0") or 0, 3), Qg=round(last(h, "Qg") or 0, 3), delta_vs_base=round(dv, 3),
                         verdict="improves" if dv >= 0.03 else ("degrades" if dv <= -0.03 else "flat")))
    rows.sort(key=lambda x: -x["delta_vs_base"])
    if rows:
        json.dump(dict(updated=today, note="procedure RSI: model rewrites its own executable procedure; Qg vs frozen procedure.", models=rows),
                  open(os.path.join(OUT, "trackc.json"), "w"), indent=2)
    return len(rows)

# ---- mechanistic dense curves (mech_* 12-round runs) ----
def mech():
    rows = []
    for fn, d in load("mech_*.json"):
        h = d.get("history", [])
        if not h: continue
        m = d.get("model", fn); short = m.split("/")[-1].replace("-Instruct", "")
        sf = [r.get("delta_self_minus_fresh") for r in h]
        rows.append(dict(model=short, tier=tier_of(short), rounds=[r["round"] for r in h],
            C_self=[r.get("C_self") for r in h], diversity=[r.get("diversity_self") for r in h],
            retention=[r.get("retention") for r in h], self_minus_fresh=sf,
            correct_rate=[r.get("correct_rate_self") for r in h]))
    order = {"small": 0, "mid": 1, "large": 2}
    rows.sort(key=lambda x: order[x["tier"]])
    if rows:
        json.dump(dict(updated=today, note="Mechanism: per-round trajectories over 12 rounds. Compounders sustain self-data DIVERSITY and RETENTION; overfitters show diversity collapse + retention decline as self-minus-fresh goes negative.", models=rows),
                  open(os.path.join(OUT, "mech.json"), "w"), indent=2)
    return len(rows)

def fork():
    rows=[]
    for fn,d in load("fork_*.json"):
        h=d.get("history",[])
        if not h: continue
        m=d.get("model",fn).split("/")[-1].replace("-Instruct","").replace("-Chat","")
        fr=d.get("fork_round",4)
        smc=[r.get("self_minus_ckpt") for r in h if r.get("self_minus_ckpt") is not None]
        rows.append(dict(model=m,fork_round=fr,rounds=[r["round"] for r in h],
            C_self=[r.get("C_self") for r in h], C_ckpt_frozen=[r.get("C_ckpt_frozen") for r in h],
            C_base_frozen=[r.get("C_base_frozen") for r in h], self_minus_ckpt=[r.get("self_minus_ckpt") for r in h],
            post_fork_mean_self_minus_ckpt=round(statistics.mean(smc),3) if smc else None,
            verdict=("genuine recursion" if (len(smc)>=2 and statistics.mean(smc[-2:])>0.03) else "one-time upgrade / iterative")))
    if rows:
        json.dump(dict(updated=today,note="A1 recursion-interruption fork: continued-self vs checkpoint-frozen vs base-frozen producers at equal budget. self_minus_ckpt>0 sustained => genuine compounding, not a one-time producer upgrade.",models=rows),open(os.path.join(OUT,"fork.json"),"w"),indent=2)
    return len(rows)

def open_rsi():
    rows=[]
    for fn,d in load("open_*.json"):
        h=d.get("history",[])
        if not h: continue
        m=d.get("model",fn).split("/")[-1].replace("-Instruct","").replace("-Chat","")
        dl=[r.get("delta_open_minus_fixed") for r in h]
        last2=statistics.mean([x for x in dl[-2:] if x is not None] or [0])
        rows.append(dict(model=m,tier=tier_of(m),rounds=[r["round"] for r in h],
            C_held_open=[r.get("C_held_open") for r in h], C_held_fixed=[r.get("C_held_fixed") for r in h],
            delta_open_minus_fixed=dl, frontier=[r.get("frontier") for r in h],
            base_correct_on_frontier=[r.get("base_correct_on_frontier") for r in h],
            final_delta=round(dl[-1] if dl and dl[-1] is not None else 0,3),
            verdict=("compounds (open>fixed)" if (len(dl)>=4 and last2>0.05) else "one-time (open==fixed)")))
    rows.sort(key=lambda x:-x["final_delta"])
    if rows:
        json.dump(dict(updated=today,note="Task 5 open-ended RSI (rigorous): OPEN(escalating frontier) vs FIXED(static) learners at equal budget, same held ladder. PRIMARY=delta_open_minus_fixed; sustained >0 = open-endedness causes compounding. base_correct_on_frontier declining verifies escalation is real.",models=rows),open(os.path.join(OUT,"open_rsi.json"),"w"),indent=2)
    return len(rows)

def _selfplay_3arm(patterns, outname, note, propkey, primary="L_minus_F"):
    """3-arm self-play (open sp_* or closed spc_*): PRIMARY = L-F (author co-evolution). Falls back to old 2-arm."""
    rows=[]
    seen=set()
    for pat in patterns:
        for fn,d in load(pat):
            h=d.get("history",[])
            if not h: continue
            m=d.get("model",fn).split("/")[-1].replace("-Instruct","").replace("-Chat","")
            key=(m,fn)
            if key in seen: continue
            seen.add(key)
            lf=[r.get(primary) for r in h]; ls=[r.get("L_minus_S") for r in h]; fs=[r.get("F_minus_S") for r in h]
            mp=sum(r.get(propkey,0) or 0 for r in h)
            if not any(x is not None for x in lf):        # legacy 2-arm fallback
                lf=[r.get("delta_selfplay_minus_static") for r in h]; mp=sum(r.get("model_proposed",0) or 0 for r in h)
            last2=statistics.mean([x for x in lf[-2:] if x is not None] or [0])
            hi = "C_held_live" if any("C_held_live" in r for r in h) else "Q_held_live"
            rows.append(dict(model=m,tier=tier_of(m),rounds=[r["round"] for r in h],
                C_held_live=[r.get(hi) for r in h],
                C_held_frozen_author=[r.get("C_held_frozen_author",r.get("Q_held_frozen_author")) for r in h],
                C_held_static=[r.get("C_held_static",r.get("Q_held_static")) for r in h],
                L_minus_F=lf, L_minus_S=ls, F_minus_S=fs,
                frontier=[r.get("frontier_L",r.get("frontier")) for r in h],
                model_proposed=[r.get(propkey,r.get("model_proposed")) for r in h], total_model_proposed=mp,
                final_L_minus_F=round(lf[-1] if lf and lf[-1] is not None else 0,3),
                verdict=("author CO-EVOLUTION compounds (L>F)" if (len(lf)>=3 and last2>0.05 and mp>0)
                         else "adaptive-curriculum only (L==F)" if (len(lf)>=3 and mp>0)
                         else "no self-referential compounding")))
    rows.sort(key=lambda x:-x["final_L_minus_F"])
    if rows:
        json.dump(dict(updated=today,note=note,models=rows),open(os.path.join(OUT,outname),"w"),indent=2)
    return len(rows)

def selfplay():
    return _selfplay_3arm(["selfplay_*.json","sp_*.json"],"selfplay.json",
        "Task 5a OPEN self-play, 3-arm: STATIC / FROZEN-AUTHOR / LIVE-AUTHOR. PRIMARY=L-F (author CO-EVOLUTION, the self-referential signal); L-S=total adaptive-curriculum benefit, F-S=benefit w/o updating the author. Sustained L-F>0 with model_proposed>0 = genuine recursive compounding, not just an adaptive curriculum.",
        "live_model_proposed")

def selfplay_closed():
    return _selfplay_3arm(["spc_*.json"],"selfplay_closed.json",
        "Task 5b CLOSED (API) self-play, 3-arm: co-evolution channel is the PROCEDURE. STATIC / FROZEN-AUTHOR (empty-procedure author) / LIVE-AUTHOR (co-evolved-procedure author). PRIMARY=L-F. Sustained L-F>0 with model_proposed>0 = closed-model self-referential RSI without weight access.",
        "live_model_proposed")

def rsi_mech():
    """WHY-RSI mechanism probes: per-round diversity/drift/retention/transfer + attributed verdict."""
    rows=[]
    for fn,d in load("rmech_*.json"):
        h=d.get("history",[]); v=d.get("verdict",{})
        if not h: continue
        m=d.get("model",fn).split("/")[-1].replace("-Instruct","").replace("-Chat","")
        rows.append(dict(model=m,tier=tier_of(m),rounds=[r["round"] for r in h],
            C_held=[r.get("C_held") for r in h], C_train=[r.get("C_train") for r in h],
            transfer_gap=[r.get("transfer_gap") for r in h], retention=[r.get("retention") for r in h],
            gen_distinct2=[r.get("gen_distinct2") for r in h], gen_dissim=[r.get("gen_dissim") for r in h],
            gen_entropy=[r.get("gen_entropy") for r in h], drift_total=[r.get("drift_total") for r in h],
            drift_early=[r.get("drift_early") for r in h], drift_mid=[r.get("drift_mid") for r in h],
            drift_late=[r.get("drift_late") for r in h], sft_loss=[r.get("sft_loss") for r in h],
            compounded=v.get("compounded"), mechanisms=v.get("mechanisms",[]), carrying_depth=v.get("carrying_depth")))
    rows.sort(key=lambda x:(not x.get("compounded"), x["model"]))
    if rows:
        json.dump(dict(updated=today,note="WHY RSI fails or HOW it passes. Per-round mechanism trajectories: gen_distinct2/gen_dissim (mode collapse), gen_entropy (over-confidence), drift_* (LoRA change per depth; ->0 = ceiling), retention (forgetting), transfer_gap (memorization). Verdict attributes each outcome to a mechanism {diversity_collapse|forgetting|drift_saturation|no_transfer|no_headroom} or PASS + carrying depth.",models=rows),open(os.path.join(OUT,"rsi_mech.json"),"w"),indent=2)
    return len(rows)

if __name__ == "__main__":
    print("aggregated: tier=%d combined=%d trackc=%d mech=%d fork=%d open=%d selfplay=%d closed=%d rsimech=%d" % (
        weight_rsi(), combined(), trackc(), mech(), fork(), open_rsi(), selfplay(), selfplay_closed(), rsi_mech()))
