#!/usr/bin/env python3
"""A4: decompose C into CORRECTNESS vs SPEED. For weight-RSI runs, per model report how C_self moved via
correct_rate (did it solve more?) vs compiled_sp (did correct kernels get faster?). Answers the reviewer:
is the gain crossing the correctness threshold, or genuine acceleration?"""
import json, glob, os, statistics
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RAW=os.path.join(ROOT,"results","raw"); OUT=os.path.join(ROOT,"docs","data")
import datetime
def load(pat):
    o=[]
    for f in glob.glob(os.path.join(RAW,pat)):
        try:o.append((os.path.basename(f),json.load(open(f))))
        except:pass
    return o
rows=[]
for fn,d in load("rerun_small_*.json")+load("rerun_mid_*.json")+load("rerun_large_*.json")+load("mech_*.json"):
    h=d.get("history",[])
    if not h: continue
    m=d.get("model",fn).split("/")[-1].replace("-Instruct","")
    c0_cr=d.get("C0_correct_rate") or 0; c0_sp=d.get("C0_compiled_sp") or 0
    fin=h[-1]
    rows.append(dict(model=m,seed=d.get("seed",0),
        C0_correct=round(c0_cr,3), final_correct=round(fin.get("correct_rate_self") or 0,3),
        d_correct=round((fin.get("correct_rate_self") or 0)-c0_cr,3),
        C0_speed=round(c0_sp,3), final_speed=round(fin.get("compiled_sp_self") or 0,3),
        d_speed=round((fin.get("compiled_sp_self") or 0)-c0_sp,3),
        C0=round(d.get("C0_frozen") or 0,3), C_final=round(fin.get("C_self") or 0,3)))
# collapse seeds
agg={}
for r in rows: agg.setdefault(r["model"],[]).append(r)
out=[]
for m,rs in agg.items():
    out.append(dict(model=m,n=len(rs),
        d_correct=round(statistics.mean([r["d_correct"] for r in rs]),3),
        d_speed=round(statistics.mean([r["d_speed"] for r in rs]),3),
        C0=round(statistics.mean([r["C0"] for r in rs]),3),
        C_final=round(statistics.mean([r["C_final"] for r in rs]),3),
        driver="correctness" if abs(statistics.mean([r["d_correct"] for r in rs]))>abs(statistics.mean([r["d_speed"] for r in rs])) else "speed"))
out.sort(key=lambda x:-(x["C_final"]-x["C0"]))
json.dump(dict(updated=datetime.date.today().isoformat(),note="A4: C decomposed into correctness (solved more) vs speed (correct kernels faster). driver = which moved C more.",models=out),open(os.path.join(OUT,"decomp.json"),"w"),indent=2)
for r in out: print(f"  {r['model']:<26} dC={r['C_final']-r['C0']:+.3f} = correctness {r['d_correct']:+.3f} / speed {r['d_speed']:+.3f}  [{r['driver']}]")
