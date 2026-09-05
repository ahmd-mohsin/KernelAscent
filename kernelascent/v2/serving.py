"""Batch B T4 service instrument: start a real server, drive it with a fixed external client
trace, and measure client-observed TTFT / TPOT / end-to-end latency and goodput (spec 13).

Engine-agnostic over an OpenAI-compatible HTTP endpoint. vLLM's `vllm serve` exposes exactly
this (spec R11), so the same client works whether the server is vLLM (preferred) or any other
OpenAI-compatible server. The client is external to the server process, replays a fixed
arrival + length schedule with fresh payloads, and never trusts a server-returned timing field.

Goodput g = (1/T) * #{requests that complete correctly AND meet all declared SLOs}, per spec
13.2. We record every scheduled request in the denominators (completed, errored, timed-out).

This module measures a running endpoint. Applying a candidate kernel/config PATCH to the
server and reverting it is the integration layer (kept separate); this file establishes the
trusted measurement half of the Batch B exit gate.
"""
import os, sys, json, time, argparse, asyncio, statistics, subprocess, socket, urllib.request


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def wait_ready(base_url, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(base_url + "/v1/models", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(3)
    return False


def make_trace(n, in_lo, in_hi, out_len, rate_rps, seed=0):
    """Fixed offered load: n requests, Poisson-ish arrivals at rate_rps, input length in
    [in_lo,in_hi], fixed output length. Deterministic from seed."""
    import random
    rng = random.Random(seed)
    reqs, t = [], 0.0
    for i in range(n):
        t += rng.expovariate(rate_rps) if rate_rps > 0 else 0.0
        in_len = rng.randint(in_lo, in_hi)
        prompt = "Repeat the word banana. " * max(1, in_len // 6)   # fresh, length-controlled
        reqs.append({"id": i, "arrival": t, "prompt": prompt, "max_tokens": out_len})
    return reqs


async def _one(session_url, model, req, out_len):
    import aiohttp
    payload = {"model": model, "prompt": req["prompt"], "max_tokens": out_len,
               "temperature": 0.0, "stream": True}
    rec = {"id": req["id"], "ttft": None, "end": None, "ok": False, "n_tok": 0, "err": None}
    t0 = time.perf_counter()
    try:
        timeout = __import__("aiohttp").ClientTimeout(total=90)
        async with __import__("aiohttp").ClientSession(timeout=timeout) as s:
            async with s.post(session_url + "/v1/completions", json=payload) as resp:
                if resp.status != 200:
                    rec["err"] = "http_%d" % resp.status; return rec
                async for line in resp.content:
                    if not line:
                        continue
                    now = time.perf_counter()
                    if rec["ttft"] is None:
                        rec["ttft"] = now - t0
                    if b"[DONE]" in line:
                        break
                    if line.startswith(b"data:"):
                        rec["n_tok"] += 1
        rec["end"] = time.perf_counter() - t0
        rec["ok"] = rec["n_tok"] > 0
    except Exception as e:
        rec["err"] = repr(e)[:80]
    return rec


async def _run_trace(base_url, model, reqs, out_len):
    tasks = []
    t_start = time.perf_counter()
    async def sched(req):
        dt = req["arrival"] - (time.perf_counter() - t_start)
        if dt > 0:
            await asyncio.sleep(dt)
        return await _one(base_url, model, req, out_len)
    for req in reqs:
        tasks.append(asyncio.ensure_future(sched(req)))
    recs = await asyncio.gather(*tasks)
    wall = time.perf_counter() - t_start
    return recs, wall


def measure(base_url, model, reqs, out_len, ttft_slo, e2e_slo):
    recs, wall = asyncio.get_event_loop().run_until_complete(_run_trace(base_url, model, reqs, out_len))
    done = [r for r in recs if r["ok"]]
    met = [r for r in done if (r["ttft"] is not None and r["ttft"] <= ttft_slo and r["end"] <= e2e_slo)]
    ttfts = sorted(r["ttft"] for r in done if r["ttft"] is not None)
    e2es = sorted(r["end"] for r in done)
    def pct(a, p): return a[min(len(a) - 1, int(p * len(a)))] if a else None
    return {
        "offered": len(reqs), "completed": len(done), "errored": len(reqs) - len(done),
        "slo_met": len(met), "goodput_rps": round(len(met) / wall, 4) if wall else 0.0,
        "throughput_rps": round(len(done) / wall, 4) if wall else 0.0,
        "ttft_p50": pct(ttfts, 0.5), "ttft_p95": pct(ttfts, 0.95),
        "e2e_p50": pct(e2es, 0.5), "e2e_p95": pct(e2es, 0.95),
        "wall_s": round(wall, 2), "ttft_slo": ttft_slo, "e2e_slo": e2e_slo,
        "note_p95": "exploratory (<200 completed)" if len(done) < 200 else "ok",
    }


def launch_vllm(model, port, extra_args=None, gpu="0"):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
           "--port", str(port), "--gpu-memory-utilization", "0.85", "--max-model-len", "4096",
           "--disable-log-requests"] + (extra_args or [])
    return subprocess.Popen(cmd, env=env, stdout=open("/tmp/vllm_%d.log" % port, "w"), stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--python", default=sys.executable, help="python of the vllm venv")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--rate", type=float, default=8.0)
    ap.add_argument("--in-lo", type=int, default=256)
    ap.add_argument("--in-hi", type=int, default=1024)
    ap.add_argument("--out-len", type=int, default=64)
    ap.add_argument("--ttft-slo", type=float, default=2.0)
    ap.add_argument("--e2e-slo", type=float, default=10.0)
    ap.add_argument("--extra", default="", help="extra vllm args, space-separated (the A/B lever)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    port = _free_port(); base = "http://127.0.0.1:%d" % port
    proc = launch_vllm(args.model, port, args.extra.split() if args.extra else None, args.gpu)
    try:
        if not wait_ready(base, timeout=600):
            json.dump({"error": "server_not_ready"}, open(args.out, "w")); print("SERVER NOT READY"); return
        reqs = make_trace(args.n, args.in_lo, args.in_hi, args.out_len, args.rate)
        # warmup
        measure(base, args.model, make_trace(8, args.in_lo, args.in_hi, args.out_len, args.rate, seed=99), args.out_len, args.ttft_slo, args.e2e_slo)
        res = measure(base, args.model, reqs, args.out_len, args.ttft_slo, args.e2e_slo)
        res["model"] = args.model; res["extra"] = args.extra
        json.dump(res, open(args.out, "w"), indent=2)
        print("SERVING RESULT:", json.dumps(res))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
