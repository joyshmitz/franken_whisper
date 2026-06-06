#!/usr/bin/env python3
"""Aggregate franken_whisper perf spans from a run's stderr log.

Span schema (one JSON obj/line, event=perf.profile.span_summary):
  span: mel|encoder_window|cross_kv|decoder_prefill|decode_loop|
        model_parse|model_weights|version_tag|backend_run
  cumulative_ms: per-CALL elapsed ms (mis-named; it's the elapsed of that span)
  at_ms: timeline offset from perf T0 (process-ish start) when span ENDED
  extra: prompt_tokens (prefill), tokens (decode_loop)

Per-window grouping: spans cycle mel?(once)->[encoder_window,cross_kv,
decoder_prefill,decode_loop]xN. We tie each window by order of encoder_window.
"""
import json, sys, statistics as st

def load(path):
    rows=[]
    with open(path) as f:
        for line in f:
            line=line.strip()
            if '"perf.profile.span_summary"' not in line: continue
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows

def pct(xs,p):
    if not xs: return 0.0
    xs=sorted(xs); k=(len(xs)-1)*p/100.0; f=int(k); c=min(f+1,len(xs)-1)
    return xs[f]+(xs[c]-xs[f])*(k-f)

def main():
    path=sys.argv[1]
    rows=load(path)
    by={}
    for r in rows: by.setdefault(r['span'],[]).append(r)
    # fixed overheads
    def one(name): return by[name][0]['cumulative_ms'] if name in by else 0.0
    parse=one('model_parse'); weights=one('model_weights'); vtag=one('version_tag')
    mel=sum(r['cumulative_ms'] for r in by.get('mel',[]))
    n_mel=len(by.get('mel',[]))
    enc=[r['cumulative_ms'] for r in by.get('encoder_window',[])]
    xkv=[r['cumulative_ms'] for r in by.get('cross_kv',[])]
    pre=[r['cumulative_ms'] for r in by.get('decoder_prefill',[])]
    pre_tok=[r.get('prompt_tokens',0) for r in by.get('decoder_prefill',[])]
    loop=[r['cumulative_ms'] for r in by.get('decode_loop',[])]
    loop_tok=[r.get('tokens',0) for r in by.get('decode_loop',[])]
    n_win=len(enc)
    backend=by.get('backend_run',[])
    backend_ms=backend[0]['cumulative_ms'] if backend else None
    backend_at=backend[0]['at_ms'] if backend else None
    last_at=max((r['at_ms'] for r in rows), default=0)

    def dist(name,xs):
        if not xs:
            print(f"  {name:16s} n=0"); return
        print(f"  {name:16s} n={len(xs):4d} sum={sum(xs)/1000:8.2f}s "
              f"mean={st.mean(xs):7.2f} p50={pct(xs,50):7.2f} p95={pct(xs,95):7.2f} "
              f"min={min(xs):7.2f} max={max(xs):7.2f}")

    print(f"== {path} ==")
    print(f"windows={n_win}  mel_calls={n_mel}")
    print(f"FIXED OVERHEAD: model_parse={parse:.1f}ms weights={weights:.1f}ms "
          f"version_tag={vtag:.1f}ms mel_total={mel/1000:.2f}s")
    dist('encoder_window',enc)
    dist('cross_kv',xkv)
    dist('decoder_prefill',pre)
    dist('decode_loop',loop)
    tot_tok=sum(loop_tok)
    print(f"TOKENS total={tot_tok}  prefill_prompt_tok total={sum(pre_tok)} "
          f"mean_prompt={st.mean(pre_tok) if pre_tok else 0:.1f} max_prompt={max(pre_tok) if pre_tok else 0}")
    if tot_tok and loop:
        print(f"  decode_loop ms/tok = {sum(loop)/tot_tok:.3f}")
    span_sum=mel+sum(enc)+sum(xkv)+sum(pre)+sum(loop)
    load_sum=parse+weights+vtag
    print(f"SPAN SUM (mel+enc+xkv+pre+loop)={span_sum/1000:.2f}s  load(parse+weights+vtag)={load_sum/1000:.2f}s")
    if backend_ms is not None:
        unacc_in_backend=backend_ms-span_sum-load_sum  # wav read + segment build + silence analyze, inside backend_run
        print(f"backend_run={backend_ms/1000:.2f}s ended at_ms={backend_at/1000:.2f}s")
        print(f"  in-backend unaccounted (wav read/seg build/silence) = backend_run - spans - load = {unacc_in_backend/1000:.2f}s")
        print(f"POST-BACKEND tail (raw_output_json + RunReport + CLI ser, before process exit) approx = last_at - backend_at = {(last_at-backend_at)/1000:.2f}s (NOTE: only if a later span exists; else use wall-backend_at externally)")
    print(f"last_at_ms={last_at/1000:.2f}s")

    # drift over window index: compare first-quarter vs last-quarter means
    def quarter(xs):
        if len(xs)<8: return None
        q=len(xs)//4
        return st.mean(xs[:q]), st.mean(xs[-q:])
    for nm,xs in [('encoder_window',enc),('cross_kv',xkv),('decoder_prefill',pre),('decode_loop',loop)]:
        qd=quarter(xs)
        if qd: print(f"DRIFT {nm:16s} first-q mean={qd[0]:.2f}  last-q mean={qd[1]:.2f}  ratio={qd[1]/qd[0] if qd[0] else 0:.2f}")

if __name__=='__main__':
    main()
