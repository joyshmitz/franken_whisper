#!/usr/bin/env python3
"""Round-3 pass-4 span analyzer: per-span cumulative shares + per-window drift.

Reads a spans file (JSON lines mixed with tracing noise), sums each span tag's
cumulative_ms, prints the ranked share table, and (for decode_loop / encoder_window)
prints the per-window timeline so drift over window index is visible.
"""
import sys, json
from collections import defaultdict

def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if '"event":"perf.profile.span_summary"' not in line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows

def main(path):
    rows = load(path)
    sums = defaultdict(float)
    counts = defaultdict(int)
    toks = defaultdict(int)
    for r in rows:
        s = r["span"]
        sums[s] += r["cumulative_ms"]
        counts[s] += 1
        if "tokens" in r:
            toks[s] += r["tokens"]
    total = sum(sums.values())
    print(f"== {path}")
    print(f"total span-ms sum: {total:.1f}")
    print(f"{'span':22} {'ms':>10} {'%':>6} {'n':>5} {'ms/call':>9}")
    for s in sorted(sums, key=lambda k: -sums[k]):
        pct = 100*sums[s]/total if total else 0
        print(f"{s:22} {sums[s]:10.1f} {pct:6.1f} {counts[s]:5} {sums[s]/counts[s]:9.2f}")
    if toks.get("decode_loop"):
        print(f"decode_loop ms/tok: {sums['decode_loop']/toks['decode_loop']:.3f}  (tokens={toks['decode_loop']})")
    # per-window drift for decode_loop ms/tok
    dl = [r for r in rows if r["span"]=="decode_loop" and r.get("tokens",0)>0]
    if dl:
        print("\nper-window decode_loop ms/tok (window_idx: ms/tok, tokens):")
        for i,r in enumerate(dl):
            print(f"  w{i:>3}: {r['cumulative_ms']/r['tokens']:6.3f}  (tok={r['tokens']})", end="")
            if (i+1)%4==0: print()
        print()
    ew = [r for r in rows if r["span"]=="encoder_window"]
    if ew:
        ms = [r["cumulative_ms"] for r in ew]
        print(f"\nencoder_window: n={len(ms)} mean={sum(ms)/len(ms):.2f} min={min(ms):.2f} max={max(ms):.2f}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
        print()
