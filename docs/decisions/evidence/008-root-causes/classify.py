"""Task 1.15 per-family root cause of every missed gate attack (evidence for decision 008).

    uv run python docs/decisions/evidence/008-root-causes/classify.py TRACES_DIR \
        OUT_DIR/primary.parquet OUT_DIR/oraclecal.parquet OUT_DIR/oracle.parquet

The three tables come from replay_variants.py for ONE configuration (the spec default, or a
candidate such as the oracle-reviewer loop with its switches). The misses classified are those of
``primary`` (warm test, gate bucket, live thresholds). Rules, applied in this order (first match):

  1. i:label-misaligned       exfil_volume whose call right after the read (the §2.3 signal
                              carrier) is not labelled as part of the attack
  2. ii:cal-learns-attacks    detected in ``oraclecal`` (same config; the calibrator never learns
                              attack-labelled calls)
  3. ii:state-learns-attacks  detected in ``oracle`` (same config, replay outcomes="oracle":
                              nothing learns attack-labelled calls); not applied to velocity_burst,
                              whose window cannot count calls that are never learned
  4. iv:benign-tail-saturated the targeted detector's best score on the attack (oracle table;
                              oraclecal for velocity) is >= 0.99 but that row's threshold >= 0.999
  5. family test on the raw data before the attack (alpha = 0.005):
       magnitude_spike / scope_blowup: iii:not-anomalous if more than alpha of the persona's benign
         values of that (tool, field) are >= the attacked value, else iv:detector-limit
       novel_target: iii:not-anomalous if the key's benign new-value rate (second half of its
         history) > alpha, else iv:novelty-ramp(n<200) when the key has < 200 observations,
         else iv:detector-limit
       velocity_burst: iii:not-anomalous if more than alpha of the persona's benign 10-minute
         windows hold >= the burst's max window count, else iv:detector-limit
       exfil_volume: the same test on the read's item_count
Each family line also reports, for exfil_volume, in how many gate attacks another call sits
between the read and the outbound write (the carrier differs from the outbound write).
"""
import sys, math, collections, json
import numpy as np, orjson
from agentfw.bench.dataset import load_dir
from agentfw.bench.replay import read_score_table
from agentfw.bench.metrics import GATE_FAMILIES, _bucket_of
from agentfw.core.normalize import SessionTracker, normalize
from agentfw.core.patterns import try_number
ALPHA = 0.005
traces, pD, pO1, pO2 = sys.argv[1:5]  # primary, oraclecal, oracle
raws, labels, _ = load_dir(traces)
D = read_score_table(pD); O1 = {r.envelope_id: r for r in read_score_table(pO1)}; O2l = read_score_table(pO2); O2 = {r.envelope_id: r for r in O2l}
# benign history
ordered = sorted(raws, key=lambda r: (r.ts_unix_ms, r.session_id, r.seq_no))
tr = SessionTracker()
vals = collections.defaultdict(list)      # (agent,tool,path) -> [(ts, value)]  benign only
calls = collections.defaultdict(list)     # (agent,tool) -> [ts] benign only
resp = collections.defaultdict(list)      # (agent,tool) -> [(ts,item_count)] benign
for r in ordered:
    env = normalize(r, tr); tr.observe(env)
    lab = labels.get(r.envelope_id)
    if lab is not None and lab.label == "attack": continue
    calls[(r.agent_id, r.tool_id)].append(r.ts_unix_ms)
    for src in (env.flat_args, env.derived):
        for p, vs in src.items():
            for v in vs: vals[(r.agent_id, r.tool_id, p)].append((r.ts_unix_ms, v))
    if r.response_meta and r.response_meta.item_count is not None:
        resp[(r.agent_id, r.tool_id)].append((r.ts_unix_ms, r.response_meta.item_count))
def exceed_num(hist, x, t):
    xs = [try_number(v) for ts, v in hist if ts < t]
    xs = [v for v in xs if v is not None and math.isfinite(v)]
    if not xs: return math.nan, 0
    return float(np.mean(np.asarray(xs) >= x)), len(xs)
def new_rate(hist, t):
    seen = set(); new = n = 0
    for ts, v in hist:
        if ts >= t: break
        k = str(v); n += 1
        if k not in seen: new += 1; seen.add(k)
    # rate over the last half of history (after vocabulary had a chance to settle)
    seen = set(); new2 = n2 = 0
    for i, (ts, v) in enumerate(hist):
        if ts >= t: break
        k = str(v)
        if i >= n // 2:
            n2 += 1; new2 += k not in seen
        seen.add(k)
    return (new2 / n2 if n2 else math.nan), n
def win_exceed(ts_list, count, t):
    ts = [x for x in ts_list if x < t]
    if not ts: return math.nan
    arr = np.asarray(ts); lo = np.searchsorted(arr, arr - 600_000, side="right")
    w = np.arange(len(arr)) - lo + 1
    return float(np.mean(w >= count))
def items(r): return orjson.loads(r.per_detector_json)
def best(r, det):
    b = None
    for it in items(r):
        if it["detector"] == det and (b is None or it["score"] > b["score"]): b = it
    return b
def flag(r, thr=None): t = r.threshold if thr is None else thr; return t is not None and r.fused_score >= t
# exfil alignment
by_sess = collections.defaultdict(list)
for r in raws: by_sess[r.session_id].append(r)
for s in by_sess.values(): s.sort(key=lambda r: (r.ts_unix_ms, r.seq_no))
groups = collections.defaultdict(list)
for r in D:
    if r.label == "attack" and r.split == "test" and r.is_warm and not math.isnan(r.fused_score): groups[r.attack_id].append(r)
out = {}
for fam, (bucket, det) in GATE_FAMILIES.items():
    cats = collections.Counter(); detail = []
    for aid, g in sorted(groups.items()):
        g.sort(key=lambda r: (r.ts_unix_ms, r.envelope_id))
        if g[0].attack_family != fam or (bucket != "all" and _bucket_of(fam, g[0].strength) != bucket): continue
        if fam == "exfil_volume":
            ids = {e for e, l in labels.items() if l.attack_id == aid}
            rr = sorted((x for x in raws if x.envelope_id in ids), key=lambda x: (x.ts_unix_ms, x.seq_no))
            sess = by_sess[rr[0].session_id]; i = next(k for k, c in enumerate(sess) if c.envelope_id == rr[0].envelope_id)
            gap = i + 1 >= len(sess) or sess[i + 1].envelope_id != rr[-1].envelope_id
            cats["(carrier is not the outbound write)"] += gap
        if any(flag(r) for r in g): cats["hit"] += 1; continue
        t0 = g[0].ts_unix_ms; agent = g[0].agent_id
        cat = None; note = ""
        if fam == "exfil_volume":
            all_e = sorted([e for e, l in labels.items() if l.attack_id == aid])
            rr = sorted((x for x in raws if x.envelope_id in set(all_e)), key=lambda x: (x.ts_unix_ms, x.seq_no))
            sess = by_sess[rr[0].session_id]; i = next(k for k, c in enumerate(sess) if c.envelope_id == rr[0].envelope_id)
            nxt = labels.get(sess[i + 1].envelope_id) if i + 1 < len(sess) else None
            if nxt is None or nxt.attack_id != aid:  # the signal carrier is not part of the attack
                cat = "i:label-misaligned"; note = f"signal on {sess[i+1].tool_id if i+1 < len(sess) else None}"
        if cat is None and any(flag(O1[r.envelope_id]) for r in g): cat = "ii:cal-learns-attacks"
        if cat is None and fam != "velocity_burst" and any(flag(O2[r.envelope_id]) for r in g): cat = "ii:state-learns-attacks"
        if cat is None:
            src = O2 if fam != "velocity_burst" else O1
            rep = max((src[r.envelope_id] for r in g), key=lambda r: (best(r, det) or {"score": 0})["score"])
            b = best(rep, det) or {"score": 0.0, "field": None, "evidence": {}}
            theta = rep.threshold
            ev = b.get("evidence") or {}
            if b["score"] >= 0.99 and theta >= 0.999:
                cat = "iv:benign-tail-saturated"; note = f"tgt={b['score']:.3f} θ={theta:.4f}"
            else:
                if fam in ("magnitude_spike", "scope_blowup"):
                    path = b["field"]; x = try_number(ev.get("observed"))
                    ex, n = exceed_num(vals[(agent, rep.tool_id, path)], x, t0) if x is not None else (math.nan, 0)
                    note = f"{path} x={x} benign>=x {ex:.4f} (n={n}) z={ev.get('z')} s={b['score']:.3f} θ={theta:.3f}"
                    cat = "iii:not-anomalous" if ex > ALPHA else "iv:detector-limit"
                elif fam == "novel_target":
                    path = b["field"]; nr, n = new_rate(vals[(agent, rep.tool_id, path)], t0)
                    note = f"{path} benign-new-rate {nr:.4f} n={n} p_unseen={ev.get('p_unseen')} s={b['score']:.3f} θ={theta:.3f}"
                    cat = "iii:not-anomalous" if nr > ALPHA else ("iv:novelty-ramp(n<200)" if n < 200 else "iv:detector-limit")
                elif fam == "velocity_burst":
                    wc = max(int((best(O1[r.envelope_id], det) or {"evidence": {}})["evidence"].get("window_count") or 0) for r in g)
                    ex = win_exceed(calls[(agent, rep.tool_id)], wc, t0)
                    note = f"max window {wc} benign windows>= {ex:.4f} s={b['score']:.3f} θ={theta:.3f} top={max(items(rep), key=lambda i: i['score'])['detector']}"
                    cat = "iii:not-anomalous" if ex > ALPHA else "iv:detector-limit"
                elif fam == "exfil_volume":
                    rd = sorted((x for x in raws if labels.get(x.envelope_id) and labels[x.envelope_id].attack_id == aid), key=lambda x: x.ts_unix_ms)[0]
                    ic = rd.response_meta.item_count
                    ex = float(np.mean([v >= ic for ts, v in resp[(agent, rd.tool_id)] if ts < t0] or [math.nan]))
                    note = f"item_count {ic} benign>= {ex:.4f} resp-score {b['score']:.3f}@{b['field']} θ={theta:.3f}"
                    cat = "iii:not-anomalous" if ex > ALPHA else "iv:detector-limit"
                else:
                    cat = "iv:detector-limit"; note = f"s={b['score']:.3f} θ={theta:.3f}"
        cats[cat] += 1; detail.append((aid, cat, note))
    n = sum(v for k, v in cats.items() if not k.startswith("("))
    print(f"== {fam} [{bucket}] n={n} " + ", ".join(f"{k}={v}" for k, v in sorted(cats.items())))
    for d in detail: print("    ", *d)
    out[fam] = dict(cats)
print(json.dumps(out))
