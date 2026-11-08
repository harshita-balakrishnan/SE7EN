from collections import defaultdict
from pathlib import Path
from slot_forecasting.io import load_inputs
from slot_forecasting.schema import index_history, unique_queries

DATA = Path("public")
queries, history = load_inputs(DATA / "train")
queries = tuple(queries)
qmap = unique_queries(queries)
indexed = index_history(queries, history)

info = {}
for q in qmap.values():
    info[(q.episode_id, q.machine_id)] = (q.machine_type, q.signal)

LAGS = range(1, 9)
effects = defaultdict(lambda: {k: [] for k in LAGS})
winrate = defaultdict(lambda: [0, 0])

for key, group in info.items():
    plays = indexed.get(key, [])
    win = {p.tick: 1 if p.payout > 0 else 0 for p in plays}
    for w in win.values():
        winrate[group][0] += 1
        winrate[group][1] += w
    for k in LAGS:
        a = [0, 0]  # earlier spin was a win: [count, wins now]
        b = [0, 0]  # earlier spin was not a win
        for t, w in win.items():
            e = win.get(t - k)
            if e is None:
                continue
            side = a if e else b
            side[0] += 1
            side[1] += w
        if a[0] >= 10 and b[0] >= 10:
            effects[group][k].append(a[1] / a[0] - b[1] / b[0])

print("Effect = chance of a win now if the spin k ticks ago was a win, minus if it was not.")
for g in sorted(effects):
    n, w = winrate[g]
    print(f"\n{g}   win rate {w / n:.3f}")
    for k in LAGS:
        v = effects[g][k]
        if v:
            mean = sum(v) / len(v)
            sd = (sum((x - mean) ** 2 for x in v) / len(v)) ** 0.5
            pos = sum(1 for x in v if x > 0) / len(v)
            print(f"  lag {k}: mean {mean:+.3f}  spread {sd:.3f}  positive {pos:.0%}  machines {len(v)}")