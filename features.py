import math
from slot_forecasting.schema import PAYOUTS, index_history, unique_queries

N_FEATURES = 80


def build_maps(indexed):
    by_ep = {}
    for (ep, m), plays in indexed.items():
        by_ep.setdefault(ep, {})[m] = {p.tick: PAYOUTS.index(p.payout) for p in plays}
    return by_ep


def features(by_ep, ep, m, c, signal):
    """Features for predicting the spin at tick c. Uses only ticks < c."""
    machines = by_ep.get(ep, {})
    mine = machines.get(m, {})
    f = []
    # last 8 spins, one-hot, plus a "missing" flag for each lag
    for k in range(1, 9):
        v = mine.get(c - k)
        if v is None:
            f += [0, 0, 0, 0, 0, 1]
        else:
            row = [0, 0, 0, 0, 0]
            row[v] = 1
            f += row + [0]
    # prize frequencies over the last 10, last 40, and all earlier spins
    for width in (10, 40, 10 ** 6):
        counts = [0.0] * 5
        n = 0
        for t, k in mine.items():
            if c - width <= t < c:
                counts[k] += 1
                n += 1
        if n:
            counts = [x / n for x in counts]
        f += counts + [math.log1p(n) / 6.0]
    # what the other machines in the same episode did recently
    a20 = [0.0] * 5
    a60 = [0.0] * 5
    n20 = n60 = 0
    for mm, ticks in machines.items():
        if mm == m:
            continue
        for t, k in ticks.items():
            if c - 60 <= t < c:
                a60[k] += 1
                n60 += 1
                if t >= c - 20:
                    a20[k] += 1
                    n20 += 1
    f += [x / n20 for x in a20] if n20 else [0.0] * 5
    f += [x / n60 for x in a60] if n60 else [0.0] * 5
    s = int(signal)
    f += [1.0 if s == -1 else 0.0, 1.0 if s == 0 else 0.0, 1.0 if s == 1 else 0.0]
    f.append(1.0)  # bias
    return f


def predict(model, queries, history):
    """Plain Python only. model = {machine_type: 80x5 weight table}."""
    queries = tuple(queries)
    qmap = unique_queries(queries)
    by_ep = build_maps(index_history(queries, history))
    fallback = next(iter(model.values()))
    out = {}
    for q in sorted(qmap.values(), key=lambda q: q.query_id):
        W = model.get(q.machine_type, fallback)
        f = features(by_ep, q.episode_id, q.machine_id, q.cutoff, q.signal)
        z = [sum(f[i] * W[i][k] for i in range(N_FEATURES)) for k in range(5)]
        mx = max(z)
        e = [math.exp(v - mx) for v in z]
        s = sum(e)
        p = [max(0.001, v / s) for v in e]
        s = sum(p)
        out[q.query_id] = tuple(v / s for v in p)
    return out