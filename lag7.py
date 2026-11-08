import argparse
import json
from pathlib import Path

from slot_forecasting.io import load_inputs, write_submission
from slot_forecasting.schema import PAYOUTS, index_history, unique_queries

LAG = 7
DATA = Path("public")


def key_of(q):
    return f"{q.machine_type}|{q.signal}"


def machine_plays(queries, history):
    queries = tuple(queries)
    qmap = unique_queries(queries)
    indexed = index_history(queries, history)
    out = []
    for q in sorted(qmap.values(), key=lambda q: q.query_id):
        plays = {}
        for p in indexed.get((q.episode_id, q.machine_id), []):
            if p.tick < q.cutoff:
                plays[p.tick] = PAYOUTS.index(p.payout)
        out.append((q, plays))
    return out


def fit(queries, history):
    acc = {}
    for q, plays in machine_plays(queries, history):
        A = acc.setdefault(key_of(q), {"n": 0, "w": 0, "na": 0, "wa": 0,
                                       "nb": 0, "wb": 0, "d": [0] * 5})
        for t, k in plays.items():
            win = 1 if k > 0 else 0
            A["n"] += 1
            A["w"] += win
            A["d"][k] += 1
            e = plays.get(t - LAG)
            if e is not None:
                if e > 0:
                    A["na"] += 1
                    A["wa"] += win
                else:
                    A["nb"] += 1
                    A["wb"] += win
    priors = {}
    for g, A in acc.items():
        wins = sum(A["d"][1:])
        priors[g] = {
            "w0": (A["w"] + 1) / (A["n"] + 2),
            "a0": (A["wa"] + 1) / (A["na"] + 2),
            "b0": (A["wb"] + 1) / (A["nb"] + 2),
            "d0": [(A["d"][j] + 1) / (wins + 4) for j in range(1, 5)],
        }
    return priors


def predict(priors, queries, history, strength, kd, half_life):
    fallback = next(iter(priors.values()))
    out = {}
    for q, plays in machine_plays(queries, history):
        pr = priors.get(key_of(q), fallback)
        c = q.cutoff
        n = w = na = wa = nb = wb = 0.0
        dist = [0.0] * 5
        for t, k in plays.items():
            wt = 2.0 ** (-(c - 1 - t) / half_life) if half_life else 1.0
            win = 1.0 if k > 0 else 0.0
            n += wt
            w += wt * win
            dist[k] += wt
            e = plays.get(t - LAG)
            if e is not None:
                if e > 0:
                    na += wt
                    wa += wt * win
                else:
                    nb += wt
                    wb += wt * win
        x = plays.get(c - LAG)
        if x is None:
            pw = (w + strength * pr["w0"]) / (n + strength)
        elif x > 0:
            pw = (wa + strength * pr["a0"]) / (na + strength)
        else:
            pw = (wb + strength * pr["b0"]) / (nb + strength)
        tw = sum(dist[1:])
        d = [(dist[j] + kd * pr["d0"][j - 1]) / (tw + kd) for j in range(1, 5)]
        p = [1.0 - pw] + [pw * v for v in d]
        p = [max(0.001, v) for v in p]
        s = sum(p)
        out[q.query_id] = tuple(v / s for v in p)
    return out


parser = argparse.ArgumentParser()
parser.add_argument("--strength", type=float, nargs="+", default=[5, 10, 20, 40])
parser.add_argument("--half", type=float, nargs="+", default=[0, 40, 100])
parser.add_argument("--kd", type=float, default=10)
args = parser.parse_args()

tq, th = load_inputs(DATA / "train")
priors = fit(tq, th)
val = load_inputs(DATA / "validation")
test = load_inputs(DATA / "test")

for s in args.strength:
    for h in args.half:
        out = Path(f"l7-s{int(s)}-h{int(h)}")
        if out.exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.json").write_text(json.dumps(
            {"priors": priors, "strength": s, "kd": args.kd, "half_life": h}))
        write_submission(out / "validation_predictions.csv",
                         predict(priors, val[0], val[1], s, args.kd, h))
        write_submission(out / "submission.csv",
                         predict(priors, test[0], test[1], s, args.kd, h))
        print("done", out, flush=True)