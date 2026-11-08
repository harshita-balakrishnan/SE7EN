import argparse
import json
import math
import random
from pathlib import Path

import features as F
from slot_forecasting.io import load_inputs, read_labels, write_submission
from slot_forecasting.schema import PAYOUTS, payout, index_history, unique_queries

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="learn")
parser.add_argument("--stride", type=int, default=6)
parser.add_argument("--l2", type=float, nargs="+", default=[0.0001, 0.001])
parser.add_argument("--epochs", type=int, default=4)
args = parser.parse_args()

DATA = Path("public")


def build_rows(queries, history, labels, stride):
    queries = tuple(queries)
    qmap = unique_queries(queries)
    by_ep = F.build_maps(index_history(queries, history))
    data, seen = {}, set()
    for q in sorted(qmap.values(), key=lambda q: q.query_id):
        X, y = data.setdefault(q.machine_type, ([], []))
        mine = by_ep.get(q.episode_id, {}).get(q.machine_id, {})
        key = (q.episode_id, q.machine_id)
        if key not in seen:
            seen.add(key)
            for t in sorted(mine)[::stride]:
                X.append(F.features(by_ep, q.episode_id, q.machine_id, t, q.signal))
                y.append(mine[t])
        X.append(F.features(by_ep, q.episode_id, q.machine_id, q.cutoff, q.signal))
        y.append(PAYOUTS.index(payout(labels[q.query_id])))
    return data


def train(X, y, l2, epochs, lr=0.05):
    d = F.N_FEATURES
    W = [[0.0] * 5 for _ in range(d)]
    rng = random.Random(0)
    idx = list(range(len(y)))
    for ep in range(epochs):
        rng.shuffle(idx)
        step = lr / (1 + ep)
        for i in idx:
            x = X[i]
            z = [0.0] * 5
            for j in range(d):
                xj = x[j]
                if xj:
                    Wj = W[j]
                    for k in range(5):
                        z[k] += xj * Wj[k]
            mx = max(z)
            e = [math.exp(v - mx) for v in z]
            s = sum(e)
            p = [v / s for v in e]
            p[y[i]] -= 1.0
            for j in range(d):
                xj = x[j]
                Wj = W[j]
                for k in range(5):
                    Wj[k] -= step * (xj * p[k] + l2 * Wj[k])
    return W


tq, th = load_inputs(DATA / "train")
labels = read_labels(DATA / "train/labels.csv", tq)
rows = build_rows(tq, th, labels, args.stride)
for t, (X, y) in rows.items():
    print(t, "training rows:", len(y), flush=True)

val = load_inputs(DATA / "validation")
test = load_inputs(DATA / "test")

for l2 in args.l2:
    model = {}
    for t, (X, y) in rows.items():
        print("training", t, "l2", l2, flush=True)
        model[t] = train(X, y, l2, args.epochs)
    out = Path(f"{args.out}-l2-{l2}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps(model))
    write_submission(out / "validation_predictions.csv", F.predict(model, *val))
    write_submission(out / "submission.csv", F.predict(model, *test))
    print("done", out, flush=True)