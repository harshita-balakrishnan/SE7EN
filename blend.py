import csv
import sys
from pathlib import Path

A = Path(sys.argv[1])  # counting model folder
B = Path(sys.argv[2])  # learned model folder
COLS = ("p_0", "p_1", "p_2", "p_5", "p_20")


def read(path):
    with open(path, newline="") as f:
        return {r["query_id"]: [float(r[c]) for c in COLS] for r in csv.DictReader(f)}


for w in (0.2, 0.35, 0.5, 0.65):
    out = Path(f"blend-w{int(w * 100)}")
    out.mkdir(exist_ok=True)
    for name in ("validation_predictions.csv", "submission.csv"):
        a = read(A / name)
        b = read(B / name)
        with open(out / name, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["query_id", *COLS])
            for q in a:
                p = [(1 - w) * x + w * y for x, y in zip(a[q], b[q])]
                s = sum(p)
                wr.writerow([q] + [repr(v / s) for v in p])
    print("done", out)