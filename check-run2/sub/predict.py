"""Offline predictor for the code ZIP.  python predict.py --data /data --out /output/submission.csv
Loads model.json (next to this file) and predicts on <data>/test. Standard library only."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

here = Path(__file__).resolve().parent
model = core.load_model(here / "model.json")
machines = core.load_split(Path(args.data) / "test")
core.write_csv(args.out, core.predict_all(model, machines))
print("wrote", args.out, len(machines), "rows")