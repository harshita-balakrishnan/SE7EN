#!/usr/bin/env bash
# Usage: ./make_entry.sh RUN_DIR [main|adaptation] [TEAM_ID]
# Builds my-entry/, proves it reproduces RUN_DIR/submission.csv within 1e-8, checks format, zips it.
set -euo pipefail
RUN="${1:?run folder, e.g. run1}"; ROUND="${2:-main}"; TEAM="${3:-TEAM_ID}"
PY="${PYTHON:-python3}"
rm -rf my-entry "check-$RUN"; mkdir my-entry
cp core.py predict.py my-entry/ ; cp "$RUN/model.json" my-entry/
cat > my-entry/reproduce.json <<'JSON'
{
  "image": "python",
  "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"]
}
JSON
SEASON=$($PY -c "import json;print(json.load(open('public/metadata.json'))['season_id'])")
SHA=$(sha256sum "$RUN/submission.csv" | cut -d' ' -f1)
cat > my-entry/README.md <<MD
# House Edge entry
- Team ID: $TEAM
- Round: $ROUND
- Season ID: $SEASON
- Forecast SHA-256: $SHA
- Python 3.13, standard library only (no requirements.txt needed). Runs in seconds on one core.

## Recreate the forecast
    python predict.py --data /data --out /output/submission.csv
- Reads /data/test/{queries,history}.csv and the saved model.json beside predict.py (no training, no network).
- Deterministic: no randomness; fixed formulas; same output on every run.

## Method
Per-machine ridge regression of win probability on its own earlier wins (lags 1-8), neighbour machines' earlier wins,
and optional trend terms, recency-weighted, shrunk toward the machine_type|signal group (empirical Bayes).
Payout sizes: recency-weighted counts shrunk to the group size distribution. Only ticks strictly before each cutoff are used.
Separate settings for short-history and long-history machines (chosen by observed history length).
Training command (writes model.json): python fit.py --data public --out $RUN
MD
echo "== reproduce from a fresh copy"
rm -rf "check-$RUN"; mkdir "check-$RUN"; cp -r my-entry "check-$RUN/sub"
(cd "check-$RUN" && $PY sub/predict.py --data "$OLDPWD/public" --out out.csv)
$PY - "$RUN/submission.csv" "check-$RUN/out.csv" <<'PY'
import csv, sys
rd = lambda p: {r['query_id']: [float(r[c]) for c in ('p_0','p_1','p_2','p_5','p_20')] for r in csv.DictReader(open(p))}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
assert set(a) == set(b), "query ids differ"
mx = max(abs(x - y) for q in a for x, y in zip(a[q], b[q]))
print("max probability difference vs submitted CSV: %.3e (limit 1e-8)" % mx)
sys.exit(0 if mx < 1e-9 else 1)
PY
if [ -f check_submission.py ]; then
  echo "== kit format check"; $PY check_submission.py "$RUN/submission.csv"
  $PY check_submission.py "$RUN/validation_predictions.csv" --split validation
fi
if [ -f build_entry.py ]; then
  echo "== build zip"; rm -f "entry-$ROUND.zip"; $PY build_entry.py --source my-entry --out "entry-$ROUND.zip"
fi
echo "UPLOAD: $RUN/submission.csv  +  entry-$ROUND.zip"