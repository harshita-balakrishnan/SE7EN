# House Edge entry
- Team ID: TEAM_ID
- Round: main
- Season ID: 5fffdc4d0c700bbf61c3f911
- Forecast SHA-256: 9032c25e1b484fe018dadaa4f7f83fc3fa0b277a6cc9709ff42b48dc614e9278
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
Training command (writes model.json): python fit.py --data public --out run2
