# House Edge runbook (plain Python 3, no pip)

Put core.py fit.py predict.py make_entry.sh in the kit folder (next to public/, check_submission.py, build_entry.py).
chmod +x make_entry.sh

## Main round (now)
1. python3 fit.py --data public --out run1 --quick      # ~1-3 min; look at the "HONEST validation" line
2. python3 fit.py --data public --out run1              # full grid, a few minutes
3. python3 check_submission.py run1/validation_predictions.csv --split validation   # must agree with fit.py's points
4. ./make_entry.sh run1 main YOUR_TEAM_ID               # builds my-entry/, proves reproduction, runs checker, makes entry-main.zip
5. Upload run1/submission.csv + entry-main.zip to the MAIN form. Re-upload later if something better appears (latest response counts).

## Round two (9 PM Sat)
1. Extract new kit to a NEW folder (public/metadata.json must say adaptation). Copy core.py fit.py predict.py make_entry.sh
   and run1/model.json into it.
2. Safe entry in ~1 min (keeps main configs/params, refits priors on the new data):
   python3 fit.py --data public --out r2a --reuse run1/model.json
   python3 check_submission.py r2a/validation_predictions.csv --split validation
   ./make_entry.sh r2a adaptation YOUR_TEAM_ID   -> upload to the ADAPTATION form immediately
3. Re-tune on the new patterns (a few minutes):  python3 fit.py --data public --out r2b
   If its HONEST/FINAL validation points beat r2a: ./make_entry.sh r2b adaptation YOUR_TEAM_ID, upload again.