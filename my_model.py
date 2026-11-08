from pathlib import Path
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, read_labels, write_submission

HALF_LIFE = 16
DATA = Path("public")
OUT = Path("run-h16")

if OUT.exists():
    raise SystemExit("Choose a new output folder before running again.")

train_queries, train_history = load_inputs(DATA / "train")
labels = read_labels(DATA / "train/labels.csv", train_queries)
model = FrequencyForecaster(method="pooled_recent", half_life=HALF_LIFE)
model.fit(train_queries, train_history, labels)

for split, filename in (
    ("validation", "validation_predictions.csv"),
    ("test", "submission.csv"),
):
    queries, history = load_inputs(DATA / split)
    predictions = model.predict(queries, history)
    write_submission(OUT / filename, predictions)