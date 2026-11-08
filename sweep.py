from pathlib import Path
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, read_labels, write_submission

DATA = Path("public")
train_queries, train_history = load_inputs(DATA / "train")
labels = read_labels(DATA / "train/labels.csv", train_queries)
val = load_inputs(DATA / "validation")
test = load_inputs(DATA / "test")

for prior in (5, 10, 20, 40):
    for half in (8, 16, 32, 64):
        out = Path(f"sweep-p{prior}-h{half}")
        if out.exists():
            continue
        model = FrequencyForecaster(method="pooled_recent", prior_strength=prior, half_life=half)
        model.fit(train_queries, train_history, labels)
        write_submission(out / "validation_predictions.csv", model.predict(*val))
        write_submission(out / "submission.csv", model.predict(*test))
        print("done", out)
        