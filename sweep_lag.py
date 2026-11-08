"""Score lagged forecasts over lag_alpha and lag_count on public validation."""

from pathlib import Path

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.contest import contest_score
from slot_forecasting.io import load_inputs, read_labels, read_rows, read_submission, read_targets
from slot_forecasting.rounds import VERSION, round_metadata

DATA = Path("public")
ALPHAS = (0.05, 0.1, 0.2, 0.4)
COUNTS = (2, 4, 8)

train_queries, train_history = load_inputs(DATA / "train")
labels = read_labels(DATA / "train/labels.csv", train_queries)
val_queries, val_history = load_inputs(DATA / "validation")
targets = read_targets(DATA / "validation/targets.csv")
groups = {row["query_id"]: row["condition"]
          for row in read_rows(DATA / "validation/conditions.csv", ("query_id", "condition"))}
reference = read_submission(DATA / "validation/reference.csv")
metadata = round_metadata(DATA)

best = None
for lag_alpha in ALPHAS:
    for lag_count in COUNTS:
        model = FrequencyForecaster("lagged", lag_alpha=lag_alpha, lag_count=lag_count)
        model.fit(train_queries, train_history, labels)
        predictions = model.predict(val_queries, val_history)
        report = contest_score(predictions, targets, groups, reference,
                               max_points=metadata["max_points"], version=VERSION)
        ratio = report["mean_loss_ratio"]
        print(f"lag_alpha={lag_alpha} lag_count={lag_count} loss_ratio={ratio:.6f}")
        if best is None or ratio < best[0]:
            best = (ratio, lag_alpha, lag_count)

print(f"best lag_alpha={best[1]} lag_count={best[2]} loss_ratio={best[0]:.6f}")
