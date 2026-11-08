"""Trainable frequency baselines with optional context pooling and recency weighting."""

from collections import defaultdict
import json
import math
from pathlib import Path

from .io import require_ids, write_json
from .schema import PAYOUTS, ValidationError, finite_number, index_history, integer, payout, probabilities, unique_queries

METHODS = ("global", "context", "empirical", "recent", "pooled", "pooled_recent", "lagged")
LAGS = tuple(range(1, 9))
PROBABILITY_FLOOR = 0.001
LAG_SHRINK_COUNT = 200.0


class FrequencyForecaster:
    def __init__(self, method="pooled", prior_strength=20.0, half_life=24.0, lag_alpha=0.15, lag_count=8):
        if method not in METHODS:
            raise ValidationError(f"Unknown baseline: {method}.")
        self.method = method
        self.prior_strength = finite_number(prior_strength, "prior_strength")
        self.half_life = finite_number(half_life, "half_life")
        self.lag_alpha = finite_number(lag_alpha, "lag_alpha")
        self.lag_count = integer(lag_count, "lag_count", 1)
        if self.prior_strength <= 0 or self.half_life <= 0 or self.lag_alpha <= 0 or self.lag_count > len(LAGS):
            raise ValidationError("prior_strength, half_life, and lag_alpha must be positive, and lag_count must be from 1 to 8.")
        self.global_prior = None
        self.context_priors = {}
        self.lag_adjustments = {}

    @staticmethod
    def _key(query):
        return f"{query.machine_type}|{query.signal}"

    def fit(self, queries, history, labels=None):
        queries = tuple(queries)
        query_map = unique_queries(queries)
        indexed = index_history(queries, history)
        if labels is not None:
            require_ids(labels, query_map)
        counts = [0.0] * len(PAYOUTS)
        contexts = defaultdict(lambda: [0.0] * len(PAYOUTS))
        for query in sorted(query_map.values(), key=lambda q: q.query_id):
            outcomes = [p.payout for p in indexed.get((query.episode_id, query.machine_id), [])]
            if labels is not None:
                outcomes.append(payout(labels[query.query_id]))
            for outcome in outcomes:
                index = PAYOUTS.index(outcome)
                counts[index] += 1
                contexts[self._key(query)][index] += 1
        total = math.fsum(counts)
        if total == 0:
            raise ValidationError("Training needs at least one observed payout or label.")
        self.global_prior = tuple((value + 0.5) / (total + 2.5) for value in counts)
        self.context_priors = {
            key: tuple((value + 12.0 * prior) / (math.fsum(values) + 12.0)
                       for value, prior in zip(values, self.global_prior))
            for key, values in sorted(contexts.items())
        }
        self.lag_adjustments = self._lag_adjustments(query_map.values(), indexed) if self.method == "lagged" else {}
        return self

    def predict(self, queries, history):
        if self.global_prior is None:
            raise ValidationError("Fit or load the model before predicting.")
        queries = tuple(queries)
        query_map = unique_queries(queries)
        indexed = index_history(queries, history)
        predictions = {}
        for query in sorted(query_map.values(), key=lambda q: q.query_id):
            if self.method == "lagged":
                plays = indexed.get((query.episode_id, query.machine_id), [])
                predictions[query.query_id] = self._predict_lagged(query, plays)
                continue
            context = self.context_priors.get(self._key(query), self.global_prior)
            if self.method in ("global", "context"):
                predictions[query.query_id] = context if self.method == "context" else self.global_prior
                continue
            pooled = self.method in ("pooled", "pooled_recent")
            prior = context if pooled else self.global_prior
            strength = self.prior_strength if pooled else 1.0
            counts = [strength * value for value in prior]
            for play in indexed.get((query.episode_id, query.machine_id), []):
                age = query.cutoff - 1 - play.tick
                weight = 2 ** (-age / self.half_life) if self.method in ("recent", "pooled_recent") else 1.0
                counts[PAYOUTS.index(play.payout)] += weight
            total = math.fsum(counts)
            predictions[query.query_id] = tuple(value / total for value in counts)
        return predictions

    def _lag_adjustments(self, queries, indexed):
        """Per machine type, ratio of P(outcome | lag, earlier payout) to the type average.

        A missing earlier tick is skipped. It is never filled in as payout 0.
        """
        outcome_counts = defaultdict(lambda: [0] * len(PAYOUTS))
        pair_counts = defaultdict(lambda: [0] * len(PAYOUTS))
        for query in queries:
            plays = indexed.get((query.episode_id, query.machine_id), [])
            by_tick = {play.tick: play.payout for play in plays}
            for play in plays:
                outcome_counts[query.machine_type][PAYOUTS.index(play.payout)] += 1
                for lag in LAGS:
                    earlier = by_tick.get(play.tick - lag)
                    if earlier is None:
                        continue
                    pair_counts[(query.machine_type, lag, earlier)][PAYOUTS.index(play.payout)] += 1
        adjustments = {}
        for machine_type, counts in outcome_counts.items():
            total = math.fsum(counts)
            average = [count / total for count in counts]
            by_lag = {}
            for lag in LAGS:
                by_payout = {}
                for earlier in PAYOUTS:
                    conditional = pair_counts.get((machine_type, lag, earlier))
                    if conditional is None:
                        continue
                    seen = math.fsum(conditional)
                    weight = seen / (seen + LAG_SHRINK_COUNT)
                    shrunk = []
                    for index in range(len(PAYOUTS)):
                        raw = (conditional[index] / seen) / average[index] if average[index] else 1.0
                        shrunk.append(1.0 + weight * (raw - 1.0))
                    by_payout[str(earlier)] = shrunk
                if by_payout:
                    by_lag[str(lag)] = by_payout
            if by_lag:
                adjustments[machine_type] = by_lag
        return adjustments

    def _pooled_recent(self, query, plays):
        context = self.context_priors.get(self._key(query), self.global_prior)
        counts = [self.prior_strength * value for value in context]
        for play in plays:
            age = query.cutoff - 1 - play.tick
            counts[PAYOUTS.index(play.payout)] += 2 ** (-age / self.half_life)
        total = math.fsum(counts)
        return [value / total for value in counts]

    def _predict_lagged(self, query, plays):
        adjusted = self._pooled_recent(query, plays)
        by_tick = {play.tick: play.payout for play in plays}
        table = self.lag_adjustments.get(query.machine_type, {})
        for lag in range(1, self.lag_count + 1):
            earlier = by_tick.get(query.cutoff - lag)
            if earlier is None:
                continue
            lifts = table.get(str(lag), {}).get(str(earlier))
            if lifts is None:
                continue
            adjusted = [value * math.pow(lift, self.lag_alpha) for value, lift in zip(adjusted, lifts)]
        return self._floor_distribution(adjusted)

    @staticmethod
    def _floor_distribution(weights):
        weights = [max(0.0, weight) for weight in weights]
        total = math.fsum(weights)
        if total <= 0:
            weights = [1.0] * len(PAYOUTS)
            total = float(len(PAYOUTS))
        probs = [weight / total for weight in weights]
        for _ in range(len(PAYOUTS) + 1):
            under = [prob < PROBABILITY_FLOOR for prob in probs]
            if not any(under):
                break
            budget = 1.0 - PROBABILITY_FLOOR * sum(under)
            free_mass = math.fsum(prob for prob, flag in zip(probs, under) if not flag)
            if budget <= 0 or free_mass <= 0:
                probs = [1.0 / len(PAYOUTS)] * len(PAYOUTS)
                break
            scale = budget / free_mass
            probs = [PROBABILITY_FLOOR if flag else prob * scale for prob, flag in zip(probs, under)]
        return tuple(probs)

    def save(self, path):
        if self.global_prior is None:
            raise ValidationError("Cannot save an unfitted model.")
        payload = {"model_version": 1, "method": self.method,
                   "prior_strength": self.prior_strength, "half_life": self.half_life,
                   "global_prior": self.global_prior, "context_priors": self.context_priors}
        if self.method == "lagged":
            payload["lag_alpha"] = self.lag_alpha
            payload["lag_count"] = self.lag_count
            payload["lag_adjustments"] = self.lag_adjustments
        write_json(path, payload)

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        base = {"model_version", "method", "prior_strength", "half_life", "global_prior", "context_priors"}
        lag_fields = {"lag_adjustments", "lag_alpha", "lag_count"}
        keys = set(value) if isinstance(value, dict) else set()
        extra = keys - base
        if (not isinstance(value, dict) or not base <= keys or not extra <= lag_fields
                or ("lag_alpha" in extra) != ("lag_count" in extra)
                or type(value["model_version"]) is not int or value["model_version"] != 1):
            raise ValidationError("Unsupported or malformed saved model.")
        lag_alpha = value["lag_alpha"] if "lag_alpha" in value else 0.15
        lag_count = value["lag_count"] if "lag_count" in value else 8
        model = cls(value["method"], value["prior_strength"], value["half_life"], lag_alpha, lag_count)
        model.global_prior = probabilities(value["global_prior"])
        if not isinstance(value["context_priors"], dict):
            raise ValidationError("context_priors must be an object.")
        model.context_priors = {key: probabilities(probs) for key, probs in value["context_priors"].items()}
        if model.method == "lagged":
            if "lag_adjustments" not in value:
                raise ValidationError("Saved lagged model is missing lag_adjustments.")
            model.lag_adjustments = cls._read_adjustments(value["lag_adjustments"])
        elif extra:
            raise ValidationError("lag fields are only valid for the lagged method.")
        return model

    @staticmethod
    def _read_adjustments(value):
        if not isinstance(value, dict):
            raise ValidationError("lag_adjustments must be an object.")
        payout_keys = {str(item) for item in PAYOUTS}
        lag_keys = {str(lag) for lag in LAGS}
        parsed = {}
        for machine_type, lags in value.items():
            if not isinstance(machine_type, str) or not isinstance(lags, dict):
                raise ValidationError("lag_adjustments must map machine types to objects.")
            parsed_lags = {}
            for lag, earlier in lags.items():
                if lag not in lag_keys or not isinstance(earlier, dict):
                    raise ValidationError("lag_adjustments lags must be 1 through 8.")
                parsed_earlier = {}
                for payout_key, lifts in earlier.items():
                    if payout_key not in payout_keys:
                        raise ValidationError("lag_adjustments payout must be one of 0, 1, 2, 5, 20.")
                    try:
                        lifts = tuple(lifts)
                    except TypeError as exc:
                        raise ValidationError("Each lag adjustment must contain five numbers.") from exc
                    if len(lifts) != len(PAYOUTS):
                        raise ValidationError("Each lag adjustment must contain five numbers.")
                    parsed_lifts = tuple(finite_number(item, "lag adjustment") for item in lifts)
                    if any(item < 0 for item in parsed_lifts):
                        raise ValidationError("lag adjustment must be non-negative.")
                    parsed_earlier[payout_key] = parsed_lifts
                parsed_lags[lag] = parsed_earlier
            parsed[machine_type] = parsed_lags
        return parsed
