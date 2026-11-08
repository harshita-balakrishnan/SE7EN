"""Tune on validation (true target probabilities), then write model.json + CSVs.

The model has two sets, chosen by observed history length (no condition labels needed at test):
  short = machine has < 60% of its earlier ticks observed   long = everything else
Each set picks its own feature config and hyper-parameters.

Examples
  python fit.py --data public --out run1                         # full tuning
  python fit.py --data public --out run1 --quick                 # small grid
  python fit.py --data public --out r2 --reuse run1/model.json   # round two: refit priors only (~1 min)
"""
import argparse
import json
import time
from pathlib import Path

import core
from core import (CONFIGS, DEFAULT_PARAMS, accumulate, assemble, build_rows, count_dist,
                  feature_classes, fit_priors, load_split, load_truth, make_row, make_set,
                  predict_all, pw_from_stats, save_model, score, size_dist, write_csv, _g)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="public")
ap.add_argument("--out", required=True)
ap.add_argument("--configs", nargs="+", default=["own", "pairs", "top3", "top3t", "mix3", "mix4", "mix5"],
                help="topK/topKt = ridge on the K strongest pairs; mixK = each machine picks ONE of the K strongest pairs")
ap.add_argument("--prior-splits", nargs="+", default=["train"],
                help="histories used to fit priors DURING tuning (keeps validation honest)")
ap.add_argument("--final-prior-splits", nargs="+", default=["train", "validation"],
                help="histories used to fit priors of the FINAL model (add 'test' to also use test histories)")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--reuse", help="model.json from an earlier run: keep its configs+params, refit priors only")
args = ap.parse_args()

T0 = time.time()
data = Path(args.data)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def log(*a):
    print("[%5.0fs]" % (time.time() - T0), *a, flush=True)


splits = {s: load_split(data / s) for s in ("train", "validation", "test")}
val, test = splits["validation"], splits["test"]
meta_path = data / "metadata.json"
max_points = float(json.loads(meta_path.read_text())["max_points"]) if meta_path.exists() else 60.0
truth = load_truth(data)
cond = truth[1]
log("loaded", {k: len(v) for k, v in splits.items()}, "max_points", max_points)
tab = {}
for m in val:
    tab[(m.short, cond[m.qid])] = tab.get((m.short, cond[m.qid]), 0) + 1
log("short-detection vs condition label (validation):", sorted(tab.items(), key=str))


def pool(names):
    return [m for s in names for m in splits[s]]


def rep(preds):
    return score(preds, truth, max_points)


def set_ratios(res):
    r = res["ratios"]
    other = [v for k, v in r.items() if k != "short_history"]
    return r.get("short_history", 1.0), sum(other) / len(other)


def fmt(res):
    return "points %.3f  ratios %s" % (res["points"], {k: round(v, 4) for k, v in res["ratios"].items()})


pri_cache = {}


def priors(cfg_name, cfg, names):
    key = (cfg_name, tuple(names))
    if key not in pri_cache:
        pri_cache[key] = fit_priors(pool(names), cfg)
    return pri_cache[key]


def build_model(choice, names):
    """choice = {'short': (cfg_name, cfg, params), 'long': (cfg_name, cfg, params)}"""
    return {"sets": {k: make_set(c, cfg, priors(c, cfg, names), p) for k, (c, cfg, p) in choice.items()}}


def register_auto_configs(names):
    """Create top-k pair configs from the all-pairs priors (on the tuning prior splits)."""
    wanted = [n for n in args.configs if n.startswith("top") or n.startswith("mix")]
    if not wanted:
        return
    base = priors("pairs", CONFIGS["pairs"], names)
    for n in wanted:
        k = int(n[3:].rstrip("t"))
        CONFIGS[n] = core.top_pairs_config(CONFIGS["pairs"], base, k, n.endswith("t") and n.startswith("top"),
                                           n.startswith("mix"))
        log("auto config %s uses pairs %s" % (n, list(CONFIGS[n]["pairs"])))


# ------------------------------------------------------------------ reuse path (round two refit)
if args.reuse:
    prev = core.load_model(args.reuse)
    choice = {k: (S["cfg_name"], S["cfg"], S["params"]) for k, S in prev["sets"].items()}
    final_model = build_model(choice, args.final_prior_splits)
    log("REUSE configs %s" % {k: v[0] for k, v in choice.items()})
else:
    HLS = (40.0, 120.0) if args.quick else (15.0, 30.0, 60.0, 120.0, 0.0)
    MULTS = (1.0, 3.0) if args.quick else (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
    MINTS = (0.5, 2.0) if args.quick else (0.1, 0.3, 1.0, 3.0, 10.0)
    register_auto_configs(args.prior_splits)
    summary = {}     # config -> {"short": params+ratio, "long": params+ratio}
    grid_log = []
    for name in args.configs:
        cfg = CONFIGS[name]
        d = len(feature_classes(cfg))
        pri = priors(name, cfg, args.prior_splits)
        log("config %-6s d=%d  prior sd per feature %s" % (name, d, [round(t ** 0.5, 3) for t in pri["tau2"]]))
        prep = []
        for m in val:
            pb = _g(pri["pbar"], m)
            prep.append((m, build_rows(cfg, m, pb), make_row(cfg, m, m.cutoff, pb),
                         size_dist(m, _g(pri["d0"], m), 0.0, 10.0)))

        def run(hl, mult, mint, stats):
            preds = {}
            for (m, rows, xc, dsz), (A, b, n) in zip(prep, stats):
                preds[m.qid] = assemble(pw_from_stats(cfg, pri, m, A, b, xc, mult, mint), dsz, None, 0.0)
            res = rep(preds)
            grid_log.append((name, hl, mult, mint, res["points"]))
            return set_ratios(res)

        best = {"short": None, "long": None}
        for hl in HLS:
            stats = [accumulate(rows, d, m.cutoff, hl) for (m, rows, xc, dsz) in prep]
            for mult in MULTS:
                rs, rl = run(hl, mult, 1.0, stats)
                for k, r in (("short", rs), ("long", rl)):
                    if best[k] is None or r < best[k]["ratio"]:
                        best[k] = {"hl": hl, "mult": mult, "mint": 1.0, "ratio": r}
        for k in ("short", "long"):
            cur = best[k]
            stats = [accumulate(rows, d, m.cutoff, cur["hl"]) for (m, rows, xc, dsz) in prep]
            for mint in MINTS:
                rs, rl = run(cur["hl"], cur["mult"], mint, stats)
                r = rs if k == "short" else rl
                if r < cur["ratio"]:
                    cur["mint"], cur["ratio"] = mint, r
        summary[name] = best
        log("  %-6s short ratio %.4f %s | long ratio %.4f %s" % (
            name, best["short"]["ratio"], {k: v for k, v in best["short"].items() if k != "ratio"},
            best["long"]["ratio"], {k: v for k, v in best["long"].items() if k != "ratio"}))

    choice = {}
    for k in ("short", "long"):
        name = min(summary, key=lambda n: summary[n][k]["ratio"])
        b = summary[name][k]
        choice[k] = (name, CONFIGS[name], dict(DEFAULT_PARAMS, hl=b["hl"], mult=b["mult"], mint=b["mint"]))
        log("BEST for %s set: config=%s ratio=%.4f" % (k, name, b["ratio"]))

    # ---- stage 2: payout sizes / count-blend (win-prob cached)
    tune_model = build_model(choice, args.prior_splits)
    stats_cache = {}
    for m in val:
        S = tune_model["sets"]["short" if m.short else "long"]
        cfg, prm = S["cfg"], S["params"]
        pb = _g(S["pbar"], m)
        A, b, _ = accumulate(build_rows(cfg, m, pb), len(feature_classes(cfg)), m.cutoff, prm["hl"])
        stats_cache[m.qid] = (A, b, make_row(cfg, m, m.cutoff, pb))

    def eval2(over):
        preds = {}
        for m in val:
            k = "short" if m.short else "long"
            S = tune_model["sets"][k]
            prm = dict(S["params"], **over[k])
            A, b, xc = stats_cache[m.qid]
            pw = pw_from_stats(S["cfg"], S, m, A, b, xc, prm["mult"], prm["mint"], prm)
            c5 = count_dist(m, _g(S["base5"], m), prm["hb"], prm["sb"]) if prm["blend"] > 0 else None
            preds[m.qid] = assemble(pw, size_dist(m, _g(S["d0"], m), prm["hs"], prm["kd"]),
                                    c5, prm["blend"])
        return set_ratios(rep(preds))

    grids = [("T", (1.0, 2.0, 4.0)), ("pt", (1.0, 3.0, 8.0)), ("p0", (0.02, 0.1, 0.3)),
             ("hs", (0.0, 60.0, 150.0)), ("kd", (5.0, 10.0, 30.0)), ("blend", (0.0, 0.15, 0.3))]
    if args.quick:
        grids = [("T", (1.0, 3.0)), ("pt", (1.0, 4.0)), ("hs", (0.0, 100.0)), ("kd", (10.0, 30.0)), ("blend", (0.0, 0.2))]
    for key, values in grids:
        best = {"short": None, "long": None}
        for v in values:
            rs, rl = eval2({"short": {key: v}, "long": {key: v}})
            for k, r in (("short", rs), ("long", rl)):
                if best[k] is None or r < best[k][0]:
                    best[k] = (r, v)
        for k in ("short", "long"):
            choice[k][2][key] = best[k][1]
            tune_model["sets"][k]["params"][key] = best[k][1]
        log("stage2 %s: short=%s long=%s" % (key, best["short"][1], best["long"][1]))

    honest = rep(predict_all(build_model(choice, args.prior_splits), val))
    log("HONEST validation (priors from %s only): %s" % (args.prior_splits, fmt(honest)))
    (out / "tuning.json").write_text(json.dumps(
        {"summary": summary, "grid": grid_log, "choice": {k: [v[0], list(map(list, v[1].get("pairs", ()))), v[2]] for k, v in choice.items()},
         "honest_points": honest["points"]}, indent=1))
    final_model = build_model(choice, args.final_prior_splits)

# ------------------------------------------------------------------ write outputs
save_model(out / "model.json", final_model)
vp = predict_all(final_model, val)
write_csv(out / "validation_predictions.csv", vp)
log("FINAL model on validation: %s" % fmt(rep(vp)))
tp = predict_all(final_model, test)
write_csv(out / "submission.csv", tp)
log("wrote %s, validation_predictions.csv, submission.csv (%d test rows)" % (out / "model.json", len(tp)))
