"""House Edge core library: plain Python 3, standard library only.

Model (per machine, empirical-Bayes hierarchical ridge):
  P(win at tick t) = linear model on
      own earlier wins at lags (default 1..8), neighbour-machine earlier wins at lags,
      optional time-trend terms and an "episode shift" term.
  Coefficients are fit on THAT machine's own history (recency-weighted) and shrunk toward
  the (machine_type|signal) group mean; shrinkage strength per feature is estimated from how
  much machines really differ (empirical Bayes), then scaled by tuned multipliers.
  Payout sizes given a win: recency-weighted counts shrunk to the group distribution.
Every feature for a spin at tick t uses ONLY ticks < t.
"""
import csv
import json
import math
from pathlib import Path

PAYOUTS = (0, 1, 2, 5, 20)
PIDX = {0: 0, 1: 1, 2: 2, 5: 3, 20: 4}
COLS = ("p_0", "p_1", "p_2", "p_5", "p_20")
SHORT_FRAC = 0.6   # machine is "short history" if observed ticks < SHORT_FRAC * cutoff
MIN_T = 8          # earliest tick used as a training row
P_LO, P_HI = 0.01, 0.98
PROB_FLOOR = 1e-3

CONFIGS = {
    "own":   {"own": (1, 2, 3, 4, 5, 6, 7, 8), "nb": (), "trend": 0, "shift": False},
    "nb":    {"own": (1, 2, 3, 4, 5, 6, 7, 8), "nb": (1, 2, 3, 4, 5, 6, 7, 8), "trend": 0, "shift": False},
    "trend": {"own": (1, 2, 3, 4, 5, 6, 7, 8), "nb": (1, 2, 3, 4, 5, 6, 7, 8), "trend": 2, "shift": False},
    "full":  {"own": (1, 2, 3, 4, 5, 6, 7, 8), "nb": (1, 2, 3, 4, 5, 6, 7, 8), "trend": 2, "shift": True},
    # pair interactions of own earlier outcomes: (win_a - p) * (win_b - p)
    "pairs7": {"own": (1, 2, 3, 4, 5, 6, 7, 8), "nb": (), "trend": 0, "shift": False,
               "pairs": tuple((k, 7) if k < 7 else (7, k) for k in (1, 2, 3, 4, 5, 6, 8))},
    "pairs": {"own": (1, 2, 3, 4, 5, 6, 7, 8), "nb": (), "trend": 0, "shift": False,
              "pairs": tuple((a, b) for a in range(1, 9) for b in range(a + 1, 9))},
}

DEFAULT_PARAMS = {"hl": 100.0, "mult": 1.0, "mint": 1.0, "hs": 0.0, "kd": 10.0,
                  "blend": 0.0, "hb": 16.0, "sb": 10.0}


# ----------------------------------------------------------------------------- data
def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


class Machine:
    def __init__(self, qid, ep, mid, mtype, signal, cutoff):
        self.qid, self.ep, self.mid = qid, ep, mid
        self.mtype, self.signal, self.cutoff = mtype, int(signal), int(cutoff)
        self.group = "%s|%d" % (mtype, self.signal)
        self.plays = {}          # tick -> payout index (0..4); only ticks < cutoff
        self.epctx = None
        self.pw_own = self.pn_own = None

    @property
    def nobs(self):
        return len(self.plays)

    @property
    def short(self):
        return len(self.plays) < SHORT_FRAC * self.cutoff


class Episode:
    """Per-tick win/obs totals over all machines in the episode (+ prefix sums)."""

    def __init__(self, machines):
        c = max(m.cutoff for m in machines)
        self.tw = [0] * (c + 1)
        self.tn = [0] * (c + 1)
        for m in machines:
            for t, k in m.plays.items():
                self.tn[t] += 1
                if k > 0:
                    self.tw[t] += 1
        self.ptw = [0] * (c + 2)
        self.ptn = [0] * (c + 2)
        for i in range(c + 1):
            self.ptw[i + 1] = self.ptw[i] + self.tw[i]
            self.ptn[i + 1] = self.ptn[i] + self.tn[i]
        for m in machines:
            m.epctx = self
            pw = [0] * (c + 2)
            pn = [0] * (c + 2)
            for i in range(c + 1):
                v = m.plays.get(i)
                pn[i + 1] = pn[i] + (v is not None)
                pw[i + 1] = pw[i] + (v is not None and v > 0)
            m.pw_own, m.pn_own = pw, pn


def load_split(directory):
    """Load queries + history of one split folder. Ticks >= cutoff are dropped (never used)."""
    d = Path(directory)
    machines = []
    index = {}
    for r in read_rows(d / "queries.csv"):
        m = Machine(r["query_id"], r["episode_id"], r["machine_id"], r["machine_type"],
                    r["signal"], r["cutoff"])
        key = (m.ep, m.mid)
        if key in index:
            raise ValueError("duplicate machine %s" % (key,))
        index[key] = m
        machines.append(m)
    for r in read_rows(d / "history.csv"):
        m = index.get((r["episode_id"], r["machine_id"]))
        if m is None:
            continue
        t = int(r["tick"])
        if t >= m.cutoff:
            continue
        m.plays[t] = PIDX[int(float(r["payout"]))]
    by_ep = {}
    for m in machines:
        by_ep.setdefault(m.ep, []).append(m)
    for ms in by_ep.values():
        Episode(ms)
    machines.sort(key=lambda m: m.qid)
    return machines


# ----------------------------------------------------------------------------- features
def feature_classes(cfg):
    cls = ["int"] + ["lag"] * (len(cfg["own"]) + len(cfg["nb"]))
    cls += ["lag"] * (1 if cfg["trend"] >= 1 else 0)
    cls += ["lag"] * (1 if cfg["trend"] >= 2 else 0)
    cls += ["lag"] * (1 if cfg["shift"] else 0)
    cls += ["lag"] * len(cfg.get("pairs", ()))
    cls += ["lag"] * (len(cfg.get("pairs", ())) if cfg.get("ptrend") else 0)
    return cls


def feature_names(cfg):
    n = ["intercept"] + ["own%d" % k for k in cfg["own"]] + ["nb%d" % k for k in cfg["nb"]]
    if cfg["trend"] >= 1:
        n.append("trend")
    if cfg["trend"] >= 2:
        n.append("trend*own7")
    if cfg["shift"]:
        n.append("nb_shift")
    n += ["pair%d_%d" % p for p in cfg.get("pairs", ())]
    if cfg.get("ptrend"):
        n += ["trend*pair%d_%d" % p for p in cfg.get("pairs", ())]
    return n


def _nb_rate(mv, a, b):
    """Neighbour-machine win rate over ticks [a, b); None if nothing observed."""
    ep = mv.epctx
    n = (ep.ptn[b] - ep.ptn[a]) - (mv.pn_own[b] - mv.pn_own[a])
    if n <= 0:
        return None
    s = (ep.ptw[b] - ep.ptw[a]) - (mv.pw_own[b] - mv.pw_own[a])
    return s / n


def make_row(cfg, mv, t, pb):
    """Features for predicting the spin at tick t. Reads ticks < t only."""
    P = mv.plays
    ep = mv.epctx
    x = [1.0]
    x7 = 0.0
    cen = {}
    for k in cfg["own"]:
        v = P.get(t - k)
        val = 0.0 if v is None else (1.0 if v > 0 else 0.0) - pb
        cen[k] = val
        if k == 7:
            x7 = val
        x.append(val)
    for k in cfg["nb"]:
        u = t - k
        if u < 0:
            x.append(0.0)
            continue
        v = P.get(u)
        n = ep.tn[u] - (v is not None)
        if n <= 0:
            x.append(0.0)
        else:
            s = ep.tw[u] - (v is not None and v > 0)
            x.append(s / n - pb)
    if cfg["trend"] >= 1:
        tau = (t - mv.cutoff) / 100.0
        x.append(tau)
        if cfg["trend"] >= 2:
            x.append(tau * x7)
    if cfg["shift"]:
        r = _nb_rate(mv, max(0, t - 10), t)
        l = _nb_rate(mv, max(0, t - 110), max(0, t - 10))
        x.append(0.0 if (r is None or l is None) else r - l)
    for a, b in cfg.get("pairs", ()):
        x.append(cen[a] * cen[b])
    if cfg.get("ptrend"):
        tau = (t - mv.cutoff) / 100.0
        for a, b in cfg.get("pairs", ()):
            x.append(tau * cen[a] * cen[b])
    return x


def build_rows(cfg, mv, pb):
    P = mv.plays
    return [(make_row(cfg, mv, t, pb), 1.0 if P[t] > 0 else 0.0, t)
            for t in sorted(P) if t >= MIN_T]


# ----------------------------------------------------------------------------- linear algebra
def accumulate(rows, d, c, hl):
    A = [[0.0] * d for _ in range(d)]
    b = [0.0] * d
    n = 0.0
    for x, y, t in rows:
        w = 2.0 ** (-(c - 1 - t) / hl) if hl > 0 else 1.0
        n += w
        for i in range(d):
            xi = x[i] * w
            if xi != 0.0:
                Ai = A[i]
                for j in range(i, d):
                    Ai[j] += xi * x[j]
                b[i] += xi * y
    return A, b, n


def _full(A, add):
    d = len(A)
    M = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i, d):
            M[i][j] = M[j][i] = A[i][j]
        M[i][i] += add[i]
    return M


def chol(M):
    d = len(M)
    L = [[0.0] * d for _ in range(d)]
    for i in range(d):
        Li = L[i]
        for j in range(i + 1):
            Lj = L[j]
            s = M[i][j]
            for k in range(j):
                s -= Li[k] * Lj[k]
            if i == j:
                Li[j] = math.sqrt(s if s > 1e-12 else 1e-12)
            else:
                Li[j] = s / Lj[j]
    return L


def chol_solve(L, b):
    d = len(b)
    y = [0.0] * d
    for i in range(d):
        s = b[i]
        Li = L[i]
        for k in range(i):
            s -= Li[k] * y[k]
        y[i] = s / Li[i]
    x = [0.0] * d
    for i in range(d - 1, -1, -1):
        s = y[i]
        for k in range(i + 1, d):
            s -= L[k][i] * x[k]
        x[i] = s / L[i][i]
    return x


def ridge_solve(A, b, beta0, lam):
    """argmin  sum_w (y-x.beta)^2 + sum_j lam_j (beta_j-beta0_j)^2   (A,b = X'WX, X'Wy)."""
    L = chol(_full(A, lam))
    return chol_solve(L, [b[i] + lam[i] * beta0[i] for i in range(len(b))])


# ----------------------------------------------------------------------------- priors (empirical Bayes)
def fit_priors(machines, cfg, tau_floor=2.5e-4):
    d = len(feature_classes(cfg))
    # group win rates (centering constants) and size/class distributions
    g_n, g_w, g_cnt = {}, {}, {}
    for m in machines:
        for g in (m.group, "*"):
            g_n[g] = g_n.get(g, 0) + m.nobs
            cnt = g_cnt.setdefault(g, [0.0] * 5)
            for k in m.plays.values():
                cnt[k] += 1
    pbar, d0, base5 = {}, {}, {}
    for g, cnt in g_cnt.items():
        tot = sum(cnt)
        pbar[g] = sum(cnt[1:]) / tot
        wins = sum(cnt[1:])
        d0[g] = [(cnt[j] + 1.0) / (wins + 4.0) for j in range(1, 5)]
        base5[g] = [(cnt[j] + 0.5) / (tot + 2.5) for j in range(5)]
    # unweighted per-machine normal equations
    stats = []
    pooled = {}
    for m in machines:
        pb = pbar[m.group]
        rows = build_rows(cfg, m, pb)
        if not rows:
            continue
        A, b, n = accumulate(rows, d, m.cutoff, 0.0)
        stats.append((m, A, b, n))
        for g in (m.group, "*"):
            if g not in pooled:
                pooled[g] = ([[0.0] * d for _ in range(d)], [0.0] * d, 0)
            PA, Pb, Pn = pooled[g]
            for i in range(d):
                Pb[i] += b[i]
                for j in range(i, d):
                    PA[i][j] += A[i][j]
            pooled[g] = (PA, Pb, Pn + 1)
    beta0 = {}
    ridge = [0.05] * d
    for g, (PA, Pb, Pn) in pooled.items():
        if g != "*" and Pn < 10:
            continue
        beta0[g] = ridge_solve(PA, Pb, [0.0] * d, ridge)
    # heterogeneity across machines: var(beta_hat_m - beta0_g) - mean sampling variance
    dev2 = [0.0] * d
    se2 = [0.0] * d
    cntm = 0
    eps = [0.05] * d
    for m, A, b, n in stats:
        if n < 0.6 * m.cutoff:
            continue
        pb = pbar[m.group]
        sig2 = max(pb * (1 - pb), 0.02)
        b0 = beta0.get(m.group, beta0["*"])
        L = chol(_full(A, eps))
        bh = chol_solve(L, [b[i] + eps[i] * b0[i] for i in range(d)])
        for j in range(d):
            e = [0.0] * d
            e[j] = 1.0
            se2[j] += sig2 * chol_solve(L, e)[j]
            dev2[j] += (bh[j] - b0[j]) ** 2
        cntm += 1
    tau2 = []
    for j in range(d):
        if cntm:
            tau2.append(max(tau_floor, (dev2[j] - se2[j]) / cntm))
        else:
            tau2.append(1.0)
    return {"pbar": pbar, "beta0": beta0, "tau2": tau2, "d0": d0, "base5": base5}


# ----------------------------------------------------------------------------- prediction pieces
def _g(table, mv):
    return table.get(mv.group, table["*"])


def lam_vector(cfg, pri, mv, mult, mint):
    pb = _g(pri["pbar"], mv)
    sig2 = max(pb * (1 - pb), 0.02)
    cls = feature_classes(cfg)
    return [sig2 / pri["tau2"][j] * (mint if cls[j] == "int" else mult) for j in range(len(cls))]


def pw_from_stats(cfg, pri, mv, A, b, xc, mult, mint):
    beta0 = _g(pri["beta0"], mv)
    lam = lam_vector(cfg, pri, mv, mult, mint)
    beta = ridge_solve(A, b, beta0, lam)
    p = sum(xj * bj for xj, bj in zip(xc, beta))
    return min(P_HI, max(P_LO, p))


def size_dist(mv, d0, hs, kd):
    c = mv.cutoff
    cnt = [0.0, 0.0, 0.0, 0.0]
    for t, k in mv.plays.items():
        if k > 0:
            cnt[k - 1] += 2.0 ** (-(c - 1 - t) / hs) if hs > 0 else 1.0
    tot = cnt[0] + cnt[1] + cnt[2] + cnt[3]
    return [(cnt[j] + kd * d0[j]) / (tot + kd) for j in range(4)]


def count_dist(mv, base5, hb, sb):
    c = mv.cutoff
    cnt = [0.0] * 5
    for t, k in mv.plays.items():
        cnt[k] += 2.0 ** (-(c - 1 - t) / hb)
    tot = sum(cnt)
    return [(cnt[j] + sb * base5[j]) / (tot + sb) for j in range(5)]


def assemble(pw, dsz, count5, blend):
    p = [1.0 - pw] + [pw * v for v in dsz]
    if blend > 0:
        p = [(1 - blend) * a + blend * c for a, c in zip(p, count5)]
    p = [max(PROB_FLOOR, v) for v in p]
    s = sum(p)
    return [v / s for v in p]


def set_of(mv):
    return "short" if mv.short else "long"


def predict_machine(model, mv):
    """model["sets"][short|long] = {cfg_name,cfg,pbar,beta0,tau2,d0,base5,params}."""
    S = model["sets"][set_of(mv)]
    cfg = S["cfg"]
    prm = S["params"]
    pb = _g(S["pbar"], mv)
    d = len(feature_classes(cfg))
    rows = build_rows(cfg, mv, pb)
    A, b, _ = accumulate(rows, d, mv.cutoff, prm["hl"])
    xc = make_row(cfg, mv, mv.cutoff, pb)
    pw = pw_from_stats(cfg, S, mv, A, b, xc, prm["mult"], prm["mint"])
    dsz = size_dist(mv, _g(S["d0"], mv), prm["hs"], prm["kd"])
    c5 = count_dist(mv, _g(S["base5"], mv), prm["hb"], prm["sb"]) if prm["blend"] > 0 else None
    return assemble(pw, dsz, c5, prm["blend"])


def predict_all(model, machines):
    return {m.qid: predict_machine(model, m) for m in machines}


def write_csv(path, preds):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("query_id,p_0,p_1,p_2,p_5,p_20\n")
        for q in sorted(preds):
            f.write(q + "," + ",".join(format(v, ".17g") for v in preds[q]) + "\n")


def read_pred_csv(path):
    return {r["query_id"]: [float(r[c]) for c in COLS] for r in read_rows(path)}


# ----------------------------------------------------------------------------- model file
def top_pairs_config(cfg_all, pri_all, k, ptrend):
    """Config with only the k pairs whose between-machine spread (prior sd) is largest."""
    pairs = cfg_all["pairs"]
    off = len(feature_classes(cfg_all)) - len(pairs)
    order = sorted(range(len(pairs)), key=lambda i: -pri_all["tau2"][off + i])[:k]
    cfg = {"own": cfg_all["own"], "nb": (), "trend": 1 if ptrend else 0, "shift": False,
           "pairs": tuple(pairs[i] for i in sorted(order)), "ptrend": ptrend}
    return cfg


def make_set(cfg_name, cfg, pri, params):
    return {"cfg_name": cfg_name, "cfg": cfg, "pbar": pri["pbar"], "beta0": pri["beta0"],
            "tau2": pri["tau2"], "d0": pri["d0"], "base5": pri["base5"], "params": params}


def save_model(path, model):
    sets = {}
    for k, S in model["sets"].items():
        c = S["cfg"]
        sets[k] = dict(S, cfg={"own": list(c["own"]), "nb": list(c["nb"]),
                               "trend": c["trend"], "shift": c["shift"],
                               "pairs": [list(p) for p in c.get("pairs", ())],
                               "ptrend": bool(c.get("ptrend", False))})
    Path(path).write_text(json.dumps({"version": 2, "sets": sets}, indent=1), encoding="utf-8")


def load_model(path):
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    for S in m["sets"].values():
        c = S["cfg"]
        S["cfg"] = {"own": tuple(c["own"]), "nb": tuple(c["nb"]), "trend": c["trend"], "shift": c["shift"],
                    "pairs": tuple(tuple(p) for p in c.get("pairs", ())),
                    "ptrend": bool(c.get("ptrend", False))}
    return m


# ----------------------------------------------------------------------------- scoring (mirrors RULES.md)
def load_truth(data_dir, split="validation"):
    d = Path(data_dir) / split
    targets = {r["query_id"]: (r["episode_id"], [float(r[c]) for c in COLS])
               for r in read_rows(d / "targets.csv")}
    cond = {r["query_id"]: r["condition"] for r in read_rows(d / "conditions.csv")}
    ref = read_pred_csv(d / "reference.csv")
    return targets, cond, ref


def cond_errors(preds, targets, cond):
    """condition -> mean over episodes of mean over queries of sum sq error."""
    per_ep = {}
    for q, (ep, tp) in targets.items():
        e = sum((a - b) ** 2 for a, b in zip(preds[q], tp))
        per_ep.setdefault((cond[q], ep), []).append(e)
    out = {}
    for (c, ep), es in per_ep.items():
        out.setdefault(c, []).append(sum(es) / len(es))
    return {c: sum(v) / len(v) for c, v in out.items()}


def score(preds, truth, max_points=60.0):
    targets, cond, ref = truth
    E = cond_errors(preds, targets, cond)
    B = cond_errors(ref, targets, cond)
    ratios = {c: E[c] / max(B[c], 1e-9) for c in E}
    R = sum(ratios.values()) / len(ratios)
    return {"points": max_points / (1.0 + R), "R": R, "ratios": ratios}