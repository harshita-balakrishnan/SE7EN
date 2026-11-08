"""Where is the structure?  python3 explore2.py public   (about 1-2 min, prints a small table)
For each long-history machine (train split) compare the strongest effect found in REAL history with the strongest
found after SHUFFLING the same machine's outcomes (no structure by construction). |z| = effect / standard error.
Families: single-lag win, single-lag big payout (>=2x), pair of lags (win_a x win_b interaction), pair where one lag is 7."""
import random, sys
from core import load_split

ms = [m for m in load_split(sys.argv[1] + "/train") if not m.short]
rng = random.Random(0)
LAGS = range(1, 9)
PAIRS = [(a, b) for a in LAGS for b in LAGS if a < b]


def z2(cells):  # cells: [[n, s]] for a 2x2 table keyed 0..3 ; interaction contrast z
    r = []
    var = 0.0
    for n, s in cells:
        if n < 5:
            return 0.0
        p = s / n
        r.append(p)
        var += max(p * (1 - p), 0.02) / n
    return (r[3] - r[2] - r[1] + r[0]) / (var ** 0.5)


def z1(cells):
    (n0, s0), (n1, s1) = cells
    if n0 < 5 or n1 < 5:
        return 0.0
    p0, p1 = s0 / n0, s1 / n1
    v = max(p0 * (1 - p0), .02) / n0 + max(p1 * (1 - p1), .02) / n1
    return (p1 - p0) / v ** 0.5


def stats(win, big, ticks):
    out = {}
    best1 = best1b = bestp = bestp7 = 0.0
    arg1 = argp = None
    for k in LAGS:
        c = [[0, 0], [0, 0]]
        cb = [[0, 0], [0, 0]]
        for t in ticks:
            if (t - k) in win:
                i = win[t - k]
                c[i][0] += 1; c[i][1] += win[t]
                j = big[t - k]
                cb[j][0] += 1; cb[j][1] += win[t]
        z = abs(z1(c)); zb = abs(z1(cb))
        if z > best1: best1, arg1 = z, k
        best1b = max(best1b, zb)
    for a, b in PAIRS:
        c = [[0, 0] for _ in range(4)]
        for t in ticks:
            if (t - a) in win and (t - b) in win:
                i = 2 * win[t - a] + win[t - b]
                c[i][0] += 1; c[i][1] += win[t]
        z = abs(z2(c))
        if z > bestp: bestp, argp = z, (a, b)
        if 7 in (a, b): bestp7 = max(bestp7, z)
    return best1, best1b, bestp, bestp7, arg1, argp


real = [0.0] * 4; null = [0.0] * 4; hit = [0] * 4; nhit = [0] * 4
args1, argsp = {}, {}
for m in ms:
    ticks = sorted(t for t in m.plays if t >= 8)
    win = {t: 1 if k > 0 else 0 for t, k in m.plays.items()}
    big = {t: 1 if k >= 2 else 0 for t, k in m.plays.items()}
    r = stats(win, big, ticks)
    vals = list(win.values()); rng.shuffle(vals)
    win2 = dict(zip(win.keys(), vals))
    bv = list(big.values()); rng.shuffle(bv)
    big2 = dict(zip(big.keys(), bv))
    n = stats(win2, big2, ticks)
    for i in range(4):
        real[i] += r[i]; null[i] += n[i]; hit[i] += r[i] > 3; nhit[i] += n[i] > 3
    args1[r[4]] = args1.get(r[4], 0) + 1
    argsp[r[5]] = argsp.get(r[5], 0) + 1
N = len(ms)
print("long-history machines:", N)
print("%-28s %10s %10s %12s %12s" % ("family (best |z| per machine)", "real mean", "null mean", "real |z|>3", "null |z|>3"))
for i, name in enumerate(["single lag: win", "single lag: big payout", "pair interaction (28)", "pair interaction w/ lag 7"]):
    print("%-28s %10.2f %10.2f %11.0f%% %11.0f%%" % (name, real[i] / N, null[i] / N, 100 * hit[i] / N, 100 * nhit[i] / N))
print("most common best single lag:", sorted(args1.items(), key=lambda kv: -kv[1])[:4])
print("most common best pair:", sorted(argsp.items(), key=lambda kv: -kv[1])[:6])