"""Stacked classifier over walk-forward features from the base Glicko model.

Features are computed strictly before each match is used to update anything. The stacker is
trained on one window and evaluated on a later one, and compared with the base model on the
same evaluation window.

Also refits the shipped blend weight and temperature jointly (z_batch, z_glicko), once and monthly.

Usage: .venv/bin/python -m predict.stack [--data data/matchdata_pandascore.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from collections import defaultdict, deque

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .data import DEFAULT_DATA, load_matches
from .evaluate import Prediction, summarize
from .models import PlayerGlicko, elo_expected, series_prob
from .rankings import best_model, synthetic_rosters

TIERS = ["s", "a", "b", "c", "d"]
BOS = [1, 3, 5]
DAY = 86400


def ts(d):
    return int(dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc).timestamp())


def logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


class FeatureExtractor:
    """Runs the base model walk-forward and emits a feature row per match before updating."""

    def __init__(self, tiers, synthetic, **glicko_kw):
        self.model = PlayerGlicko(**glicko_kw)
        # the shipped model and its two halves, so the blend weight and temperature can be refit
        self.shipped, self.glicko, self.batch = best_model(synthetic, stacked=False)
        self.tiers = tiers
        self.last = {}                      # team -> last match time
        self.n = defaultdict(int)           # team -> series played
        self.recent = defaultdict(lambda: deque(maxlen=10))  # team -> last 10 series results
        self.names = None

    def team_feats(self, tid, t):
        rest = (t - self.last[tid]) / DAY if tid in self.last else 30.0
        rec = self.recent[tid]
        form = (sum(rec) / len(rec)) if rec else 0.5
        return [min(rest, 30.0), math.log1p(self.n[tid]), form, len(rec)]

    def row(self, m):
        r1, rd1 = self.model.team(m.team1_players)
        r2, rd2 = self.model.team(m.team2_players)
        p_map = elo_expected(self.model.g(math.sqrt(rd1 ** 2 + rd2 ** 2)) * (r1 - r2))
        p_series = series_prob(p_map, m.best_of)
        tier = self.tiers.get(m.event_id)
        shipped = self.shipped.predict(m)
        f = [
            logit(self.batch.predict(m)), logit(self.glicko.predict(m)), logit(shipped),
            logit(p_series), logit(p_map), (r1 - r2) / 100, rd1 / 100, rd2 / 100, (r1 + r2) / 2000,
            math.log1p(m.prize_pool) / 10,
        ]
        f += [1.0 if tier == t else 0.0 for t in TIERS]
        f += [1.0 if m.best_of == b else 0.0 for b in BOS]
        f += self.team_feats(m.team1_id, m.time) + self.team_feats(m.team2_id, m.time)
        if self.names is None:
            self.names = (["z_batch", "z_glicko", "z_shipped", "z_series", "z_map", "rdiff", "rd1", "rd2", "rmean", "logprize"]
                          + [f"tier_{t}" for t in TIERS] + [f"bo{b}" for b in BOS]
                          + [f"t1_{k}" for k in ("rest", "logn", "form", "nrec")]
                          + [f"t2_{k}" for k in ("rest", "logn", "form", "nrec")])
        return f, p_series, shipped

    def update(self, m):
        self.model.update(m)
        self.shipped.update(m)
        for tid, won in ((m.team1_id, m.t1_won), (m.team2_id, not m.t1_won)):
            self.last[tid] = m.time
            self.n[tid] += 1
            self.recent[tid].append(1.0 if won else 0.0)


def walk_fit(make, X, y, t, fit_from, test_mask, every_days=30):
    """Refit `make()` every `every_days` on all rows in [fit_from, cutoff) and predict the next block.

    This is how a stacker would run live: its test-window predictions only use rows before them.
    """
    p = np.full(len(y), np.nan)
    idx = np.where(test_mask)[0]
    start = t[idx[0]]
    while start <= t[idx[-1]]:
        end = start + every_days * DAY
        block = test_mask & (t >= start) & (t < end)
        if block.any():
            fit = (t >= fit_from) & (t < start)
            p[block] = make().fit(X[fit], y[fit]).predict_proba(X[block])[:, 1]
        start = end
    return p[test_mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--train", default=None, help="start,end of the stacker's training window "
                    "(default 2024-07-01,2025-07-01 on PandaScore, 2022-11-01,2023-03-01 on the Valve sample)")
    ap.add_argument("--test", default=None, help="start,end of the evaluation window")
    args = ap.parse_args()
    matches = load_matches(args.data)
    synthetic = synthetic_rosters(matches)
    if synthetic:
        train, test, rd0 = args.train or "2024-07-01,2025-07-01", args.test or "2025-07-01,2027-01-01", 150
    else:
        train, test, rd0 = args.train or "2022-11-01,2023-03-01", args.test or "2023-03-01,2024-01-01", 200
    tiers = {e["eventId"]: e.get("tier") for e in json.load(open(args.data))["events"]}
    tr = tuple(ts(x) for x in train.split(","))
    te = tuple(ts(x) for x in test.split(","))

    fx = FeatureExtractor(tiers, synthetic, start_rd=rd0, c=20, min_rd=30)
    X, y, base, shipped, times, split, meta = [], [], [], [], [], [], []
    for m in matches:
        if tr[0] <= m.time < te[1]:
            f, p, ps = fx.row(m)
            X.append(f); y.append(1 if m.t1_won else 0); base.append(p); shipped.append(ps); times.append(m.time)
            split.append("train" if m.time < tr[1] else "test"); meta.append(m)
        else:
            fx.shipped.predict(m)  # keep the online temperature's predict/update order intact
        fx.update(m)
    X, y, base, shipped, t, split = (np.array(v) for v in (X, y, base, shipped, times, split))
    trm, tem = split == "train", split == "test"
    Xtr, ytr, Xte, yte = X[trm], y[trm], X[tem], y[tem]
    col = {n: i for i, n in enumerate(fx.names)}
    cols = lambda *ns: [col[n] for n in ns]
    print(f"train rows {len(Xtr)}  test rows {len(Xte)}  features {X.shape[1]}  (train {train}, test {test})")
    test_meta = [mm for mm, s in zip(meta, split) if s == "test"]

    def report(name, p):
        preds = [Prediction(m.time, float(pp), bool(yy), m) for pp, yy, m in zip(p, yte, test_meta)]
        s = summarize(preds)
        print(f"{name:52s} logloss {s['logloss']:.4f}  brier {s['brier']:.4f}  acc {s['acc']:.3f}  auc {s['auc']:.3f}  ece {s['ece']:.3f}")
        return s

    report(f"base glicko ({rd0},20,30)", base[tem])
    report(f"shipped: online scale(0.3 batch + 0.7 glicko)", shipped[tem])

    # 0. blend weight and temperature fitted jointly: P = sigmoid(a_b z_batch + a_g z_glicko [+ b]).
    #    w = a_b / (a_b + a_g) is the blend weight, a_b + a_g the temperature.
    zbg = cols("z_batch", "z_glicko")
    for icpt in (False, True):
        lr_bg = LogisticRegression(C=1e6, fit_intercept=icpt, max_iter=1000).fit(Xtr[:, zbg], ytr)
        ab, ag = lr_bg.coef_[0]
        tag = f", b={lr_bg.intercept_[0]:+.3f}" if icpt else ""
        report(f"joint LR (w={ab / (ab + ag):.2f}, a={ab + ag:.2f}{tag})", lr_bg.predict_proba(Xte[:, zbg])[:, 1])
    # same, refit monthly on everything before (what a live refit would see)
    raw = lambda: LogisticRegression(C=1e6, fit_intercept=False, max_iter=1000)
    report("joint LR, refit monthly", walk_fit(raw, X[:, zbg], y, t, tr[0], tem))
    # grid over w with the temperature fitted for each, to see how flat the surface is
    for w in (0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        ztr = (w * Xtr[:, col["z_batch"]] + (1 - w) * Xtr[:, col["z_glicko"]])[:, None]
        zte = (w * Xte[:, col["z_batch"]] + (1 - w) * Xte[:, col["z_glicko"]])[:, None]
        lr_w = LogisticRegression(C=1e6, fit_intercept=False, max_iter=1000).fit(ztr, ytr)
        report(f"   w={w:.1f}, offline temperature a={lr_w.coef_[0][0]:.2f}", lr_w.predict_proba(zte)[:, 1])

    # 1. logistic on the base logit only (= offline temperature + intercept)
    z0 = cols("z_series")
    lr0 = LogisticRegression(C=1e6, max_iter=1000).fit(Xtr[:, z0], ytr)
    report(f"glicko logit-only LR (a={lr0.coef_[0][0]:.3f}, b={lr0.intercept_[0]:.3f})", lr0.predict_proba(Xte[:, z0])[:, 1])

    # 2. full logistic (with and without the blend halves)
    old = [i for i, n in enumerate(fx.names) if n not in ("z_batch", "z_glicko", "z_shipped")]
    full = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    report("full LR, glicko features only", full().fit(Xtr[:, old], ytr).predict_proba(Xte[:, old])[:, 1])
    lr = full().fit(Xtr, ytr)
    report("full LR, all features incl. batch/blend logits", lr.predict_proba(Xte)[:, 1])
    coefs = sorted(zip(fx.names, lr[-1].coef_[0]), key=lambda x: -abs(x[1]))
    print("   top |coef| (standardized):", ", ".join(f"{n}={c:+.3f}" for n, c in coefs[:10]))
    report("full LR, all features, refit monthly", walk_fit(full, X, y, t, tr[0], tem))

    # 3. gradient boosting
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
                                        min_samples_leaf=100, l2_regularization=1.0, random_state=0).fit(Xtr, ytr)
    report("HistGradientBoosting (all features)", gb.predict_proba(Xte)[:, 1])

    # 4. ablation: LR without the base model's outputs (how much do the side features carry alone?)
    side = [i for i, n in enumerate(fx.names) if not n.startswith("z_") and n not in ("rdiff", "rmean")]
    lr_side = full().fit(Xtr[:, side], ytr)
    report("LR side features only (no ratings)", lr_side.predict_proba(Xte[:, side])[:, 1])


if __name__ == "__main__":
    main()
