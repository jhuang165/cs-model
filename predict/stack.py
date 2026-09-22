"""Stacked classifier over walk-forward features from the base Glicko model.

Features are computed strictly before each match is used to update anything. The stacker is
trained on one window and evaluated on a later one, and compared with the base model on the
same evaluation window.

Usage: .venv/bin/python -m predict.stack --data data/matchdata_pandascore.json
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

    def __init__(self, tiers, **glicko_kw):
        self.model = PlayerGlicko(**glicko_kw)
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
        f = [
            logit(p_series), logit(p_map), (r1 - r2) / 100, rd1 / 100, rd2 / 100, (r1 + r2) / 2000,
            math.log1p(m.prize_pool) / 10,
        ]
        f += [1.0 if tier == t else 0.0 for t in TIERS]
        f += [1.0 if m.best_of == b else 0.0 for b in BOS]
        f += self.team_feats(m.team1_id, m.time) + self.team_feats(m.team2_id, m.time)
        if self.names is None:
            self.names = (["z_series", "z_map", "rdiff", "rd1", "rd2", "rmean", "logprize"]
                          + [f"tier_{t}" for t in TIERS] + [f"bo{b}" for b in BOS]
                          + [f"t1_{k}" for k in ("rest", "logn", "form", "nrec")]
                          + [f"t2_{k}" for k in ("rest", "logn", "form", "nrec")])
        return f, p_series

    def update(self, m):
        self.model.update(m)
        for tid, won in ((m.team1_id, m.t1_won), (m.team2_id, not m.t1_won)):
            self.last[tid] = m.time
            self.n[tid] += 1
            self.recent[tid].append(1.0 if won else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--train", default="2024-07-01,2025-07-01")
    ap.add_argument("--test", default="2025-07-01,2027-01-01")
    args = ap.parse_args()
    matches = load_matches(args.data)
    tiers = {e["eventId"]: e.get("tier") for e in json.load(open(args.data))["events"]}
    tr = tuple(ts(x) for x in args.train.split(","))
    te = tuple(ts(x) for x in args.test.split(","))

    fx = FeatureExtractor(tiers, start_rd=150, c=20, min_rd=30)
    X, y, base, split, meta = [], [], [], [], []
    for m in matches:
        if tr[0] <= m.time < te[1]:
            f, p = fx.row(m)
            X.append(f); y.append(1 if m.t1_won else 0); base.append(p)
            split.append("train" if m.time < tr[1] else "test"); meta.append(m)
        fx.update(m)
    X, y, base, split = np.array(X), np.array(y), np.array(base), np.array(split)
    Xtr, ytr, Xte, yte = X[split == "train"], y[split == "train"], X[split == "test"], y[split == "test"]
    print(f"train rows {len(Xtr)}  test rows {len(Xte)}  features {X.shape[1]}")

    def report(name, p):
        preds = [Prediction(m.time, float(pp), bool(yy), m) for pp, yy, m in zip(p, yte, [mm for mm, s in zip(meta, split) if s == "test"])]
        s = summarize(preds)
        print(f"{name:44s} logloss {s['logloss']:.4f}  brier {s['brier']:.4f}  acc {s['acc']:.3f}  auc {s['auc']:.3f}  ece {s['ece']:.3f}")
        return s

    report("base glicko (150,20,30)", base[split == "test"])

    # 1. logistic on the base logit only (= offline temperature + intercept)
    lr0 = LogisticRegression(C=1e6, max_iter=1000).fit(Xtr[:, :1], ytr)
    report(f"logit-only LR (a={lr0.coef_[0][0]:.3f}, b={lr0.intercept_[0]:.3f})", lr0.predict_proba(Xte[:, :1])[:, 1])

    # 2. full logistic
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(Xtr, ytr)
    report("full LR (all features)", lr.predict_proba(Xte)[:, 1])
    coefs = sorted(zip(fx.names, lr[-1].coef_[0]), key=lambda x: -abs(x[1]))
    print("   top |coef| (standardized):", ", ".join(f"{n}={c:+.3f}" for n, c in coefs[:10]))

    # 3. gradient boosting
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
                                        min_samples_leaf=100, l2_regularization=1.0, random_state=0).fit(Xtr, ytr)
    report("HistGradientBoosting (all features)", gb.predict_proba(Xte)[:, 1])

    # 4. ablation: LR without the base model's outputs (how much do the side features carry alone?)
    side = [i for i, n in enumerate(fx.names) if not n.startswith("z_") and n not in ("rdiff", "rmean")]
    lr_side = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(Xtr[:, side], ytr)
    report("LR side features only (no ratings)", lr_side.predict_proba(Xte[:, side])[:, 1])


if __name__ == "__main__":
    main()
