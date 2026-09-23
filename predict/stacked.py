"""Stacked logistic layer over the shipped blend, refit periodically on its own past predictions.

`Stacked(inner, glicko, batch)` wraps the shipped `OnlineScale(Blend(batch, glicko))`. For every match
it builds a feature row before anything updates (the blend and both halves' logits, Glicko team
RDs and the blend logit times their sum, experience, rest, recent form, format and event size), and once a label arrives the row joins the
training set. Every `refit_days` a standardized L2 logistic regression is refit on all rows collected
so far; until `min_rows` rows exist it passes the inner prediction through unchanged. Everything it
predicts is from rows strictly before the match, so it runs under `evaluate.walk_forward` as is.

Usage: .venv/bin/python -m predict.stacked [--data data/matchdata_pandascore.json] [--eval-from 2025-07-01]
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
from collections import defaultdict, deque

import numpy as np
from sklearn.linear_model import LogisticRegression

from .models import Model

DAY = 86400
NAMES = (["z_shipped", "z_batch", "z_glicko", "rd1", "rd2", "z_x_rd", "bo1", "bo3", "bo5", "logprize", "lan"]
         + [f"t{i}_{k}" for i in (1, 2) for k in ("rest", "logn", "plogn", "form", "nrec")])


def _logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


class Stacked(Model):
    def __init__(self, inner: Model, glicko, batch, refit_days: float = 30.0, min_rows: int = 1000,
                 C: float = 1.0, burn_in_days: float = 60.0):
        self.inner, self.glicko, self.batch = inner, glicko, batch
        self.refit, self.min_rows, self.C, self.burn_in = refit_days * DAY, min_rows, C, burn_in_days * DAY
        self.X: list[list[float]] = []
        self.y: list[float] = []
        self.coef = None                     # (mean, std, weights, intercept) of the last fit
        self.last_fit = -math.inf
        self.first = None                    # time of the first match seen
        self.last = {}                       # team -> last match time
        self.n = defaultdict(int)            # team -> series played
        self.pn = defaultdict(int)           # player -> series played
        self.recent = defaultdict(lambda: deque(maxlen=10))
        self._cache = (None, None, None)     # (match id, features, inner p) from the last predict
        self.name = f"stacked-lr[refit={refit_days:g}d,C={C:g},min={min_rows}]({inner.name})"

    def _team(self, tid, players, t):
        rest = (t - self.last[tid]) / DAY if tid in self.last else 30.0
        rec = self.recent[tid]
        plogn = sum(math.log1p(self.pn[p]) for p in players) / len(players)
        return [min(rest, 30.0), math.log1p(self.n[tid]), plogn,
                sum(rec) / len(rec) if rec else 0.5, float(len(rec))]

    def features(self, m):
        if self._cache[0] is m:
            return self._cache[1], self._cache[2]
        p = self.inner.predict(m)             # registers newcomers in Glicko before we read RDs
        _, rd1 = self.glicko.team(m.team1_players)
        _, rd2 = self.glicko.team(m.team2_players)
        z = _logit(p)
        # z_x_rd lets the stacker shrink the blend's logit when either lineup's rating is uncertain
        f = [z, _logit(self.batch.predict(m)), _logit(self.glicko.predict(m)), rd1 / 100, rd2 / 100, z * (rd1 + rd2) / 100]
        f += [1.0 if m.best_of == b else 0.0 for b in (1, 3, 5)]
        f += [math.log1p(m.prize_pool) / 10, 1.0 if m.lan else 0.0]
        f += self._team(m.team1_id, m.team1_players, m.time) + self._team(m.team2_id, m.team2_players, m.time)
        self._cache = (m, f, p)
        return f, p

    def _fit(self, now):
        self.last_fit = now
        if len(self.y) < self.min_rows:
            return
        X, y = np.array(self.X), np.array(self.y)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        lr = LogisticRegression(C=self.C, max_iter=2000).fit((X - mu) / sd, y)
        self.coef = (mu, sd, lr.coef_[0], lr.intercept_[0])

    def predict(self, m):
        f, p = self.features(m)
        if m.time - self.last_fit >= self.refit:
            self._fit(m.time)
        if self.coef is None:
            return p
        mu, sd, w, b = self.coef
        z = float(((np.array(f) - mu) / sd) @ w + b)
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, m):
        f, _ = self.features(m)
        if self.first is None:
            self.first = m.time
        if m.time - self.first >= self.burn_in:   # the first weeks' ratings are mostly priors
            self.X.append(f)
            self.y.append(1.0 if m.t1_won else 0.0)
        self._cache = (None, None, None)
        self.inner.update(m)
        for tid, players, won in ((m.team1_id, m.team1_players, m.t1_won), (m.team2_id, m.team2_players, not m.t1_won)):
            self.last[tid] = m.time
            self.n[tid] += 1
            self.recent[tid].append(1.0 if won else 0.0)
            for pl in players:
                self.pn[pl] += 1

    def coefficients(self):
        return {} if self.coef is None else dict(zip(NAMES, self.coef[2]))


def main():
    from .data import DEFAULT_DATA, load_matches
    from .evaluate import summarize, walk_forward
    from .rankings import best_model, synthetic_rosters

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--eval-from", default=None, help="default 2023-03-01 (Valve sample) / 2025-07-01 (PandaScore)")
    ap.add_argument("--burn-in", type=float, default=60.0, help="days before rows are collected")
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()
    matches = load_matches(args.data)
    syn = synthetic_rosters(matches)
    ev = args.eval_from or ("2025-07-01" if syn else "2023-03-01")
    t0 = int(dt.datetime.fromisoformat(ev).replace(tzinfo=dt.timezone.utc).timestamp())
    for make in (lambda: best_model(syn, stacked=False)[0],
                 lambda: Stacked(*best_model(syn, stacked=False), C=args.C, burn_in_days=args.burn_in)):
        model = make()
        s = summarize(walk_forward(model, matches, t0))
        print(f"{model.name[:60]:60s} logloss {s['logloss']:.4f}  acc {s['acc']:.3f}  auc {s['auc']:.3f}  ece {s['ece']:.3f}")
    print("last fit (standardized):", ", ".join(f"{k}={v:+.3f}" for k, v in
                                                  sorted(model.coefficients().items(), key=lambda x: -abs(x[1]))))


if __name__ == "__main__":
    main()
