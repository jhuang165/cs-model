"""Time-weighted batch Bradley-Terry, refit on a schedule and evaluated walk-forward.

The online models only let a result inform the ratings of matches that come after it. A batch
fit uses every result to inform every rating: a January upset changes what we think about
March, and a team's whole trailing year is weighed at once. Recency is handled by exponentially
down-weighting old maps (half-life `tau_days`) instead of by an update step.

Model: per map, P(team1 wins) = sigmoid(theta_1 - theta_2 [+ region_1 - region_2]) where a
team's theta is the mean of its players' thetas. Fitted as a weighted L2-regularised logistic
regression on a sparse design matrix; the ridge shrinks every entity toward zero, which with
the region columns means "toward its region's mean" - the hierarchical regional prior.

It plugs into the same walk-forward loop as the online models: predict() refits when the last
fit is older than `refit_days`, using only maps that started before the match being predicted.

Usage: .venv/bin/python -m predict.batch --data data/matchdata_pandascore.json --eval-from 2025-07-01
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import time

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from .data import DEFAULT_DATA, Match, load_matches
from .evaluate import calibration, summarize, walk_forward
from .models import Model, OnlineScale, series_prob
from .regions import country_region

DAY = 86400.0
ELO_PER_LOGIT = 400.0 / math.log(10)


class BatchBT(Model):
    def __init__(self, tau_days: float = 180.0, refit_days: float = 1.0, C: float = 3.0,
                 region: bool = True, region_C: float | None = None, intercept: bool = False,
                 min_weight: float = 0.02, warm: bool = True, round_weight: float = 0.0,
                 round_scale: float = 0.25):
        # round_weight > 0 adds a margin likelihood: each round of a valid map is a Bernoulli with
        # logit round_scale * (theta_1 - theta_2), weighted by round_weight because rounds within a
        # map are correlated (economy, momentum) and carry less than one independent observation each.
        self.round_weight, self.round_scale = round_weight, round_scale
        self.tau, self.refit, self.C, self.region = tau_days * DAY, refit_days * DAY, C, region
        self.region_C = region_C or C   # region columns see thousands of maps, so the ridge barely binds anyway
        self.intercept, self.min_weight, self.warm = intercept, min_weight, warm
        self.idx: dict[str, int] = {}          # entity -> column
        self.ridx: dict[str, int] = {}         # region -> column (after entities)
        self.entity_region: dict[str, str] = {}
        # (time, cols1, cols2, y, weight multiplier, design scale); round rows use scale round_scale
        self.obs: list[tuple[float, tuple[int, ...], tuple[int, ...], float, float, float]] = []
        self.last_fit = -math.inf
        self.theta = np.zeros(0)
        self.b = 0.0
        self.fits = 0
        self.fit_seconds = 0.0
        tag = f"tau={tau_days:g}d,refit={refit_days:g}d,C={C:g}" + (",region" if region else "") + (",b" if intercept else "")
        if round_weight:
            tag += f",rounds={round_weight:g}x{round_scale:g}"
        self.name = f"batch-bt[{tag}]"

    # ---- bookkeeping -------------------------------------------------------------------------
    def _cols(self, players, countries):
        out = []
        for p, cc in zip(players, countries):
            if p not in self.idx:
                self.idx[p] = len(self.idx)
                self.entity_region[p] = country_region(cc)
            out.append(self.idx[p])
        return tuple(out)

    def _rcol(self, reg):
        if reg not in self.ridx:
            self.ridx[reg] = len(self.ridx)
        return self.ridx[reg]

    def _theta(self, col):
        return self.theta[col] if col < len(self.theta) else 0.0

    # ---- fitting -----------------------------------------------------------------------------
    def fit(self, now: float):
        t0 = time.time()
        keep = [o for o in self.obs if math.exp(-(now - o[0]) / self.tau) >= self.min_weight]
        self.obs = keep
        n_ent, n_reg = len(self.idx), len(self.ridx)
        n_cols = n_ent + (n_reg if self.region else 0)
        if len(keep) < 50 or n_cols == 0 or len({o[3] for o in keep}) < 2:
            return
        rows, cols, vals, y, w = [], [], [], [], []
        for i, (t, c1, c2, out, wm, sc) in enumerate(keep):
            for c in c1:
                rows.append(i); cols.append(c); vals.append(sc / len(c1))
            for c in c2:
                rows.append(i); cols.append(c); vals.append(-sc / len(c2))
            if self.region:
                r1 = self._rcol(self.entity_region[self._ent(c1[0])])
                r2 = self._rcol(self.entity_region[self._ent(c2[0])])
                if r1 != r2:
                    rows += [i, i]; cols += [n_ent + r1, n_ent + r2]; vals += [sc, -sc]
            y.append(out)
            w.append(wm * math.exp(-(now - t) / self.tau))
        n_cols = n_ent + (len(self.ridx) if self.region else 0)
        X = sp.csr_matrix((vals, (rows, cols)), shape=(len(keep), n_cols))
        y, w = np.array(y), np.array(w)
        # per-column penalty: scale region columns up so their effective C is region_C
        if self.region and len(self.ridx):
            scale = np.ones(n_cols)
            scale[n_ent:] = math.sqrt(self.region_C / self.C)
            X = X @ sp.diags(scale)
        else:
            scale = np.ones(n_cols)
        lr = LogisticRegression(C=self.C, fit_intercept=self.intercept, solver="lbfgs", max_iter=3000,
                                tol=1e-5, warm_start=self.warm)
        if self.warm and len(self.theta):
            lr.coef_ = np.zeros((1, n_cols)); lr.coef_[0, :len(self.theta)] = self.theta[:n_cols] / scale[:len(self.theta)]
            lr.intercept_ = np.array([self.b])
        lr.fit(X, y, sample_weight=w)
        self.theta = lr.coef_[0] * scale
        self.b = float(lr.intercept_[0]) if self.intercept else 0.0
        self.last_fit = now
        self.fits += 1
        self.fit_seconds += time.time() - t0

    def _ent(self, col):
        if not hasattr(self, "_inv") or len(self._inv) != len(self.idx):
            self._inv = {v: k for k, v in self.idx.items()}
        return self._inv[col]

    # ---- Model interface ---------------------------------------------------------------------
    def logit(self, m: Match) -> float:
        c1 = self._cols(m.team1_players, m.team1_countries)
        c2 = self._cols(m.team2_players, m.team2_countries)
        n_ent = len(self.theta) - (len(self.ridx) if self.region else 0)
        z = sum(self._theta(c) for c in c1) / len(c1) - sum(self._theta(c) for c in c2) / len(c2)
        if self.region:
            r1, r2 = m.team1_region, m.team2_region
            if r1 != r2:
                for reg, sign in ((r1, 1.0), (r2, -1.0)):
                    if reg in self.ridx and n_ent + self.ridx[reg] < len(self.theta):
                        z += sign * self.theta[n_ent + self.ridx[reg]]
        return z + self.b

    def predict(self, m: Match) -> float:
        if m.time - self.last_fit >= self.refit:
            self.fit(m.time)
        p = 1.0 / (1.0 + math.exp(-self.logit(m)))
        return series_prob(p, m.best_of)

    def update(self, m: Match) -> None:
        c1 = self._cols(m.team1_players, m.team1_countries)
        c2 = self._cols(m.team2_players, m.team2_countries)
        for mp in m.maps:
            self.obs.append((m.time, c1, c2, 1.0 if mp.t1_won else 0.0, 1.0, 1.0))
            if self.round_weight and mp.valid_for_margin:
                self.obs.append((m.time, c1, c2, 1.0, self.round_weight * mp.t1, self.round_scale))
                self.obs.append((m.time, c1, c2, 0.0, self.round_weight * mp.t2, self.round_scale))

    def rating(self, players) -> float:
        """Elo-scale rating (1500 = region-neutral zero) for display."""
        cols = [self.idx[p] for p in players if p in self.idx]
        if not cols:
            return 1500.0
        return 1500.0 + ELO_PER_LOGIT * sum(self._theta(c) for c in cols) / len(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--eval-from", default="2023-03-01")
    ap.add_argument("--tau", default="90,180,365", help="comma list of half-lives in days")
    ap.add_argument("--C", default="3", help="comma list of inverse ridge strengths")
    ap.add_argument("--refit-days", type=float, default=1.0)
    ap.add_argument("--no-region", action="store_true")
    ap.add_argument("--scale", action="store_true", help="also report each config wrapped in OnlineScale")
    args = ap.parse_args()
    matches = load_matches(args.data)
    eval_from = int(dt.datetime.fromisoformat(args.eval_from).replace(tzinfo=dt.timezone.utc).timestamp())
    print(f"{args.data}: {len(matches)} matches, eval from {args.eval_from}")
    print(f"{'model':60s} {'n':>5s} {'logloss':>8s} {'brier':>7s} {'acc':>6s} {'auc':>6s} {'ece':>6s} {'fits':>5s} {'sec':>6s}")
    best = None
    for tau in [float(x) for x in args.tau.split(",")]:
        for C in [float(x) for x in args.C.split(",")]:
            for region in ([False] if args.no_region else [True, False]):
                for scaled in ([False, True] if args.scale else [False]):
                    inner = BatchBT(tau_days=tau, refit_days=args.refit_days, C=C, region=region)
                    model = OnlineScale(inner) if scaled else inner
                    preds = walk_forward(model, matches, eval_from)
                    s = summarize(preds)
                    print(f"{model.name:60s} {s['n']:5d} {s['logloss']:8.4f} {s['brier']:7.4f} {s['acc']:6.3f} {s['auc']:6.3f} {s['ece']:6.3f} {inner.fits:5d} {inner.fit_seconds:6.1f}", flush=True)
                    if best is None or s["logloss"] < best[0]:
                        best = (s["logloss"], model.name, preds, inner)
    ll, name, preds, inner = best
    print(f"\nbest: {name}")
    for lo, hi, n, mp, obs in calibration(preds, 10):
        print(f"{lo:4.1f}-{hi:4.1f}  {n:5d} {mp:6.3f} {obs:6.3f}")
    if inner.region and inner.ridx:
        n_ent = len(inner.theta) - len(inner.ridx)
        print("region effects (Elo points):", {r: round(ELO_PER_LOGIT * inner.theta[n_ent + i]) for r, i in inner.ridx.items()})


if __name__ == "__main__":
    main()
