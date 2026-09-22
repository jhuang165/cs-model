"""Walk-forward evaluation and metrics."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .data import Match
from .models import Model

EPS = 1e-6


@dataclass
class Prediction:
    time: int
    p: float
    won: bool
    match: Match


def walk_forward(model: Model, matches: list[Match], eval_from: int) -> list[Prediction]:
    """Predict every match before updating on it; keep predictions from eval_from onward."""
    preds = []
    for m in matches:
        if m.time >= eval_from:
            preds.append(Prediction(m.time, model.predict(m), m.t1_won, m))
        model.update(m)
    return preds


def walk_forward_maps(model, matches, eval_from):
    """Like walk_forward but one prediction per played map (for models with predict_map)."""
    preds = []
    for m in matches:
        if m.time >= eval_from:
            for mp in m.maps:
                preds.append(Prediction(m.time, model.predict_map(m, mp.name), mp.t1_won, m))
        model.update(m)
    return preds


def log_loss(preds):
    return -sum(math.log(max(EPS, x.p if x.won else 1 - x.p)) for x in preds) / len(preds)


def brier(preds):
    return sum((x.p - (1.0 if x.won else 0.0)) ** 2 for x in preds) / len(preds)


def accuracy(preds):
    return sum(1 for x in preds if (x.p >= 0.5) == x.won) / len(preds)


def auc(preds):
    """Rank-based AUC of P(team1) against the team1-won label."""
    pos = sorted(x.p for x in preds if x.won)
    neg = sorted(x.p for x in preds if not x.won)
    if not pos or not neg:
        return float("nan")
    import bisect
    total = 0.0
    for p in pos:
        lo = bisect.bisect_left(neg, p)
        hi = bisect.bisect_right(neg, p)
        total += lo + 0.5 * (hi - lo)
    return total / (len(pos) * len(neg))


def calibration(preds, bins: int = 10):
    """Return rows of (bin_lo, bin_hi, n, mean_pred, observed_rate)."""
    rows = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        sel = [x for x in preds if lo <= x.p < hi or (i == bins - 1 and x.p == 1.0)]
        if sel:
            rows.append((lo, hi, len(sel), sum(x.p for x in sel) / len(sel), sum(x.won for x in sel) / len(sel)))
    return rows


def expected_calibration_error(preds, bins: int = 10):
    n = len(preds)
    return sum(c * abs(mp - obs) for _, _, c, mp, obs in calibration(preds, bins)) / n


def summarize(preds) -> dict:
    return {
        "n": len(preds),
        "logloss": log_loss(preds),
        "brier": brier(preds),
        "acc": accuracy(preds),
        "auc": auc(preds),
        "ece": expected_calibration_error(preds),
    }
