"""Grid-search Glicko parameters on a tuning window, confirm on a later held-out window.

Usage: .venv/bin/python -m predict.sweep --data data/matchdata_pandascore.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools

from .data import DEFAULT_DATA, load_matches
from .evaluate import summarize
from .models import OnlineScale, PlayerGlicko


def ts(d):
    return int(dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc).timestamp())


def run(make, matches, windows):
    """One walk-forward pass, returning summaries for several [start, end) windows."""
    model = make()
    preds = {w: [] for w in windows}
    from .evaluate import Prediction
    for m in matches:
        for w in windows:
            if w[0] <= m.time < w[1]:
                preds[w].append(Prediction(m.time, model.predict(m), m.t1_won, m))
        model.update(m)
    return {w: summarize(p) for w, p in preds.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--tune", default="2024-07-01,2025-07-01")
    ap.add_argument("--test", default="2025-07-01,2027-01-01")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    matches = load_matches(args.data)
    tune = tuple(ts(x) for x in args.tune.split(","))
    test = tuple(ts(x) for x in args.test.split(","))
    windows = [tune, test]

    grid = {
        "start_rd": [100, 150, 200, 250, 350],
        "c": [5, 10, 15, 20, 30, 45],
        "min_rd": [15, 30, 50, 75],
    }
    rows = []
    for start_rd, c, min_rd in itertools.product(*grid.values()):
        if min_rd >= start_rd:
            continue
        r = run(lambda: PlayerGlicko(start_rd=start_rd, c=c, min_rd=min_rd), matches, windows)
        rows.append(((start_rd, c, min_rd), r[tune]["logloss"], r[test]["logloss"], r[test]["acc"]))
    rows.sort(key=lambda x: x[1])
    print(f"{len(rows)} configs; tune window {args.tune}, test window {args.test}")
    print(f"{'(rd0, c, min_rd)':20s} {'tune ll':>8s} {'test ll':>8s} {'test acc':>8s}")
    for cfg, a, b, acc in rows[: args.top]:
        print(f"{str(cfg):20s} {a:8.4f} {b:8.4f} {acc:8.3f}")
    best = rows[0][0]
    ref = next(x for x in rows if x[0] == (200, 20, 30))
    print(f"\nprevious default (200, 20, 30): tune {ref[1]:.4f} test {ref[2]:.4f}")
    print(f"best on tune {best}: test {rows[0][2]:.4f}")

    # temperature on top of best
    for lr in (0.001, 0.002, 0.005):
        r = run(lambda: OnlineScale(PlayerGlicko(start_rd=best[0], c=best[1], min_rd=best[2]), lr=lr), matches, windows)
        print(f"best + OnlineScale(lr={lr}): tune {r[tune]['logloss']:.4f} test {r[test]['logloss']:.4f} ece {r[test]['ece']:.3f}")


if __name__ == "__main__":
    main()
