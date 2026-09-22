"""Backtest map-aware models at both the series and the individual-map level.

Usage: .venv/bin/python -m predict.run_maps [--eval-from 2023-03-01] [--sweep]
"""
from __future__ import annotations

import argparse
import datetime as dt

from .data import DEFAULT_DATA, load_matches
from .evaluate import calibration, summarize, walk_forward, walk_forward_maps
from .models import PlayerGlicko, PlayerMapGlicko


def zoo(sweep: bool):
    models = [
        lambda: PlayerGlicko(start_rd=200, c=20),
        lambda: PlayerMapGlicko(k_map=10, shrink=0.01, known_maps=False),
        lambda: PlayerMapGlicko(k_map=10, shrink=0.01, known_maps=True),
    ]
    if sweep:
        for k in (5, 10, 20, 30):
            for sh in (0.0, 0.01, 0.03):
                models.append(lambda k=k, sh=sh: PlayerMapGlicko(k_map=k, shrink=sh, known_maps=True))
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="matchdata JSON in Valve's schema")
    ap.add_argument("--eval-from", default="2023-03-01")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    matches = load_matches(args.data)
    eval_from = int(dt.datetime.fromisoformat(args.eval_from).replace(tzinfo=dt.timezone.utc).timestamp())

    hdr = f"{'model':58s} {'n':>5s} {'logloss':>8s} {'brier':>7s} {'acc':>6s} {'auc':>6s} {'ece':>6s}"
    for level, fn in (("SERIES", walk_forward), ("MAP", walk_forward_maps)):
        print(f"\n== {level}-level predictions ==")
        print(hdr)
        rows = []
        for make in zoo(args.sweep):
            model = make()
            preds = fn(model, matches, eval_from)
            s = summarize(preds)
            rows.append((s["logloss"], model.name, preds))
            print(f"{model.name:58s} {s['n']:5d} {s['logloss']:8.4f} {s['brier']:7.4f} {s['acc']:6.3f} {s['auc']:6.3f} {s['ece']:6.3f}")
        rows.sort(key=lambda x: x[0])
        print(f"best: {rows[0][1]}")
        print(f"{'bin':>11s} {'n':>5s} {'pred':>6s} {'obs':>6s}")
        for lo, hi, n, mp, obs in calibration(rows[0][2], 10):
            print(f"{lo:4.1f}-{hi:4.1f}  {n:5d} {mp:6.3f} {obs:6.3f}")


if __name__ == "__main__":
    main()
