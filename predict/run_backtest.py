"""Walk-forward backtest of every model. Usage: .venv/bin/python -m predict.run_backtest [--plot out.png]"""
from __future__ import annotations

import argparse
import datetime as dt
from collections import defaultdict

from .data import DEFAULT_DATA, load_matches
from .evaluate import calibration, summarize, walk_forward
from .models import Constant, OnlineBias, OnlineScale, PlayerElo, PlayerGlicko, RegionalGlicko, Team1Bias, TeamElo
from .rankings import best_model, synthetic_rosters


def model_zoo(synthetic: bool = False):
    return [
        Constant(),
        Team1Bias(),
        TeamElo(k=20),
        TeamElo(k=40),
        PlayerElo(k=20),
        PlayerElo(k=40),
        PlayerElo(k=20, per_map=True),
        PlayerElo(k=30, per_map=True),
        PlayerElo(k=20, per_map=True, round_share=True),
        PlayerElo(k=30, per_map=True, round_share=True),
        PlayerElo(k=20, per_map=True, round_share=True, margin_weight=0.5),
        PlayerGlicko(),
        PlayerGlicko(start_rd=200, c=20),
        PlayerGlicko(start_rd=150, c=15),
        PlayerGlicko(start_rd=200, c=20, min_rd=50),
        PlayerGlicko(per_map=False),
        OnlineBias(PlayerElo(k=30, per_map=True)),
        OnlineBias(PlayerGlicko(start_rd=200, c=20)),
        OnlineBias(PlayerGlicko(start_rd=200, c=20), lr=0.005),
        OnlineScale(PlayerGlicko(start_rd=150, c=15)),
        RegionalGlicko(seed=True, offset=False),
        RegionalGlicko(seed=False, offset=True),
        RegionalGlicko(),
        OnlineScale(RegionalGlicko()),
        best_model(synthetic)[0],  # what rankings.py ships
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="matchdata JSON in Valve's schema")
    ap.add_argument("--eval-from", default="2023-03-01", help="first date whose matches count toward metrics")
    ap.add_argument("--min-history", type=int, default=0,
                    help="only score matches where every player has at least this many prior matches")
    ap.add_argument("--plot", default=None, help="write a calibration PNG for the best model")
    args = ap.parse_args()

    matches = load_matches(args.data)
    eval_from = int(dt.datetime.fromisoformat(args.eval_from).replace(tzinfo=dt.timezone.utc).timestamp())

    # Player match counts up to each match, to optionally filter out cold-start matches.
    seen = defaultdict(int)
    history_ok = {}
    for m in matches:
        players = m.team1_players + m.team2_players
        history_ok[id(m)] = min(seen[p] for p in players) >= args.min_history
        for p in players:
            seen[p] += 1

    print(f"matches: {len(matches)}  eval from {args.eval_from}  min-history={args.min_history}")
    print(f"{'model':48s} {'n':>5s} {'logloss':>8s} {'brier':>7s} {'acc':>6s} {'auc':>6s} {'ece':>6s}")
    results = []
    for model in model_zoo(synthetic_rosters(matches)):
        preds = walk_forward(model, matches, eval_from)
        preds = [x for x in preds if history_ok[id(x.match)]]
        s = summarize(preds)
        results.append((s["logloss"], model.name, preds))
        print(f"{model.name:48s} {s['n']:5d} {s['logloss']:8.4f} {s['brier']:7.4f} {s['acc']:6.3f} {s['auc']:6.3f} {s['ece']:6.3f}")

    results.sort(key=lambda x: x[0])
    best_ll, best_name, best_preds = results[0]
    print(f"\nbest by log loss: {best_name}")
    print(f"{'bin':>11s} {'n':>5s} {'pred':>6s} {'obs':>6s}")
    for lo, hi, n, mp, obs in calibration(best_preds, 10):
        print(f"{lo:4.1f}-{hi:4.1f}  {n:5d} {mp:6.3f} {obs:6.3f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        for _, name, preds in results[:3]:
            rows = calibration(preds, 10)
            ax.plot([r[3] for r in rows], [r[4] for r in rows], marker="o", label=name)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("predicted P(team1 wins)")
        ax.set_ylabel("observed win rate")
        ax.set_title("Calibration, walk-forward")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
