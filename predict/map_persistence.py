"""Does a team's map-specific edge persist? Diagnostic behind the per-map offset question (see README).

Runs the shipped Glicko half walk-forward and records, for every map played, the residual (map won minus the
map-agnostic probability). For every (team id, map) with at least --min maps it splits that history into an
early and a late half and correlates the two halves' mean residuals across pairs, raw and after removing the
team's overall residual over the same maps. With --window-days the halves are consecutive windows of that
length instead (map pools and rosters change, so an edge might persist only short-term).

Usage: .venv/bin/python -m predict.map_persistence --data data/matchdata_liquipedia_sides.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from .data import load_matches
from .models import UNKNOWN_MAPS
from .rankings import best_model, synthetic_rosters


def residuals(matches, glicko):
    """[(time, team id, map, residual, team's residual on any map)] from team1's and team2's side."""
    out = []
    for m in matches:
        p = glicko.predict_map(m, "")
        for mp in m.maps:
            if mp.name in UNKNOWN_MAPS or mp.name == "unknown":
                continue
            r = (1.0 if mp.t1_won else 0.0) - p
            out.append((m.time, m.team1_id, mp.name, r))
            out.append((m.time, m.team2_id, mp.name, -r))
        glicko.update(m)
    return out


def persistence(rows, min_maps, window_days=None):
    by_pair = defaultdict(list)
    by_team = defaultdict(list)
    for t, team, mp, r in rows:
        by_pair[(team, mp)].append((t, r))
        by_team[team].append((t, r))
    a, b, a_adj, b_adj = [], [], [], []
    for (team, mp), xs in by_pair.items():
        if window_days:
            # consecutive windows: early = first window_days of this pair's history, late = the next window_days
            t0 = xs[0][0]
            w = window_days * 86400
            early = [x for x in xs if x[0] < t0 + w]
            late = [x for x in xs if t0 + w <= x[0] < t0 + 2 * w]
            if len(early) < min_maps // 2 or len(late) < min_maps // 2:
                continue
        else:
            if len(xs) < min_maps:
                continue
            h = len(xs) // 2
            early, late = xs[:h], xs[h:]
        team_all = by_team[team]

        def adj(part):
            lo, hi = part[0][0], part[-1][0]
            tr = [r for t, r in team_all if lo <= t <= hi]
            return np.mean([r for _, r in part]) - np.mean(tr)
        a.append(np.mean([r for _, r in early])); b.append(np.mean([r for _, r in late]))
        a_adj.append(adj(early)); b_adj.append(adj(late))
    n = len(a)
    return n, np.corrcoef(a, b)[0, 1], np.corrcoef(a_adj, b_adj)[0, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/matchdata_liquipedia_sides.json")
    ap.add_argument("--min", type=int, default=16, help="maps per (team, map) pair")
    args = ap.parse_args()
    matches = load_matches(args.data)
    _, glicko, _ = best_model(synthetic_rosters(matches), stacked=False)
    rows = residuals(matches, glicko)
    print(f"{len(rows) // 2} maps")
    print(f"{'split':28s} {'pairs':>6s} {'corr':>6s} {'corr adj':>9s}")
    for label, kw in [(f"halves, >= {args.min} maps", {}), (f"halves, >= {2 * args.min} maps", {"min_maps": 2 * args.min}),
                      ("90-day windows", {"window_days": 90}), ("180-day windows", {"window_days": 180})]:
        n, c, ca = persistence(rows, kw.pop("min_maps", args.min), **kw)
        print(f"{label:28s} {n:6d} {c:6.3f} {ca:9.3f}")


if __name__ == "__main__":
    main()
