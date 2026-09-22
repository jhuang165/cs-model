"""Fit the best model on all data, print current ratings, and optionally price a head-to-head.

The model is the README's best: regional player Glicko blended with a daily-refit batch
Bradley-Terry (0.7 / 0.3 on series logits), under an online logit temperature. On data with
round scores both halves also fit round margins.

Usage:
  .venv/bin/python -m predict.rankings                 # top 30 rosters
  .venv/bin/python -m predict.rankings --top 50
  .venv/bin/python -m predict.rankings --vs "Vitality" "FaZe" --bo 3
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt

from .batch import ELO_PER_LOGIT, BatchBT
from .data import DEFAULT_DATA, load_matches
from .models import Blend, OnlineScale, RegionalGlicko, series_prob
from .regions import team_region

BATCH_WEIGHT = 0.3


def synthetic_rosters(matches) -> bool:
    """PandaScore exports stand a single `team:<id>` entry in for the lineup."""
    return any(p.startswith("team:") for m in matches[:50] for p in m.team1_players)


def best_model(synthetic: bool):
    # Tuned per dataset (see README): entities are players on the Valve sample and teams on
    # PandaScore, which moves the batch fit's best half-life and ridge strength.
    if synthetic:
        glicko, batch = RegionalGlicko(start_rd=150, c=20), BatchBT(tau_days=180, C=3)
    else:
        # real round scores: both halves also learn from round margins (PandaScore maps are 1-0)
        glicko = RegionalGlicko(start_rd=200, c=20, round_weight=1.0, round_scale=0.25)
        batch = BatchBT(tau_days=365, C=3, round_weight=2.0, round_scale=0.25)
    return OnlineScale(Blend(batch, glicko, w=BATCH_WEIGHT)), glicko, batch


def current_lineups(matches):
    """Latest lineup per source teamId, dropping teams whose players have moved on.

    A team is kept only if a majority of its latest lineup last played for it, so a roster that
    was signed wholesale (Outsiders -> Virtus.pro) or broken up shows once, under its current team.
    """
    lineups, player_team = {}, {}
    for m in matches:  # time-ordered, so later entries overwrite
        for tid, name, players, countries in ((m.team1_id, m.team1_name, m.team1_players, m.team1_countries),
                                              (m.team2_id, m.team2_name, m.team2_players, m.team2_countries)):
            lineups[tid] = (name, players, countries, m.time)
            for p in players:
                player_team[p] = tid
    return {tid: v for tid, v in lineups.items()
            if sum(player_team[p] == tid for p in v[1]) * 2 > len(v[1])}


def display_rating(model, glicko, batch, players, region):
    """Blend of the two halves on the Elo scale, times the learned temperature.

    Only differences are meaningful. This is linear in map strength, so it ranks teams the way
    the blend does, but head-to-head prices should come from model.predict (series logits).
    """
    rg, rd = glicko.team(players)
    rg += glicko.o[region]
    rb = batch.rating(players)
    if batch.region and region in batch.ridx:
        n_ent = len(batch.theta) - len(batch.ridx)
        rb += ELO_PER_LOGIT * batch._theta(n_ent + batch.ridx[region])
    return 1500 + model.a * (BATCH_WEIGHT * (rb - 1500) + (1 - BATCH_WEIGHT) * (rg - 1500)), rd


def implied_map_prob(p_series: float, best_of: int) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if series_prob(mid, best_of) < p_series else (lo, mid)
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="matchdata JSON in Valve's schema")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--min-matches", type=int, default=10, help="hide teams with fewer matches")
    ap.add_argument("--active-days", type=float, default=180,
                    help="hide teams that have not played in this many days before the last match")
    ap.add_argument("--vs", nargs=2, metavar=("TEAM_A", "TEAM_B"), help="team names (substring match)")
    ap.add_argument("--bo", type=int, default=3)
    args = ap.parse_args()

    matches = load_matches(args.data)
    model, glicko, batch = best_model(synthetic_rosters(matches))
    counts = {}
    for m in matches:
        model.update(m)
        counts[m.team1_id] = counts.get(m.team1_id, 0) + 1
        counts[m.team2_id] = counts.get(m.team2_id, 0) + 1
    now = matches[-1].time
    batch.fit(now + 1)  # include the last day's maps

    rows = []
    for tid, (name, players, countries, last) in current_lineups(matches).items():
        if counts[tid] < args.min_matches or now - last > args.active_days * 86400:
            continue
        region = team_region(countries)
        r, rd = display_rating(model, glicko, batch, players, region)
        rows.append((r, rd, name, counts[tid], last, region, players, countries))
    rows.sort(reverse=True)

    if args.vs:
        def find(q):
            hits = [x for x in rows if q.lower() in x[2].lower()]
            if not hits:
                raise SystemExit(f"no active team matching {q!r}")
            exact = [x for x in hits if x[2].lower() == q.lower()]
            return (exact or hits)[0]
        a, b = find(args.vs[0]), find(args.vs[1])
        m = dataclasses.replace(matches[-1], time=now + 1, team1_name=a[2], team2_name=b[2],
                                team1_players=a[6], team2_players=b[6],
                                team1_countries=a[7], team2_countries=b[7], best_of=args.bo)
        p = model.predict(m)
        print(f"{a[2]} ({a[0]:.0f}±{a[1]:.0f}) vs {b[2]} ({b[0]:.0f}±{b[1]:.0f})")
        print(f"BO{args.bo} P({a[2]}) = {p:.3f}   implied per-map P = {implied_map_prob(p, args.bo):.3f}")
        return

    asof = dt.datetime.fromtimestamp(now, dt.timezone.utc).date()
    print(f"ratings as of {asof} (regional Glicko + batch Bradley-Terry blend, temperature {model.a:.2f};"
          f" rating incl. region effects ± Glicko team RD)")
    print(f"{'#':>3s} {'team':28s} {'rating':>7s} {'rd':>4s} {'games':>5s} {'reg':>4s}  last played")
    for i, (r, rd, name, n, last, region, *_) in enumerate(rows[: args.top], 1):
        d = dt.datetime.fromtimestamp(last, dt.timezone.utc).date()
        print(f"{i:3d} {name[:28]:28s} {r:7.0f} {rd:4.0f} {n:5d} {region:>4s}  {d}")
    print("Glicko region offsets:", ", ".join(f"{k} {v:+.0f}" for k, v in sorted(glicko.o.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
