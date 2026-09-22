"""Score Valve's own ranking model walk-forward on the same matches and metrics as the Python models.

Every `--step` days Valve's standings are rebuilt (model/ranking.js, six-month window, prize and
network seeding, fixed-RD Glicko) from matches before that date, and the following week's
matches are priced from the two rosters' rank values with the formula in model/fit.js. Matches
where either roster is not in the standings get no Valve prediction; they are scored at 0.5 in
the "all matches" row and excluded in the "matched" rows.

PandaScore files have one synthetic player per team, which Valve's loader would drop, so they
are padded to five copies of that player before being handed to node. Matches after 2025-01-01
are also marked valveRanked so the loader keeps them, and each event's winner is credited with
the prize pool so Valve's prize-based seeding is not all zeros (NaN otherwise), and tier S/A
events stand in for LAN events, which the free tier does not flag.

Usage: .venv/bin/python -m predict.valve_baseline [--data ...] [--eval-from 2023-03-01] [--eval-to ...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .data import DEFAULT_DATA, load_matches
from .evaluate import Prediction, calibration, summarize, walk_forward
from .models import OnlineScale, PlayerGlicko, RegionalGlicko
import math


def online_tempered(preds, lr=0.002):
    """Valve's expectation with a logit temperature learned online (same rule as OnlineScale),
    so its ranking quality can be judged apart from its calibration."""
    a, out = 1.0, []
    for x in sorted(preds, key=lambda x: x.time):
        z = math.log(max(1e-6, min(1 - 1e-6, x.p)) / (1 - max(1e-6, min(1 - 1e-6, x.p))))
        p = 1 / (1 + math.exp(-a * z))
        out.append(Prediction(x.time, p, x.won, x.match))
        a += lr * ((1.0 if x.won else 0.0) - p) * z
    return out, a

JS = Path(__file__).resolve().parent / "valve_baseline.js"


def ts(d):
    return int(dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc).timestamp())


def prepare(path: str, out_dir: str) -> str:
    raw = json.loads(Path(path).read_text())
    padded = 0
    for m in raw["matches"]:
        m["valveRanked"] = True
        for k in ("team1Players", "team2Players"):
            if len(m[k]) == 1:
                m[k] = [dict(m[k][0], playerId=f"{m[k][0]['playerId']}#{j}") for j in range(5)]
                padded += 1
    for e in raw["events"]:
        e.setdefault("finished", True)
        e.setdefault("prizeDistribution", [])
        e.setdefault("lan", False)
        if not e["prizeDistribution"] and e.get("winnerTeamId") and e.get("prizePool"):
            # PandaScore export: no placements, but the tournament winner is known. Give it the
            # pool so Valve's prize-based seeding has something to work with.
            e["prizeDistribution"] = [{"teamId": e["winnerTeamId"], "placement": 1, "prize": int(e["prizePool"].strip("$")),
                                       "clubShare": 0, "shared": False, "qualifiedEvents": []}]
        for t in e["prizeDistribution"]:  # the 2023 sample predates these fields
            t.setdefault("qualifiedEvents", [])
            t.setdefault("clubShare", 0)
            t.setdefault("prize", 0)
            t.setdefault("shared", False)
    if not any(e.get("lan") for e in raw["events"]):
        # PandaScore's free tier has no LAN flag and Valve's LAN factor divides by the reference
        # LAN count; tier S/A events are almost all LAN, so use that as the proxy.
        for e in raw["events"]:
            e["lan"] = e.get("tier") in ("s", "a")
    out = os.path.join(out_dir, "valve_input.json")
    Path(out).write_text(json.dumps(raw, separators=(",", ":")))
    return out


def run_valve(path, eval_from, eval_to, step_days) -> dict:
    with tempfile.TemporaryDirectory() as d:
        prepared = prepare(path, d)
        proc = subprocess.run(["node", str(JS), prepared, str(eval_from), str(eval_to), str(int(step_days * 86400))],
                              capture_output=True, text=True, cwd=str(JS.parent.parent))
    if proc.returncode != 0:
        sys.exit(proc.stderr[-2000:])
    out = {}
    for line in proc.stdout.splitlines():
        t, a, b, p = line.split()
        out[(int(t), a, b)] = float(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--eval-from", default="2023-03-01")
    ap.add_argument("--eval-to", default="2027-01-01")
    ap.add_argument("--step", type=float, default=7.0, help="days between standings rebuilds (Valve's fit uses 7)")
    ap.add_argument("--rd0", type=float, default=200.0)
    ap.add_argument("--c", type=float, default=20.0)
    args = ap.parse_args()

    matches = load_matches(args.data)
    eval_from, eval_to = ts(args.eval_from), ts(args.eval_to)
    print(f"running Valve model weekly from {args.eval_from} on {args.data} ...", flush=True)
    valve = run_valve(args.data, eval_from, eval_to, args.step)

    evals = [m for m in matches if eval_from <= m.time < eval_to]
    vp = [valve.get((m.time, m.team1_id, m.team2_id), -1.0) for m in evals]
    matched = [i for i, p in enumerate(vp) if p >= 0]
    print(f"eval matches {len(evals)}, Valve has a prediction for {len(matched)} ({100 * len(matched) / len(evals):.1f}%)")

    ours = {}
    for model in (PlayerGlicko(start_rd=args.rd0, c=args.c), OnlineScale(RegionalGlicko(start_rd=args.rd0, c=args.c))):
        preds = walk_forward(model, matches, eval_from)
        ours[model.name] = [x for x in preds if x.time < eval_to]

    def row(name, preds):
        s = summarize(preds)
        print(f"{name:60s} {s['n']:6d} {s['logloss']:8.4f} {s['brier']:7.4f} {s['acc']:6.3f} {s['auc']:6.3f} {s['ece']:6.3f}")

    print(f"\n{'model':60s} {'n':>6s} {'logloss':>8s} {'brier':>7s} {'acc':>6s} {'auc':>6s} {'ece':>6s}")
    print("-- all eval matches (Valve scores 0.5 where it has no standing) --")
    row("valve ranking.js + fit.js expectation", [Prediction(m.time, p if p >= 0 else 0.5, m.t1_won, m) for m, p in zip(evals, vp)])
    for name, preds in ours.items():
        row(name, preds)
    print("-- matches where both rosters are in Valve's standings --")
    vpreds = [Prediction(evals[i].time, vp[i], evals[i].t1_won, evals[i]) for i in matched]
    row("valve ranking.js + fit.js expectation", vpreds)
    tempered, a = online_tempered(vpreds)
    row(f"valve + online logit temperature (a -> {a:.2f})", tempered)
    keep = {id(evals[i]) for i in matched}
    for name, preds in ours.items():
        row(name, [x for x in preds if id(x.match) in keep])
    print("\nValve calibration on matched matches:")
    for lo, hi, n, mp, obs in calibration(vpreds, 10):
        print(f"{lo:4.1f}-{hi:4.1f}  {n:5d} {mp:6.3f} {obs:6.3f}")


if __name__ == "__main__":
    main()
