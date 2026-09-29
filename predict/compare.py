"""Paired walk-forward comparison of model variants on one data file.

Each variant is a name in VARIANTS mapping to a factory. Predictions are cached per (file, variant) in
data/compare/, so adding a variant only runs that one. Reports log loss on the tune window (Jul 2024 - Jun 2025)
and the confirm window (from Jul 2025), each variant minus the first one listed, with paired standard errors.

Usage: .venv/bin/python -m predict.compare --data data/matchdata_liquipedia_sides.json base sides
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import time
from pathlib import Path

import numpy as np

from .data import load_matches
from .evaluate import walk_forward
from .liquipedia import ROOT
from .rankings import best_model, synthetic_rosters

CACHE = ROOT / "data" / "compare"
TUNE, CONFIRM = "2024-07-01", "2025-07-01"


def ts(d):
    return int(dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc).timestamp())


def _sides(syn):
    from .models import SideRates
    return best_model(syn, glicko_kw={"side_rates": SideRates()}, batch_kw={"sides": True})[0]


def _pick(delta, stacked=True):
    def make(syn):
        model, glicko, batch = best_model(syn, stacked=stacked)
        glicko.pick_delta = batch.pick_delta = delta
        return model
    return make


def _whr(kind):
    """Whole-History Rating (whr.py) alone, or in place of either blend half, bare or under the stacker."""
    def make(syn):
        from .models import Blend, OnlineScale
        from .rankings import BATCH_WEIGHT
        from .stacked import Stacked
        from .whr import WHR
        _, glicko, batch = best_model(syn, stacked=False)
        whr = WHR()
        if kind == "glicko-half":
            return OnlineScale(glicko)
        if kind == "whr":
            return OnlineScale(whr)
        if kind in ("for-batch", "stacked-for-batch"):
            blend = OnlineScale(Blend(whr, glicko, w=BATCH_WEIGHT))
            return Stacked(blend, glicko, whr) if kind == "stacked-for-batch" else blend
        blend = OnlineScale(Blend(batch, whr, w=BATCH_WEIGHT))
        return Stacked(blend, whr, batch) if kind == "stacked-for-glicko" else blend
    return make


def _region_stack(syn):
    from .stacked import Stacked
    model, glicko, batch = best_model(syn)
    return Stacked(model.inner, glicko, batch, region_feats=True)


VARIANTS = {
    "base": lambda syn: best_model(syn)[0],
    "blend": lambda syn: best_model(syn, stacked=False)[0],
    "sides": _sides,
    "region-stack": _region_stack,
    "offset-lr6": lambda syn: best_model(syn, glicko_kw={"offset_lr": 6.0})[0],
    **{f"pick{d:g}": _pick(d) for d in (0.2, 0.4, 0.6, 0.8)},
    **{f"blend-pick{d:g}": _pick(d, stacked=False) for d in (0.2, 0.4, 0.6, 0.8)},
    "glicko-half": _whr("glicko-half"),
    "whr": _whr("whr"),
    "blend-whr-glicko": _whr("for-glicko"),
    "blend-whr-batch": _whr("for-batch"),
    "base-whr-glicko": _whr("stacked-for-glicko"),
    "base-whr-batch": _whr("stacked-for-batch"),
}


def run(name, matches, syn, path: Path):
    f = CACHE / f"{path.stem}.{name}.npz"
    if f.exists() and f.stat().st_mtime > path.stat().st_mtime:
        d = np.load(f)
        return d["t"], d["p"], d["y"]
    t0 = time.time()
    preds = walk_forward(VARIANTS[name](syn), matches, ts(TUNE))
    t = np.array([x.time for x in preds]); p = np.array([x.p for x in preds]); y = np.array([x.won for x in preds], float)
    CACHE.mkdir(parents=True, exist_ok=True)
    # names and best-of let other tools (market.py) join these predictions to outside prices
    np.savez(f, t=t, p=p, y=y, team1=np.array([x.match.team1_name for x in preds]),
             team2=np.array([x.match.team2_name for x in preds]), event=np.array([x.match.event_id for x in preds]),
             best_of=np.array([x.match.best_of for x in preds]))
    print(f"  ran {name} in {time.time() - t0:.0f}s", flush=True)
    return t, p, y


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "matchdata_liquipedia_sides.json"))
    ap.add_argument("variants", nargs="+", help=f"first is the reference; known: {', '.join(VARIANTS)}")
    args = ap.parse_args()
    path = Path(args.data)
    matches = load_matches(path)
    syn = synthetic_rosters(matches)
    res = {v: run(v, matches, syn, path) for v in args.variants}
    t, p0, y = res[args.variants[0]]
    windows = [("tune", (t >= ts(TUNE)) & (t < ts(CONFIRM))), ("confirm", t >= ts(CONFIRM))]
    print(f"{path.name}: tune n={windows[0][1].sum()}, confirm n={windows[1][1].sum()}")
    print(f"{'variant':24s} {'tune':>8s} {'confirm':>8s}   {'d tune':>16s} {'d confirm':>16s}")
    for v in args.variants:
        _, p, _ = res[v]
        row = f"{v:24s}"
        for _, w in windows:
            row += f" {ll(p[w], y[w]).mean():8.4f}"
        row += "  "
        for _, w in windows:
            d = ll(p[w], y[w]) - ll(p0[w], y[w])
            row += f" {d.mean():+8.4f} ± {d.std() / math.sqrt(len(d)):.4f}"
        print(row)


if __name__ == "__main__":
    main()
