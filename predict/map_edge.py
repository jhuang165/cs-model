"""Team-level map edges, priced once the maps are known (post-veto). See README "Map edges on Liquipedia".

Runs the shipped Glicko half walk-forward and, on top of its map-agnostic per-map logit z, learns online
(SGD logistic regression, updated after each map) a map-level calibration from
  z                        the Glicko map logit
  edge_diff                team1's minus team2's learned (team id, map) residual, an Elo-style offset per
                           (team, map) updated with step k and shrunk toward zero
  share_diff               how much more of its recent maps team1 has played on this map than team2 has
                           (map pool preference: teams pick what they are good at)
The reference arm uses z alone, so the comparison is paired and both arms get the same temperature.

Usage: .venv/bin/python -m predict.map_edge --data data/matchdata_liquipedia_sides.json
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict, deque

import numpy as np

from .compare import CONFIRM, TUNE, ts
from .data import load_matches
from .models import UNKNOWN_MAPS
from .rankings import best_model, synthetic_rosters


class MapEdges:
    """Per (team id, map) residual offset, in map-logit units."""

    def __init__(self, k=0.05, shrink=0.02, recent=40):
        self.k, self.shrink = k, shrink
        self.o = defaultdict(float)
        self.hist = defaultdict(lambda: deque(maxlen=recent))   # team -> its last maps' names

    def diff(self, t1, t2, mp):
        return self.o[(t1, mp)] - self.o[(t2, mp)]

    def share(self, team, mp):
        h = self.hist[team]
        return (sum(x == mp for x in h) + 1 / 7) / (len(h) + 1)   # ~7 maps in the pool

    def share_diff(self, t1, t2, mp):
        return math.log(self.share(t1, mp)) - math.log(self.share(t2, mp))

    def update(self, t1, t2, mp, resid):
        for team, sign in ((t1, 1.0), (t2, -1.0)):
            key = (team, mp)
            self.o[key] = self.o[key] * (1 - self.shrink) + sign * self.k * resid
            self.hist[team].append(mp)


class OnlineLR:
    def __init__(self, n, lr=0.002):
        self.w = np.zeros(n); self.w[0] = 1.0; self.b = 0.0; self.lr = lr

    def p(self, x):
        return 1 / (1 + math.exp(-(self.w @ x + self.b)))

    def update(self, x, y):
        g = y - self.p(x)
        self.w += self.lr * g * x
        self.b += self.lr * g


def run(matches, syn, k, shrink):
    _, glicko, _ = best_model(syn, stacked=False)
    edges = MapEdges(k=k, shrink=shrink)
    arms = {"z": OnlineLR(1), "z+edge": OnlineLR(2), "z+share": OnlineLR(2), "z+edge+share": OnlineLR(3)}
    out = {a: [] for a in arms}
    ys, tt = [], []
    for m in matches:
        pm = glicko.predict_map(m, "")
        z = math.log(pm / (1 - pm))
        for mp in m.maps:
            if mp.name in UNKNOWN_MAPS or mp.name == "unknown":
                continue
            e = edges.diff(m.team1_id, m.team2_id, mp.name)
            s = edges.share_diff(m.team1_id, m.team2_id, mp.name)
            feats = {"z": [z], "z+edge": [z, e], "z+share": [z, s], "z+edge+share": [z, e, s]}
            y = 1.0 if mp.t1_won else 0.0
            for a, lr in arms.items():
                x = np.array(feats[a])
                out[a].append(lr.p(x))
                lr.update(x, y)
            ys.append(y); tt.append(m.time)
            edges.update(m.team1_id, m.team2_id, mp.name, y - 1 / (1 + math.exp(-(z + e))))
        glicko.update(m)
    return np.array(tt), np.array(ys), {a: np.array(v) for a, v in out.items()}, arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/matchdata_liquipedia_sides.json")
    ap.add_argument("--k", default="0.05", help="comma list of offset steps (map-logit units)")
    ap.add_argument("--shrink", default="0.02")
    args = ap.parse_args()
    matches = load_matches(args.data)
    syn = synthetic_rosters(matches)
    for k in map(float, args.k.split(",")):
        for sh in map(float, args.shrink.split(",")):
            t, y, out, arms = run(matches, syn, k, sh)
            print(f"k={k:g} shrink={sh:g}  maps scored: tune {((t >= ts(TUNE)) & (t < ts(CONFIRM))).sum()}, confirm {(t >= ts(CONFIRM)).sum()}")
            ref = None
            for a, p in out.items():
                p = np.clip(p, 1e-6, 1 - 1e-6)
                l = -(y * np.log(p) + (1 - y) * np.log(1 - p))
                ref = l if ref is None else ref
                row = f"  {a:14s}"
                for lo, hi in ((ts(TUNE), ts(CONFIRM)), (ts(CONFIRM), 1e12)):
                    w = (t >= lo) & (t < hi)
                    d = l[w] - ref[w]
                    row += f"  {l[w].mean():.4f} ({d.mean():+.4f} ± {d.std() / math.sqrt(w.sum()):.4f})"
                print(row, " w =", np.round(arms[a].w, 3))


if __name__ == "__main__":
    main()
