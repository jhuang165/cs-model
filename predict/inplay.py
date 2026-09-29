"""In-play pricing: P(team1 wins the map | score, sides) and P(team1 wins the series | maps so far).

Round model: every round is a Bernoulli trial for team1 with logit rho + b when team1 is CT and rho - b when it is
T, where b is the map's CT-side round logit (models.SideRates) and rho is solved so that the price from 0-0
equals the pre-map probability. That makes the in-play price consistent with the pre-match model by
construction. A dynamic program over the score then gives the map price at any state, with CS2's MR12 (first to
13; 12-12 goes to MR3 overtime, first to 4 of 6 with sides swapped after 3, repeated at 3-3).

Rounds are not independent (economy, momentum), so the raw DP can be over- or under-confident about a
lead. `validate` checks it against the ~100k halftime states in the Liquipedia data (the half split is the only
in-map state the data has) and fits a walk-forward correction, logit P = a*logit(DP) + c*z_pre.

Usage:
  .venv/bin/python -m predict.inplay --validate                         # halftime backtest
  .venv/bin/python -m predict.inplay --p 0.60 --score 7-5 --side ct --map Nuke
"""
from __future__ import annotations

import argparse
import math
from functools import lru_cache

import numpy as np

from .models import SideRates

MR = 12          # rounds per half in CS2
WIN = MR + 1     # rounds to win in regulation
OT_HALF = 3
# walk-forward correction fitted at the ~107k halftime states (see validate): logit P = DP_TEMP * logit(DP) +
# PRE_WEIGHT * logit(pre-map). Rounds are not iid (economy, the half resets it), so the raw DP overprices leads:
# +4 at the half is won 83%, the DP says 88%. Fitted at the half only; other states are an extrapolation.
DP_TEMP, PRE_WEIGHT = 0.71, 0.08


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _ot_win(q):
    """P(win an MR3 overtime period, repeated at 3-3) with round probability q (sides swap inside it, so the
    CT bias roughly cancels and is ignored). Vectorised over q."""
    def value(win, tie):
        v = {}
        for a in range(OT_HALF + 1, -1, -1):
            for b in range(OT_HALF + 1, -1, -1):
                if a == OT_HALF + 1:
                    v[a, b] = win
                elif b == OT_HALF + 1:
                    v[a, b] = 0.0
                elif a == OT_HALF and b == OT_HALF:
                    v[a, b] = tie
                else:
                    v[a, b] = q * v[a + 1, b] + (1 - q) * v[a, b + 1]
        return v[0, 0]

    # P = P(win without reaching 3-3) + P(reach 3-3) * P
    return value(1.0, 0.0) / (1 - value(0.0, 1.0))


def _grid(rho, bias, first_ct):
    """Win probability at every regulation score, as {(x, y): array}; all inputs broadcast."""
    q_ct, q_t = _sig(rho + bias), _sig(rho - bias)
    q1, q2 = np.where(first_ct, q_ct, q_t), np.where(first_ct, q_t, q_ct)   # team1's round p by half
    ot = _ot_win(_sig(rho))
    v = {}
    for x in range(WIN, -1, -1):
        for y in range(WIN, -1, -1):
            if x == WIN and y == WIN:
                continue
            if x == WIN:
                v[x, y] = 1.0
            elif y == WIN:
                v[x, y] = 0.0
            elif x == MR and y == MR:
                v[x, y] = ot
            else:
                q = q1 if x + y < MR else q2
                v[x, y] = q * v[x + 1, y] + (1 - q) * v[x, y + 1]
    return v


def map_win(rho, bias, first_ct, a=0, b=0):
    """P(team1 wins the map) from score a-b in regulation, team1 starting on CT if first_ct. Vectorised."""
    rho, bias, first_ct, a, b = np.broadcast_arrays(*(np.asarray(x, float) for x in (rho, bias, first_ct, a, b)))
    v = _grid(rho, bias, first_ct.astype(bool))
    out = np.empty(rho.shape)
    for (x, y), val in v.items():
        sel = (a == x) & (b == y)
        if sel.any():
            out[sel] = np.broadcast_to(val, rho.shape)[sel]
    return out if out.ndim else float(out)


def solve_rho(p_map, bias, first_ct):
    """The round logit whose 0-0 price equals the pre-map probability (bisection, vectorised)."""
    p_map, bias, first_ct = np.broadcast_arrays(*(np.asarray(x, float) for x in (p_map, bias, first_ct)))
    lo, hi = np.full(p_map.shape, -4.0), np.full(p_map.shape, 4.0)
    for _ in range(40):
        mid = (lo + hi) / 2
        below = _grid(mid, bias, first_ct.astype(bool))[0, 0] < p_map
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return (lo + hi) / 2


def price_map(p_map, a, b, bias=0.0, first_ct=None):
    """In-play map price. first_ct=None (side unknown) averages the two starting sides. Vectorised."""
    if first_ct is None:
        return 0.5 * (price_map(p_map, a, b, bias, True) + price_map(p_map, a, b, bias, False))
    return map_win(solve_rho(p_map, bias, first_ct), bias, first_ct, a, b)


def price_map_calibrated(p_map, a, b, bias=0.0, first_ct=None):
    """price_map with the halftime correction. Before the half it is an extrapolation and shrinks a little
    toward even (0.60 -> 0.58 at 0-0); prefer the pre-map price there."""
    p = np.clip(price_map(p_map, a, b, bias, first_ct), 1e-9, 1 - 1e-9)
    pre = np.clip(np.asarray(p_map, float), 1e-9, 1 - 1e-9)
    return _sig(DP_TEMP * np.log(p / (1 - p)) + PRE_WEIGHT * np.log(pre / (1 - pre)))


def price_series(p_maps: list[float], best_of: int, won: int, lost: int) -> float:
    """P(team1 wins the series) after it has won `won` and lost `lost` maps; p_maps prices the remaining maps
    in order (pad with the map-agnostic price for maps whose veto is unknown)."""
    need = (best_of + 1) // 2
    if won >= need:
        return 1.0
    if lost >= need:
        return 0.0
    rest = p_maps[: best_of - won - lost]

    @lru_cache(maxsize=None)
    def f(i, w, l):
        if w == need:
            return 1.0
        if l == need:
            return 0.0
        p = rest[i] if i < len(rest) else rest[-1]
        return p * f(i + 1, w + 1, l) + (1 - p) * f(i + 1, w, l + 1)

    return f(0, won, lost)


def validate(path: str):
    """Walk-forward halftime backtest: pre-map price, raw DP at the half, DP + online correction."""
    from .compare import CONFIRM, TUNE, ts
    from .data import load_matches
    from .models import OnlineScale
    from .rankings import best_model, synthetic_rosters

    matches = load_matches(path)
    _, glicko, _ = best_model(synthetic_rosters(matches), stacked=False)
    sides = SideRates()
    a_temp = 1.0               # online per-map temperature on the Glicko map logit
    w = np.array([1.0, 0.0])   # online correction on [logit(DP), z_pre]
    pre, meta = [], []
    for m in matches:
        pm = glicko.predict_map(m, "")
        z = a_temp * math.log(pm / (1 - pm))
        for mp in m.maps:
            s = mp.sides
            y = 1.0 if mp.t1_won else 0.0
            if s and max(s["t1ct"] + s["t1t"], s["t2ct"] + s["t2t"]) <= WIN and mp.t1 + mp.t2 >= WIN:
                first_ct = s["first"] == "ct"
                h1 = s["t1ct"] if first_ct else s["t1t"]
                h2 = s["t2t"] if first_ct else s["t2ct"]
                if h1 + h2 == MR:
                    pre.append((1 / (1 + math.exp(-z)), h1, h2, sides.bias(mp.name), first_ct))
                    meta.append((m.time, y, z))
            a_temp += 0.002 * (y - 1 / (1 + math.exp(-z))) * math.log(pm / (1 - pm))
        glicko.update(m)
        sides.update(m)
    pre, meta = np.array(pre), np.array(meta)
    p_dp = np.clip(price_map(pre[:, 0], pre[:, 1], pre[:, 2], pre[:, 3], pre[:, 4]), 1e-6, 1 - 1e-6)
    p_cor = np.empty(len(p_dp))
    for i, (pd, (t_, y, z)) in enumerate(zip(p_dp, meta)):
        x = np.array([math.log(pd / (1 - pd)), z])
        p_cor[i] = 1 / (1 + math.exp(-float(w @ x)))
        w += 0.002 * (y - p_cor[i]) * x
    r = np.column_stack([meta[:, 0], meta[:, 1], pre[:, 0], p_dp, p_cor, pre[:, 1] - pre[:, 2]])
    t, y = r[:, 0], r[:, 1]

    def ll(p, yy):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))
    print(f"halftime states: {len(r)}; correction weights [logit DP, z_pre] = {np.round(w, 3)}")
    print(f"{'price at the half':26s} {'tune':>7s} {'confirm':>8s}")
    for name, col in (("pre-map (ignores score)", 2), ("round DP", 3), ("round DP + correction", 4)):
        row = f"{name:26s}"
        for lo, hi in ((ts(TUNE), ts(CONFIRM)), (ts(CONFIRM), 1e12)):
            sel = (t >= lo) & (t < hi)
            row += f" {ll(r[sel, col], y[sel]).mean():7.4f}"
        print(row)
    sel = t >= ts(TUNE)
    print("\nby halftime margin (team1 minus team2), from tune start: n, observed, DP, corrected")
    for mg in (-8, -6, -4, -2, 0, 2, 4, 6, 8):
        s2 = sel & (r[:, 5] == mg)
        if s2.sum() > 50:
            print(f"  {mg:+3d} {s2.sum():6d}  {y[s2].mean():.3f}  {r[s2, 3].mean():.3f}  {r[s2, 4].mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--data", default="data/matchdata_liquipedia_sides.json")
    ap.add_argument("--p", type=float, help="pre-map P(team1 wins the map)")
    ap.add_argument("--score", default="0-0", help="team1-team2 rounds")
    ap.add_argument("--side", choices=["ct", "t"], help="team1's starting side")
    ap.add_argument("--map", help="map name, for its CT bias (from the data file)")
    args = ap.parse_args()
    if args.validate:
        validate(args.data)
        return
    if args.p is None:
        ap.error("--p is required unless --validate")
    bias = 0.0
    if args.map:
        from .data import load_matches
        sides = SideRates()
        for m in load_matches(args.data):
            sides.update(m)
        bias = sides.bias(args.map)
    a, b = (int(x) for x in args.score.split("-"))
    first_ct = None if args.side is None else args.side == "ct"
    raw, cal = price_map(args.p, a, b, bias, first_ct), price_map_calibrated(args.p, a, b, bias, first_ct)
    print(f"CT round bias {bias:+.3f}  P(team1 wins the map at {a}-{b}) = {cal:.3f}  (iid-round DP {raw:.3f})")


if __name__ == "__main__":
    main()
