"""Whole-History Rating (Coulom 2008) on players, with round margins and a regional newcomer prior.

Glicko is a filter: a result only moves ratings forward in time, and each player is updated as if the
others were known. The batch Bradley-Terry fit re-reads the whole history jointly but has no notion of
a rating moving, only of old maps counting less. WHR does both in one likelihood: every player has a
rating on each day they played, consecutive days are linked by a Wiener process (variance w^2 per day),
and the MAP of all of them given every result so far is refit incrementally.

Model: per match, team logit = mean of its players' ratings on that day (+ a region entity on
cross-region matches if `region`), map wins are Bernoulli(sigmoid(z1 - z2)), and with `round_weight`
the rounds of valid maps are round_weight-weighted Bernoulli trials at logit round_scale * (z1 - z2),
as in RegionalGlicko and BatchBT. The maps of one match share the logit, so a match is one likelihood row.

Priors: a player's first point is N(mu0, rd0^2), mu0 = current mean of its region + seed_offset (or
the linked rating, as in LinkedGlicko, when a team switches between a synthetic `team:<id>` entity and
its players). A synthetic entity stands for the mean of five players, so its rd0 and w are divided by sqrt(5).

Fitting (all Newton steps are joint over the free points, so teammates who always play together do not
all take the same step): after each match, the day's points of its ten players; when the day changes,
every point in the last `window_days` of each player who played that day; every `full_every_days`, every
point in the last `full_window_days`. Predictions use each player's latest point. Its variance (Kalman-filtered
precision at that point, plus w^2 per day since) shrinks the logit like Glicko's g(RD).

Defaults were picked on the Liquipedia tune window (Jul 2024 - Jun 2025) with OnlineScale on top; the surface is
flat (w 12-20, rd0 400-500, seed -150 to -300 all within 0.001). Alone it equals the Glicko half; in place of the
batch half it improves the bare blend by 0.0008, a gain the stacker absorbs (compare.py `whr`, `blend-whr-*`, `base-whr-*`).

Usage: .venv/bin/python -m predict.whr --data data/matchdata_liquipedia_sides.json --w 15 --rd0 400
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_banded
from scipy.sparse.linalg import LinearOperator, cg

from .data import Match
from .models import Model, series_prob_pick
from .regions import country_region

DAY = 86400
PTS = 400.0 / math.log(10)   # Elo points per logit


class _Buf:
    """Growable numpy array (append rows, view the filled part)."""

    def __init__(self, dtype, width=None, fill=0):
        self.shape = (1024,) if width is None else (1024, width)
        self.a = np.full(self.shape, fill, dtype)
        self.n, self.fill = 0, fill

    def append(self, x):
        if self.n == len(self.a):
            b = np.full((2 * len(self.a),) + self.a.shape[1:], self.fill, self.a.dtype)
            b[: self.n] = self.a
            self.a = b
        self.a[self.n] = x
        self.n += 1
        return self.n - 1

    @property
    def v(self):
        return self.a[: self.n]


class WHR(Model):
    pick_delta = 0.0   # series conversion, see models.series_prob_pick

    def __init__(self, w: float = 15.0, rd0: float = 400.0, seed_offset: float = -200.0, round_weight: float = 1.0,
                 round_scale: float = 0.25, window_days: float = 30.0, full_every_days: float = 7.0,
                 full_window_days: float = 120.0, iters: int = 2, full_iters: int = 2, shrink: bool = True,
                 region: bool = True, region_rd: float = 100.0, region_w: float = 2.0, syn_scale: float = 5 ** -0.5,
                 max_step: float = 1.0, cg_tol: float = 1e-4):
        # w, rd0, seed_offset, region_rd, region_w in Elo points (w per sqrt(day)); stored in logits
        self.w, self.rd0, self.seed_offset = w / PTS, rd0 / PTS, seed_offset / PTS
        self.region_rd, self.region_w, self.syn_scale = region_rd / PTS, region_w / PTS, syn_scale
        self.round_weight, self.round_scale = round_weight, round_scale
        self.window, self.full_every, self.full_window = window_days, full_every_days, full_window_days
        self.iters, self.full_iters, self.shrink, self.region, self.max_step = iters, full_iters, shrink, region, max_step
        self.cg_tol = cg_tol
        # entities (players, synthetic teams, region pseudo-players)
        self.pid: dict[str, int] = {}
        self.e_mu0 = _Buf(float)          # first-point prior mean
        self.e_sd = _Buf(float)           # prior scale: 1, syn_scale, or region
        self.e_reg = _Buf(np.int32)       # region index for the seed mean, -1 for region entities
        self.e_last = _Buf(np.int64, fill=-1)   # latest slot
        self.regs: dict[str, int] = {}
        self.reg_sum = np.zeros(64)
        self.reg_cnt = np.zeros(64)
        self.last_lineup: dict[str, tuple] = {}
        # slots: one rating per (entity, day); slot 0 is a zero-valued pad
        self.s_r = _Buf(float); self.s_ent = _Buf(np.int64, fill=-1); self.s_day = _Buf(np.int64)
        self.s_prev = _Buf(np.int64, fill=-1); self.s_next = _Buf(np.int64, fill=-1)
        self.s_link = _Buf(float)         # precision of the Wiener link to the previous slot
        self.s_P = _Buf(float)            # filtered precision (data up to and including this day)
        self.s_rows: list[list[int]] = []
        self.loc = np.full(1024, -1, np.int64)
        self._new_slot(-1, 0, 0.0)
        self.today: dict[int, int] = {}   # entity -> today's slot
        # likelihood rows, one per match: slots and coefficients (team1 +, team2 -), maps won / played,
        # team1 rounds / rounds on valid maps
        self.width = 12
        self.r_slot = _Buf(np.int64, self.width); self.r_coef = _Buf(float, self.width)
        self.r_y = _Buf(float, 4); self.r_day = _Buf(np.int64)
        self.day = None
        self.last_full = None
        self.solve_seconds = 0.0
        self.name = (f"whr(w={w:g},rd0={rd0:g},seed={seed_offset:g},rounds={round_weight:g}x{round_scale:g},"
                     f"win={window_days:g}/{full_window_days:g}d/{full_every_days:g}d" + (",region" if region else "") + ")")

    # ---- entities ------------------------------------------------------------------------------
    def _reg(self, name):
        if name not in self.regs:
            self.regs[name] = len(self.regs)
        return self.regs[name]

    def _rating(self, e):
        s = self.e_last.a[e]
        return self.s_r.a[s] if s >= 0 else self.e_mu0.a[e]

    def _add_entity(self, key, mu0, sd, reg):
        e = self.e_mu0.append(mu0)
        self.e_sd.append(sd); self.e_reg.append(reg); self.e_last.append(-1)
        self.pid[key] = e
        if reg >= 0:
            self.reg_sum[reg] += mu0
            self.reg_cnt[reg] += 1
        return e

    def _seed(self, reg):
        return (self.reg_sum[reg] / self.reg_cnt[reg] if self.reg_cnt[reg] >= 5 else 0.0) + self.seed_offset

    def _register(self, tid, players, countries):
        """Newcomers get the region seed, or the linked rating (see models.LinkedGlicko)."""
        syn = f"team:{tid}"
        for p, cc in zip(players, countries):
            if p in self.pid:
                continue
            reg = self._reg(country_region(cc))
            mu0 = self._seed(reg)
            if p == syn and tid in self.last_lineup:
                lineup = self.last_lineup[tid]
                mu0 = sum(self._rating(self.pid[x]) for x in lineup) / len(lineup)
            elif p != syn and syn in self.pid:
                mu0 = self._rating(self.pid[syn])
            self._add_entity(p, mu0, self.syn_scale if p.startswith("team:") else 1.0, reg)
        if players[0] != syn:
            self.last_lineup[tid] = players

    def _region_entity(self, name):
        key = f"region:{name}"
        if key not in self.pid:
            self._add_entity(key, 0.0, -1.0, -1)
        return self.pid[key]

    def _sd(self, e):
        s = self.e_sd.a[e]
        return (self.region_rd, self.region_w) if s < 0 else (s * self.rd0, s * self.w)

    def _var(self, e, day):
        """Posterior variance of the entity's current rating (logits^2)."""
        rd0, w = self._sd(e)
        s = self.e_last.a[e]
        if s < 0:
            return rd0 * rd0
        return 1.0 / self.s_P.a[s] + w * w * (day - self.s_day.a[s])

    # ---- slots and rows ------------------------------------------------------------------------
    def _new_slot(self, e, day, r):
        s = self.s_r.append(r)
        self.s_ent.append(e); self.s_day.append(day); self.s_prev.append(-1); self.s_next.append(-1)
        self.s_link.append(0.0); self.s_P.append(0.0)
        self.s_rows.append([])
        if e >= 0:
            prev = self.e_last.a[e]
            if prev >= 0:
                _, w = self._sd(e)
                self.s_prev.a[s] = prev
                self.s_next.a[prev] = s
                self.s_link.a[s] = 1.0 / (w * w * (day - self.s_day.a[prev]))
            self.e_last.a[e] = s
        if len(self.loc) < self.s_r.n:
            self.loc = np.full(2 * self.s_r.n, -1, np.int64)
        return s

    def _slot_today(self, e, day):
        if e not in self.today:
            self.today[e] = self._new_slot(e, day, self._rating(e))
        return self.today[e]

    def _sides(self, m):
        out = []
        for players in (m.team1_players, m.team2_players):
            out.append([self.pid[p] for p in players])
        r1, r2 = m.team1_region, m.team2_region
        regs = (self._region_entity(r1), self._region_entity(r2)) if self.region and r1 != r2 else None
        return out, regs

    # ---- Newton solve over a set of free slots, everything else fixed --------------------------
    def _solve(self, F, iters, rows=None):
        t0 = time.time()
        F = np.unique(np.asarray(F, np.int64))
        n = len(F)
        if n == 0:
            return
        big = n > 400
        if big:   # each entity's slots adjacent, so its Wiener chain is tridiagonal (the CG preconditioner)
            F = F[np.lexsort((F, self.s_ent.a[F]))]
        loc = self.loc
        loc[F] = np.arange(n)
        if rows is None:
            rows = np.unique(np.fromiter((r for f in F for r in self.s_rows[f]), np.int64))
        S, C, Y = self.r_slot.a[rows], self.r_coef.a[rows], self.r_y.a[rows]
        L = loc[S]
        free = L >= 0
        gi, gc, grow = L[free], C[free], np.nonzero(free)[0]
        pair = free[:, :, None] & free[:, None, :]
        pr, pk, pl = np.nonzero(pair)
        ha, hb, hc = L[pr, pk], L[pr, pl], C[pr, pk] * C[pr, pl]
        # Wiener links and first-point priors
        R = self.s_r.a
        prev, nxt, link = self.s_prev.a[F], self.s_next.a[F], self.s_link.a[F]
        ents = self.s_ent.a[F]
        has_prev = prev >= 0
        lp = np.where(has_prev, loc[np.maximum(prev, 0)], -1)
        nxt_fixed = (nxt >= 0) & (loc[np.maximum(nxt, 0)] < 0)
        nlink = np.where(nxt_fixed, self.s_link.a[np.maximum(nxt, 0)], 0.0)
        first = ~has_prev
        sd0 = np.array([self._sd(e)[0] for e in ents[first]])
        p0 = np.zeros(n); p0[first] = 1.0 / sd0 ** 2
        mu0 = np.zeros(n); mu0[first] = self.e_mu0.a[ents[first]]
        both = has_prev & (lp >= 0)
        bi, bj, bl = np.nonzero(both)[0], lp[both], link[both]
        diag_prior = p0 + np.where(has_prev, link, 0.0) + nlink
        np.add.at(diag_prior, bj, bl)
        if big:
            same = ha == hb
            assert np.all(bj == bi - 1)
        rs, sc = self.round_weight, self.round_scale
        for _ in range(iters):
            x = R[F]
            z = (R[S] * C).sum(1)
            p1 = 1.0 / (1.0 + np.exp(-z))
            gz = Y[:, 0] - Y[:, 1] * p1
            hz = Y[:, 1] * p1 * (1 - p1)
            if rs:
                p2 = 1.0 / (1.0 + np.exp(-sc * z))
                gz = gz + rs * sc * (Y[:, 2] - Y[:, 3] * p2)
                hz = hz + rs * sc * sc * Y[:, 3] * p2 * (1 - p2)
            g = np.bincount(gi, gc * gz[grow], minlength=n)
            g -= p0 * (x - mu0)
            g -= np.where(has_prev, link * (x - R[np.maximum(prev, 0)]), 0.0)
            g -= nlink * (x - R[np.maximum(nxt, 0)])
            np.add.at(g, bj, -bl * (x[bj] - x[bi]))
            hv = hc * hz[pr]
            if n <= 400:
                H = np.zeros((n, n))
                np.add.at(H, (ha, hb), hv)
                H[np.arange(n), np.arange(n)] += diag_prior
                H[bi, bj] -= bl
                H[bj, bi] -= bl
                d = np.linalg.solve(H, g)
            else:
                H = sp.csr_matrix((np.concatenate([hv, diag_prior, -bl, -bl]),
                                   (np.concatenate([ha, np.arange(n), bi, bj]), np.concatenate([hb, np.arange(n), bj, bi]))),
                                  shape=(n, n))
                ab = np.zeros((3, n))
                ab[1] = diag_prior + np.bincount(ha[same], hv[same], minlength=n)
                ab[0, bi] = ab[2, bj] = -bl
                M = LinearOperator((n, n), lambda r: solve_banded((1, 1), ab, r), dtype=float)
                d, _ = cg(H, g, rtol=self.cg_tol, maxiter=200, M=M)
            R[F] = x + np.clip(d, -self.max_step, self.max_step)
        # filtered precision at each entity's latest slot, for the prediction variance
        last = F[self.e_last.a[ents] == F]
        if len(last):
            z = (R[S] * C).sum(1)
            p1 = 1.0 / (1.0 + np.exp(-z))
            hz = Y[:, 1] * p1 * (1 - p1)
            if rs:
                p2 = 1.0 / (1.0 + np.exp(-sc * z))
                hz = hz + rs * sc * sc * Y[:, 3] * p2 * (1 - p2)
            h = np.bincount(gi, gc * gc * hz[grow], minlength=n)[loc[last]]
            pv = self.s_prev.a[last]
            prior = np.where(pv >= 0, 1.0 / (1.0 / np.maximum(self.s_P.a[np.maximum(pv, 0)], 1e-12)
                                             + 1.0 / np.maximum(self.s_link.a[last], 1e-12)), p0[loc[last]])
            self.s_P.a[last] = h + prior
        loc[F] = -1
        self._refresh_regions()
        self.solve_seconds += time.time() - t0

    def _refresh_regions(self):
        reg = self.e_reg.v
        keep = reg >= 0
        last = self.e_last.v
        r = np.where(last >= 0, self.s_r.a[np.maximum(last, 0)], self.e_mu0.v)
        self.reg_sum[:] = np.bincount(reg[keep], r[keep], minlength=len(self.reg_sum))[: len(self.reg_sum)]

    # ---- schedule --------------------------------------------------------------------------------
    def _advance(self, day):
        if self.day is None:
            self.day = self.last_full = day
        if day <= self.day:
            return
        if self.today and self.window > 0:
            lo = self.day - self.window
            F = []
            for s in self.today.values():
                while s >= 0 and self.s_day.a[s] >= lo:
                    F.append(s)
                    s = self.s_prev.a[s]
            self._solve(F, self.iters)
        self.today = {}
        self.day = day
        if self.full_every and day - self.last_full >= self.full_every:
            self.last_full = day
            lo = day - self.full_window
            first = int(np.searchsorted(self.s_day.v[1:], lo)) + 1
            rows = np.arange(int(np.searchsorted(self.r_day.v, lo)), self.r_day.n)
            if first < self.s_r.n:
                self._solve(np.arange(first, self.s_r.n), self.full_iters, rows)

    # ---- Model interface -------------------------------------------------------------------------
    def _logit_var(self, m):
        day = m.time // DAY
        self._advance(day)
        self._register(m.team1_id, m.team1_players, m.team1_countries)
        self._register(m.team2_id, m.team2_players, m.team2_countries)
        (e1, e2), regs = self._sides(m)
        z = sum(self._rating(e) for e in e1) / len(e1) - sum(self._rating(e) for e in e2) / len(e2)
        v = sum(self._var(e, day) for e in e1) / len(e1) ** 2 + sum(self._var(e, day) for e in e2) / len(e2) ** 2
        if regs:
            z += self._rating(regs[0]) - self._rating(regs[1])
        return z, v

    def predict(self, m: Match) -> float:
        z, v = self._logit_var(m)
        if self.shrink:
            z /= math.sqrt(1.0 + 3.0 * v / math.pi ** 2)
        return series_prob_pick(1.0 / (1.0 + math.exp(-z)), m.best_of, self.pick_delta)

    def team(self, players):
        """(Elo-scale rating, RD in points) of a lineup, for Stacked's features."""
        day = self.day or 0
        es = [self.pid[p] for p in players if p in self.pid]
        if not es:
            return 1500.0, self.rd0 * PTS / math.sqrt(len(players))
        r = sum(self._rating(e) for e in es) / len(es)
        v = (sum(self._var(e, day) for e in es) + (len(players) - len(es)) * self.rd0 ** 2) / len(players) ** 2
        return 1500.0 + PTS * r, PTS * math.sqrt(v)

    def update(self, m: Match) -> None:
        day = m.time // DAY
        self._advance(day)
        self._register(m.team1_id, m.team1_players, m.team1_countries)
        self._register(m.team2_id, m.team2_players, m.team2_countries)
        (e1, e2), regs = self._sides(m)
        slots = np.zeros(self.width, np.int64)
        coef = np.zeros(self.width)
        k = 0
        for es, sign in ((e1, 1.0), (e2, -1.0)):
            for e in es:
                slots[k], coef[k] = self._slot_today(e, day), sign / len(es)
                k += 1
        if regs:
            for e, sign in zip(regs, (1.0, -1.0)):
                slots[k], coef[k] = self._slot_today(e, day), sign
                k += 1
        wins = sum(mp.t1_won for mp in m.maps)
        valid = [mp for mp in m.maps if mp.valid_for_margin]
        ra, rn = sum(mp.t1 for mp in valid), sum(mp.t1 + mp.t2 for mp in valid)
        row = self.r_slot.append(slots)
        self.r_coef.append(coef); self.r_y.append((wins, len(m.maps), ra, rn)); self.r_day.append(day)
        for s in slots[:k]:
            self.s_rows[s].append(row)
        self._solve(slots[:k], self.iters)


def main():
    import datetime as dt

    from .data import load_matches
    from .evaluate import summarize, walk_forward
    from .models import OnlineScale

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/matchdata_liquipedia_sides.json")
    ap.add_argument("--w", type=float, default=15.0)
    ap.add_argument("--rd0", type=float, default=400.0)
    ap.add_argument("--seed", type=float, default=-200.0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N matches (timing)")
    args = ap.parse_args()
    matches = load_matches(args.data)
    if args.limit:
        matches = matches[: args.limit]
    whr = WHR(w=args.w, rd0=args.rd0, seed_offset=args.seed)
    t0 = time.time()
    preds = walk_forward(OnlineScale(whr), matches, int(dt.datetime(2024, 7, 1, tzinfo=dt.timezone.utc).timestamp()))
    s = summarize(preds)
    print(f"{whr.name}: {time.time() - t0:.0f}s (solves {whr.solve_seconds:.0f}s), logloss {s['logloss']:.4f} acc {s['acc']:.3f}")


if __name__ == "__main__":
    main()
