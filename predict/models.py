"""Online rating models. Each exposes predict(match) -> P(team1 wins series) and update(match).

All models are evaluated walk-forward: predict() is always called before update() for a match,
so no model ever sees the outcome it is predicting.
"""
from __future__ import annotations

import math
from collections import defaultdict

from .data import Match

Q = math.log(10) / 400  # Elo scale: 400 points = 10:1 odds


def elo_expected(diff: float) -> float:
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def series_prob(p_map: float, best_of: int) -> float:
    """P(win a best-of-N) given a constant per-map win probability."""
    p = p_map
    if best_of == 1:
        return p
    if best_of == 3:
        return p * p * (3 - 2 * p)
    if best_of == 5:
        return p ** 3 * (10 - 15 * p + 6 * p * p)
    # general: need ceil(N/2) wins before the opponent does
    need = (best_of + 1) // 2
    total = 0.0
    for losses in range(need):
        total += math.comb(need - 1 + losses, losses) * p ** need * (1 - p) ** losses
    return total


def series_prob_seq(probs: list[float], best_of: int) -> float:
    """P(win a best-of-N) when map i has win probability probs[i] (list must have N entries)."""
    need = (best_of + 1) // 2
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def f(i, w, l):
        if w == need:
            return 1.0
        if l == need:
            return 0.0
        p = probs[i]
        return p * f(i + 1, w + 1, l) + (1 - p) * f(i + 1, w, l + 1)

    return f(0, 0, 0)


UNKNOWN_MAPS = {"de_default", ""}


class Model:
    name = "base"

    def predict(self, m: Match) -> float:
        raise NotImplementedError

    def update(self, m: Match) -> None:
        raise NotImplementedError


class Constant(Model):
    """Always 0.5. Sanity floor."""

    name = "constant-0.5"

    def predict(self, m):
        return 0.5

    def update(self, m):
        pass


class Team1Bias(Model):
    """Running team1 win rate. Captures the ordering bias in the data with no skill info."""

    name = "team1-rate"

    def __init__(self):
        self.w = 0
        self.n = 0

    def predict(self, m):
        return (self.w + 1) / (self.n + 2)

    def update(self, m):
        self.n += 1
        self.w += m.t1_won


class TeamElo(Model):
    """Classic Elo keyed by the source teamId, one binary update per series."""

    def __init__(self, k: float = 30.0, start: float = 1500.0):
        self.k = k
        self.r = defaultdict(lambda: start)
        self.name = f"team-elo(k={k:g})"

    def predict(self, m):
        return elo_expected(self.r[m.team1_id] - self.r[m.team2_id])

    def update(self, m):
        p = self.predict(m)
        d = self.k * ((1.0 if m.t1_won else 0.0) - p)
        self.r[m.team1_id] += d
        self.r[m.team2_id] -= d


class PlayerElo(Model):
    """Elo where a team's rating is the mean of its five players' ratings.

    Every player on the team receives the same rating delta. Roster changes are handled
    implicitly: a stand-in brings his own rating, a new lineup inherits its players' history.
    """

    def __init__(self, k: float = 30.0, start: float = 1500.0, per_map: bool = False,
                 round_share: bool = False, margin_weight: float = 1.0):
        self.k = k
        self.start = start
        self.per_map = per_map
        self.round_share = round_share
        self.margin_weight = margin_weight
        self.r = defaultdict(lambda: start)
        tag = "map" if per_map else "series"
        if round_share:
            tag += f"+rounds(w={margin_weight:g})"
        self.name = f"player-elo[{tag}](k={k:g})"

    def team_rating(self, players):
        return sum(self.r[p] for p in players) / len(players)

    def diff(self, m):
        return self.team_rating(m.team1_players) - self.team_rating(m.team2_players)

    def predict(self, m):
        if self.per_map:
            return series_prob(elo_expected(self.diff(m)), m.best_of)
        return elo_expected(self.diff(m))

    def _apply(self, m, delta):
        for p in m.team1_players:
            self.r[p] += delta
        for p in m.team2_players:
            self.r[p] -= delta

    def update(self, m):
        if not self.per_map:
            p = elo_expected(self.diff(m))
            self._apply(m, self.k * ((1.0 if m.t1_won else 0.0) - p))
            return
        for mp in m.maps:
            p = elo_expected(self.diff(m))
            outcome = 1.0 if mp.t1_won else 0.0
            if self.round_share and mp.valid_for_margin:
                # Blend the binary result with the round share: a 13-11 win moves ratings
                # less than a 13-2 win.  margin_weight=1 uses pure round share.
                # Round share is compressed toward 0.5 relative to win prob; stretch it first,
                # then blend with the binary result.
                share = min(1.0, max(0.0, 0.5 + (mp.t1_round_share - 0.5) * 2.0))
                outcome = (1 - self.margin_weight) * outcome + self.margin_weight * share
            self._apply(m, self.k * (outcome - p))


class PlayerGlicko(Model):
    """Glicko-1 on players, team rating = mean of players, team RD = RMS of player RDs / sqrt(5).

    Per-map updates. Rating deviation lets brand-new players move quickly and established
    players move slowly, which matters in a dataset with ~950 teams and heavy churn at the bottom.
    """

    def __init__(self, start: float = 1500.0, start_rd: float = 350.0, min_rd: float = 30.0,
                 c: float = 34.6, per_map: bool = True):
        self.start, self.start_rd, self.min_rd, self.c = start, start_rd, min_rd, c
        self.per_map = per_map
        self.r = defaultdict(lambda: start)
        self.rd = defaultdict(lambda: start_rd)
        self.last_seen = {}
        self.name = f"player-glicko[{'map' if per_map else 'series'}](rd0={start_rd:g},c={c:g},minrd={min_rd:g})"

    @staticmethod
    def g(rd):
        return 1.0 / math.sqrt(1.0 + 3.0 * Q * Q * rd * rd / (math.pi ** 2))

    def _decay(self, players, t):
        # Increase RD for time away (in units of ~weeks). Mirrors Valve's glicko.js constant C.
        for p in players:
            if p in self.last_seen:
                weeks = (t - self.last_seen[p]) / (7 * 24 * 3600)
                self.rd[p] = min(self.start_rd, math.sqrt(self.rd[p] ** 2 + self.c ** 2 * weeks))
            self.last_seen[p] = t

    def team(self, players):
        r = sum(self.r[p] for p in players) / len(players)
        rd = math.sqrt(sum(self.rd[p] ** 2 for p in players) / len(players)) / math.sqrt(len(players))
        return r, rd

    def _p_map(self, m):
        r1, rd1 = self.team(m.team1_players)
        r2, rd2 = self.team(m.team2_players)
        # Symmetric expected score using the combined uncertainty of both teams
        return elo_expected(self.g(math.sqrt(rd1 ** 2 + rd2 ** 2)) * (r1 - r2))

    def predict(self, m):
        p = self._p_map(m)
        return series_prob(p, m.best_of) if self.per_map else p

    def predict_map(self, m, map_name):
        """Per-map win probability; the base model ignores the map name."""
        return self._p_map(m)

    def _update_side(self, players, opp_r, opp_rd, score, off=0.0, scale=1.0, n=1.0):
        """Glicko-1 update. scale/n turn it into n Bernoulli observations whose logit is `scale`
        times the map logit (used for rounds); score is then the observed share."""
        gv = self.g(opp_rd)
        for p in players:
            e = elo_expected(scale * (gv * (self.r[p] - opp_r) + off))
            d2 = 1.0 / (n * Q * Q * scale * scale * gv * gv * e * (1 - e))
            rd2 = self.rd[p] ** 2
            new_rd2 = 1.0 / (1.0 / rd2 + 1.0 / d2)
            self.r[p] += Q * new_rd2 * gv * scale * n * (score - e)
            self.rd[p] = max(self.min_rd, math.sqrt(new_rd2))

    def update(self, m):
        self._decay(m.team1_players, m.time)
        self._decay(m.team2_players, m.time)
        outcomes = [1.0 if mp.t1_won else 0.0 for mp in m.maps] if self.per_map else [1.0 if m.t1_won else 0.0]
        for s in outcomes:
            r1, rd1 = self.team(m.team1_players)
            r2, rd2 = self.team(m.team2_players)
            self._update_side(m.team1_players, r2, rd2, s)
            self._update_side(m.team2_players, r1, rd1, 1.0 - s)


class OnlineBias(Model):
    """Wraps a model and learns a team1 logit offset online (SGD on log loss).

    The source data lists teams in a non-random order (team1 wins 56%), so an intercept
    is a legitimate, leak-free feature as long as it is learned only from past matches.
    """

    def __init__(self, inner: Model, lr: float = 0.02):
        self.inner, self.lr, self.b = inner, lr, 0.0
        self.name = f"{inner.name}+bias(lr={lr:g})"

    def predict(self, m):
        p = min(1 - 1e-6, max(1e-6, self.inner.predict(m)))
        z = math.log(p / (1 - p)) + self.b
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, m):
        p = self.predict(m)
        self.b += self.lr * ((1.0 if m.t1_won else 0.0) - p)
        self.inner.update(m)


class PlayerMapGlicko(PlayerGlicko):
    """PlayerGlicko plus a per-(player, map) Elo-style offset.

    Team strength on map M = mean player rating + mean player offset on M. The global rating
    is updated by Glicko as before (with the map offset included in the expectation, so map
    effects are not double counted); the offsets are updated with a fixed step k_map and shrink
    toward zero on every update so rarely-played maps do not accumulate noise.

    known_maps=True uses the maps actually played for the series probability (the veto is
    known before a series starts in practice); any decider that was not played is priced at the
    map-agnostic probability. known_maps=False predicts the series from global ratings only,
    so per-map offsets can only help by cleaning the global rating.
    """

    def __init__(self, k_map: float = 10.0, shrink: float = 0.01, known_maps: bool = True,
                 start_rd: float = 200.0, c: float = 20.0, **kw):
        super().__init__(start_rd=start_rd, c=c, **kw)
        self.k_map, self.shrink, self.known_maps = k_map, shrink, known_maps
        self.o = defaultdict(float)
        self.name = f"player-map-glicko(k_map={k_map:g},shrink={shrink:g},{'veto-known' if known_maps else 'veto-unknown'})"

    def map_offset(self, players, map_name):
        if map_name in UNKNOWN_MAPS:
            return 0.0
        return sum(self.o[(p, map_name)] for p in players) / len(players)

    def _off(self, m, map_name):
        return self.map_offset(m.team1_players, map_name) - self.map_offset(m.team2_players, map_name)

    def predict_map(self, m, map_name):
        r1, rd1 = self.team(m.team1_players)
        r2, rd2 = self.team(m.team2_players)
        return elo_expected(self.g(math.sqrt(rd1 ** 2 + rd2 ** 2)) * (r1 - r2) + self._off(m, map_name))

    def predict(self, m):
        base = self._p_map(m)
        if not self.known_maps:
            return series_prob(base, m.best_of)
        probs = [self.predict_map(m, mp.name) for mp in m.maps]
        probs = probs[: m.best_of] + [base] * (m.best_of - len(probs))
        return series_prob_seq(probs, m.best_of)

    def update(self, m):
        self._decay(m.team1_players, m.time)
        self._decay(m.team2_players, m.time)
        for mp in m.maps:
            s = 1.0 if mp.t1_won else 0.0
            off = self._off(m, mp.name)
            p = self.predict_map(m, mp.name)
            r1, rd1 = self.team(m.team1_players)
            r2, rd2 = self.team(m.team2_players)
            self._update_side(m.team1_players, r2, rd2, s, off)
            self._update_side(m.team2_players, r1, rd1, 1.0 - s, -off)
            if mp.name in UNKNOWN_MAPS:
                continue
            d = self.k_map * (s - p)
            for pl in m.team1_players:
                key = (pl, mp.name)
                self.o[key] = self.o[key] * (1 - self.shrink) + d
            for pl in m.team2_players:
                key = (pl, mp.name)
                self.o[key] = self.o[key] * (1 - self.shrink) - d


class OnlineScale(Model):
    """Wraps a model and learns a logit temperature online: P = sigmoid(a * logit(p_inner)).

    a < 1 means the inner model is overconfident. Learned by SGD on log loss from past matches only.
    """

    def __init__(self, inner: Model, lr: float = 0.002):
        self.inner, self.lr, self.a = inner, lr, 1.0
        self.name = f"{inner.name}+scale(lr={lr:g})"

    def _z(self, m):
        p = min(1 - 1e-6, max(1e-6, self.inner.predict(m)))
        return math.log(p / (1 - p))

    def predict(self, m):
        return 1.0 / (1.0 + math.exp(-self.a * self._z(m)))

    def update(self, m):
        z = self._z(m)
        p = self.predict(m)
        self.a += self.lr * ((1.0 if m.t1_won else 0.0) - p) * z
        self.inner.update(m)


class RegionalGlicko(PlayerGlicko):
    """PlayerGlicko with a regional prior, in two independent parts.

    seed=True: a player's first rating is the current mean rating of every rated player from the
    same region instead of the global start. Cold-start matches are where most of the residual
    miscalibration lives, and "unknown Brazilian team" is a better prior than 1500. Averaging
    only settled players (seed_rd=80) is worse: they are the survivors and rate above newcomers.

    offset=True: a per-region logit offset, learned online (SGD on log loss) from cross-region
    matches only, added to the rating difference. Teams that never leave their region are only
    rated relative to each other; the offset moves the whole region when its travellers win or
    lose abroad, which is the hierarchical fix for regional drift.
    """

    def __init__(self, seed: bool = True, offset: bool = True, seed_rd: float = float("inf"),
                 offset_lr: float = 2.0, start_rd: float = 150.0, c: float = 20.0, min_rd: float = 30.0,
                 round_weight: float = 0.0, round_scale: float = 0.25, seed_offset: float = 0.0,
                 exp_lr: float = 0.0, exp_beta: float = 0.0, exp_maps: float = 0.0, **kw):
        super().__init__(start_rd=start_rd, c=c, min_rd=min_rd, **kw)
        # experience term (off by default, not shipped; see README): team strength gets exp_beta rating points
        # times the players' mean experience, log1p(maps played) or, with exp_maps > 0, 1 - exp(-maps / exp_maps).
        # exp_lr > 0 learns exp_beta online like the region offset. rankings.display_rating ignores it.
        self.exp_lr, self.exp_beta, self.exp_maps = exp_lr, exp_beta, exp_maps
        self.nmaps = defaultdict(int)             # player -> maps played
        # seed_offset: rating points added to a newcomer's regional seed (negative = newcomers are weaker)
        self.seed_offset = seed_offset
        # round_weight > 0: after each map's binary update, a second update treating the map's rounds
        # as round_weight * rounds Bernoulli observations on a logit scale of round_scale (see BatchBT)
        self.round_weight, self.round_scale = round_weight, round_scale
        self.seed, self.offset, self.seed_rd, self.offset_lr = seed, offset, seed_rd, offset_lr
        self.region_of = {}                       # player -> region
        self.members = defaultdict(set)           # region -> players
        self.o = defaultdict(float)               # region -> rating-point offset
        tag = "+".join(x for x, on in (("seed", seed), ("offset", offset)) if on) or "none"
        self.name = f"regional-glicko[{tag}](rd0={start_rd:g},c={c:g},lr={offset_lr:g})"
        if seed_offset:
            self.name += f"+seedoff({seed_offset:g})"
        if round_weight:
            self.name += f"+rounds({round_weight:g}x{round_scale:g})"
        if exp_lr or exp_beta:
            self.name += f"+exp(b0={exp_beta:g},lr={exp_lr:g},maps={exp_maps:g})"

    def region_mean(self, region):
        vals = [self.r[p] for p in self.members[region] if self.rd[p] <= self.seed_rd]
        return sum(vals) / len(vals) if len(vals) >= 5 else self.start

    def _register(self, players, countries):
        from .regions import country_region
        for p, cc in zip(players, countries):
            if p in self.region_of:
                continue
            reg = country_region(cc)
            self.region_of[p] = reg
            if self.seed:
                self.r[p] = self.region_mean(reg) + self.seed_offset
            self.members[reg].add(p)

    def experience(self, players):
        if self.exp_maps:  # saturating: 1 - exp(-maps / exp_maps)
            return sum(1 - math.exp(-self.nmaps[p] / self.exp_maps) for p in players) / len(players)
        return sum(math.log1p(self.nmaps[p]) for p in players) / len(players)

    def _off(self, m):
        off = self.o[m.team1_region] - self.o[m.team2_region] if self.offset else 0.0
        if self.exp_beta or self.exp_lr:
            off += self.exp_beta * (self.experience(m.team1_players) - self.experience(m.team2_players))
        return off

    def _p_map(self, m):
        self._register(m.team1_players, m.team1_countries)
        self._register(m.team2_players, m.team2_countries)
        r1, rd1 = self.team(m.team1_players)
        r2, rd2 = self.team(m.team2_players)
        return elo_expected(self.g(math.sqrt(rd1 ** 2 + rd2 ** 2)) * (r1 - r2 + self._off(m)))

    def update(self, m):
        self._register(m.team1_players, m.team1_countries)
        self._register(m.team2_players, m.team2_countries)
        self._decay(m.team1_players, m.time)
        self._decay(m.team2_players, m.time)
        cross = self.offset and m.team1_region != m.team2_region
        maps = m.maps if self.per_map else [None]
        for mp in maps:
            s = (1.0 if mp.t1_won else 0.0) if mp else (1.0 if m.t1_won else 0.0)
            off = self._off(m)
            r1, rd1 = self.team(m.team1_players)
            r2, rd2 = self.team(m.team2_players)
            p = elo_expected(self.g(math.sqrt(rd1 ** 2 + rd2 ** 2)) * (r1 - r2 + off))
            if cross:
                d = self.offset_lr * (s - p)
                self.o[m.team1_region] += d
                self.o[m.team2_region] -= d
            if self.exp_lr:
                self.exp_beta += self.exp_lr * (s - p) * (self.experience(m.team1_players) - self.experience(m.team2_players))
            self._update_side(m.team1_players, r2, rd2, s, self.g(rd2) * off)
            self._update_side(m.team2_players, r1, rd1, 1.0 - s, -self.g(rd1) * off)
            if self.round_weight and mp and mp.valid_for_margin:
                r1, rd1 = self.team(m.team1_players)
                r2, rd2 = self.team(m.team2_players)
                n, share = self.round_weight * (mp.t1 + mp.t2), mp.t1_round_share
                self._update_side(m.team1_players, r2, rd2, share, self.g(rd2) * off, self.round_scale, n)
                self._update_side(m.team2_players, r1, rd1, 1.0 - share, -self.g(rd1) * off, self.round_scale, n)
        for p in m.team1_players + m.team2_players:
            self.nmaps[p] += len(maps)


class Blend(Model):
    """Average of two models' series logits (w on the first). Both are updated on every match."""

    def __init__(self, a: Model, b: Model, w: float = 0.5):
        self.a, self.b, self.w = a, b, w
        self.name = f"blend({w:g}*[{a.name}] + [{b.name}])"

    @staticmethod
    def _z(p):
        p = min(1 - 1e-6, max(1e-6, p))
        return math.log(p / (1 - p))

    def predict(self, m):
        z = self.w * self._z(self.a.predict(m)) + (1 - self.w) * self._z(self.b.predict(m))
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, m):
        self.a.update(m)
        self.b.update(m)
