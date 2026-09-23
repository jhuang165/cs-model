"""Attach real 5-man lineups from Valve's published standings details to the PandaScore export.

Every `live/<year>/details/<snapshot>/<rank>--<team>--<roster>.md` page lists the matches that fed that
roster's standing over the prior six months, one row per match with the date (America/Los_Angeles, as
report.js prints it), opponent name, W/L and the five nicks that played. Pages exist only for the
roughly 200-400 top rosters of each snapshot, so a match is covered from both sides only when both teams
were ranked at some snapshot.

The PandaScore free plan has results but no lineups. This joins the two by date and team name:
  1. every details row becomes a side record (date, team, opponent, won, roster), deduplicated across snapshots
  2. each PandaScore team id is mapped to a Valve team name. PandaScore labels matches with a team's current
     name, so exact normalized names only seed the map; ids are then (re)mapped by voting on which Valve
     opponent their mapped opponents met on the same day with the same result (catches renames)
  3. a PandaScore match side gets the roster of the Valve record for that team on that date (vs that
     opponent if possible, otherwise that team's roster that day). A side with no record that day takes
     the team's latest earlier roster within --carry-days (default 90); --carry-future also allows later ones
Sides left uncovered keep their synthetic `team:<id>` player. Players are keyed `nick:<lowercase nick>`
and inherit the team's PandaScore location as their country, so the regional prior still works. Score the
file with models.LinkedGlicko, which carries ratings between a team's
synthetic entity and its players; plain RegionalGlicko loses the team's history at every switch.

Not shipped (see README): the default coverage leaks, because a team has pages only if it is ranked at a
later snapshot. With --causal (lineups only for teams already ranked before the match) there is no gain.

  .venv/bin/python -m predict.live_rosters                    # writes data/matchdata_pandascore_rosters.json
  .venv/bin/python -m predict.run_backtest --data data/matchdata_pandascore_rosters.json --eval-from 2025-07-01
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "live"
PS_FILE = ROOT / "data" / "matchdata_pandascore.json"
OUT = ROOT / "data" / "matchdata_pandascore_rosters.json"
LA = ZoneInfo("America/Los_Angeles")

ROW = re.compile(r"^\|\s*\d+\s*\|\s*(\d+)\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|(.*?)\|\s*([WL])\s*\|.*\|([^|]*)\|\s*$")
TEAM = re.compile(r"^Team Name: (.*?)<br />", re.M)
# suffixes that PandaScore and HLTV attach inconsistently ("Team Spirit" / "Spirit", "FaZe Clan" / "FaZe")
NOISE = re.compile(r"\b(team|esports?|gaming|clan|club|gg|e-?sports?)\b")


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    stripped = re.sub(r"[^a-z0-9]", "", NOISE.sub(" ", s))
    return stripped or re.sub(r"[^a-z0-9]", "", s)


@dataclass(frozen=True)
class Side:
    date: str          # YYYY-MM-DD in Los Angeles time
    team: str          # normalized Valve team name
    opp: str
    won: bool
    roster: tuple[str, ...]


def parse_details(live: Path = LIVE) -> tuple[list[Side], dict[str, str]]:
    """All side records in every details page, deduplicated, plus normalized -> display team name."""
    sides: set[Side] = set()
    display: dict[str, str] = {}
    for f in sorted(live.glob("*/details/*/*.md")):
        text = f.read_text()
        m = TEAM.search(text)
        if not m:
            continue
        team = norm(m.group(1))
        display.setdefault(team, m.group(1))
        for line in text.splitlines():
            r = ROW.match(line)
            if not r:
                continue
            roster = tuple(sorted((p.strip() for p in r.group(5).split(",") if p.strip()), key=str.lower))
            if len(roster) != 5:
                continue
            sides.add(Side(r.group(2), team, norm(r.group(3).strip()), r.group(4) == "W", roster))
    return sorted(sides, key=lambda s: (s.date, s.team, s.opp)), display


def first_ranked(live: Path = LIVE) -> dict[str, str]:
    """Normalized Valve team name -> date (YYYY-MM-DD) of the first snapshot that has a details page for it."""
    first: dict[str, str] = {}
    for f in sorted(live.glob("*/details/*/*.md")):
        m = TEAM.search(f.read_text())
        if m:
            first.setdefault(norm(m.group(1)), f.parent.name.replace("_", "-"))
    return first


def la_date(ts: int, shift_days: int = 0) -> str:
    return (dt.datetime.fromtimestamp(ts, LA).date() + dt.timedelta(days=shift_days)).isoformat()


class Index:
    def __init__(self, sides: list[Side]):
        self.by_pair: dict[tuple[str, str, str], Side] = {}
        self.by_team_day: dict[tuple[str, str], list[Side]] = defaultdict(list)
        self.by_team: dict[str, list[Side]] = defaultdict(list)
        for s in sides:
            self.by_pair.setdefault((s.date, s.team, s.opp), s)
            self.by_team_day[(s.date, s.team)].append(s)
            self.by_team[s.team].append(s)
        self.team_dates = {t: [s.date for s in v] for t, v in self.by_team.items()}

    def lookup(self, ts: int, team: str, opp: str | None) -> tuple[Side | None, str]:
        """Record for `team` around start time `ts`: exact pairing, then same-day roster; the start
        can fall either side of midnight in LA, so the neighbouring days are tried too."""
        for shift in (0, -1, 1):
            d = la_date(ts, shift)
            if opp and (d, team, opp) in self.by_pair:
                return self.by_pair[(d, team, opp)], "pair"
        for shift in (0, -1, 1):
            day = self.by_team_day.get((la_date(ts, shift), team))
            if day:
                return _modal(day), "day"
        return None, ""

    def nearest(self, ts: int, team: str, max_days: int, past_only: bool = False) -> Side | None:
        """The team's record closest in time within `max_days`; `past_only` restricts it to earlier days."""
        dates = self.team_dates.get(team)
        if not dates:
            return None
        d = la_date(ts)
        i = bisect.bisect_left(dates, d)
        best, best_gap = None, max_days + 1
        for j in ((i - 1,) if past_only else (i - 1, i)):
            if 0 <= j < len(dates):
                gap = abs((dt.date.fromisoformat(dates[j]) - dt.date.fromisoformat(d)).days)
                if gap < best_gap:
                    best, best_gap = self.by_team[team][j], gap
        return best


def _modal(day: list[Side]) -> Side:
    top = Counter(s.roster for s in day).most_common(1)[0][0]
    return next(s for s in day if s.roster == top)


def map_team_ids(matches: list[dict], index: Index, valve_names: set[str], rounds: int = 3,
                 min_votes: int = 3, min_share: float = 0.6) -> dict[str, str]:
    """PandaScore team id -> normalized Valve name.

    PandaScore labels every match with the team's *current* name, so names only seed the map: ids whose
    normalized name is a Valve name. Then, repeatedly, for every match where one side is mapped and has an
    unambiguous Valve record that day, the record's opponent is a vote for the other side's id (only if
    the W/L agrees). Any id, seeded or not, whose votes reach `min_votes` with one name holding
    `min_share` of them takes that name; this catches renames (Copenhagen Wolves was "CPH Wolves") and
    overrides seeds that meet a different team of the same name. Several ids may share a name, since
    PandaScore keeps duplicate team entries.
    """
    seed: dict[str, str] = {}
    for m in matches:
        for side in ("team1", "team2"):
            n = norm(m[f"{side}Name"])
            if n in valve_names:
                seed.setdefault(m[f"{side}Id"], n)
    id2name = dict(seed)
    for _ in range(rounds):
        votes: dict[str, Counter] = defaultdict(Counter)
        for m in matches:
            for a, b, a_won in (("team1", "team2", m["winningTeam"] == 1), ("team2", "team1", m["winningTeam"] == 2)):
                ida, idb = m[f"{a}Id"], m[f"{b}Id"]
                if ida not in id2name:
                    continue
                for shift in (0, -1, 1):
                    day = index.by_team_day.get((la_date(m["matchStartTime"], shift), id2name[ida]))
                    if day:
                        opps = {s.opp for s in day if s.won == a_won}
                        if len(opps) == 1:
                            votes[idb][opps.pop()] += 1
                        break
        new = dict(seed)
        for tid, c in votes.items():
            name, n = c.most_common(1)[0]
            if n >= min_votes and n / sum(c.values()) >= min_share:
                new[tid] = name
        if new == id2name:
            break
        id2name = new
    return id2name


def players(roster: tuple[str, ...], country: str) -> list[dict]:
    return [{"playerId": f"nick:{p.lower()}", "nick": p, "country": "", "countryIso": country, "steamIds": []} for p in roster]


def attach(ps: dict, sides: list[Side], carry_days: int = 90, past_only: bool = True,
           ranked: dict[str, str] | None = None) -> tuple[dict, Counter]:
    """`ranked` (team -> first snapshot date) makes coverage causal: a side gets a lineup only if its team
    was already ranked before the match. Otherwise coverage depends on the team being ranked later, so
    exactly the new teams that turn out good get their players' ratings instead of the newcomer seed."""
    index = Index(sides)
    matches = ps["matches"]
    id2name = map_team_ids(matches, index, set(index.by_team) | {s.opp for s in sides})
    stats = Counter()
    out = []
    for m in matches:
        m = dict(m)
        ts = m["matchStartTime"]
        n1, n2 = id2name.get(m["team1Id"]), id2name.get(m["team2Id"])
        for side, me, opp in (("team1", n1, n2), ("team2", n2, n1)):
            stats["sides"] += 1
            if me is None:
                stats["unmapped"] += 1
                continue
            if ranked is not None and ranked.get(me, "9999") >= la_date(ts):
                stats["not_yet_ranked"] += 1
                continue
            rec, how = index.lookup(ts, me, opp)
            if rec is None and carry_days:
                rec, how = index.nearest(ts, me, carry_days, past_only), "carry"
            if rec is None:
                stats["no_record"] += 1
                continue
            if how == "pair" and rec.won != (m["winningTeam"] == (1 if side == "team1" else 2)):
                stats["result_mismatch"] += 1
                continue
            stats[how] += 1
            country = m[f"{side}Players"][0].get("countryIso") or "world"
            m[f"{side}Players"] = players(rec.roster, country)
        both = all(not m[f"{s}Players"][0]["playerId"].startswith("team:") for s in ("team1", "team2"))
        stats["both_real"] += both
        stats["matches"] += 1
        out.append(m)
    stats["mapped_ids"] = len(id2name)
    return {"matches": out, "events": ps["events"]}, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PS_FILE))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--carry-days", type=int, default=90,
                    help="fall back to the team's latest Valve roster within this many days (0 = off)")
    ap.add_argument("--carry-future", action="store_true",
                    help="also carry rosters recorded after the match (not causal)")
    ap.add_argument("--causal", action="store_true",
                    help="only attach lineups for teams ranked at a snapshot before the match")
    args = ap.parse_args()

    sides, _ = parse_details()
    print(f"{len(sides)} side records, {len({s.team for s in sides})} Valve team names, "
          f"{sides[0].date} .. {sides[-1].date}")
    ps = json.loads(Path(args.data).read_text())
    out, st = attach(ps, sides, args.carry_days, past_only=not args.carry_future,
                     ranked=first_ranked() if args.causal else None)
    first = sides[0].date
    in_range = sum(1 for m in out["matches"] if la_date(m["matchStartTime"]) >= first)
    real = st["pair"] + st["day"] + st["carry"]
    print(f"mapped {st['mapped_ids']} PandaScore team ids")
    print(f"sides: {st['sides']}, real roster {real} (pair {st['pair']}, same day {st['day']}, carried {st['carry']}), "
          f"unmapped team {st['unmapped']}, not yet ranked {st['not_yet_ranked']}, no record {st['no_record']}, result mismatch {st['result_mismatch']}")
    print(f"matches with both lineups real: {st['both_real']} of {st['matches']} ({in_range} since {first})")
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
