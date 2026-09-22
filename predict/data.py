"""Load the Valve match JSON into a flat, time-ordered list of Match records."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "matchdata_sample_20230829.json"

# Maps with fewer total rounds than this are forfeits / data errors and carry no round-share signal.
MIN_ROUNDS_FOR_MARGIN = 10


@dataclass
class MapResult:
    name: str
    t1: int
    t2: int

    @property
    def t1_won(self) -> bool:
        return self.t1 > self.t2

    @property
    def valid_for_margin(self) -> bool:
        return self.t1 + self.t2 >= MIN_ROUNDS_FOR_MARGIN

    @property
    def t1_round_share(self) -> float:
        return self.t1 / (self.t1 + self.t2)


@dataclass
class Match:
    time: int
    team1_id: str
    team2_id: str
    team1_name: str
    team2_name: str
    team1_players: tuple[str, ...]
    team2_players: tuple[str, ...]
    event_id: str
    event_name: str
    lan: bool
    prize_pool: float
    maps: list[MapResult]
    t1_won: bool
    best_of: int  # 1, 3 or 5, inferred from maps the winner took
    team1_countries: tuple[str, ...] = ()
    team2_countries: tuple[str, ...] = ()
    event_region: str = ""      # tournament region as the source labels it (PandaScore only)
    forfeited: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def team1_region(self) -> str:
        from .regions import team_region
        return team_region(self.team1_countries)

    @property
    def team2_region(self) -> str:
        from .regions import team_region
        return team_region(self.team2_countries)

    @property
    def winner_maps(self) -> int:
        return (self.best_of + 1) // 2


def _parse_prize(s) -> float:
    if not s:
        return 0.0
    s = s.replace(",", "").replace("$", "")
    return float(s) if s.isdigit() else 0.0


def load_matches(path: Path | str = DEFAULT_DATA, require_full_rosters: bool | None = None) -> list[Match]:
    """require_full_rosters=None keeps 5-man lineups only when the file has them (Valve export),
    and accepts synthetic single-entry rosters from the PandaScore export."""
    raw = json.loads(Path(path).read_text())
    if require_full_rosters is None:
        require_full_rosters = not any(
            p["playerId"].startswith("team:") for m in raw["matches"][:50] for p in m["team1Players"]
        )
    events = {e["eventId"]: e for e in raw["events"]}
    out: list[Match] = []
    for m in raw["matches"]:
        p1 = tuple(p["playerId"] for p in m["team1Players"])
        p2 = tuple(p["playerId"] for p in m["team2Players"])
        if require_full_rosters and (len(p1) != 5 or len(p2) != 5):
            continue
        if not p1 or not p2:
            continue
        maps = [MapResult(x["mapName"], x["team1Score"], x["team2Score"]) for x in m["maps"]]
        t1_won = m["winningTeam"] == 1
        winner_maps = sum(1 for mp in maps if mp.t1_won == t1_won)
        best_of = m.get("bestOf") or {1: 1, 2: 3, 3: 5}.get(winner_maps, 2 * winner_maps - 1)
        ev = events.get(m.get("eventId"), {})
        out.append(
            Match(
                time=m["matchStartTime"],
                team1_id=m["team1Id"],
                team2_id=m["team2Id"],
                team1_name=m["team1Name"],
                team2_name=m["team2Name"],
                team1_players=p1,
                team2_players=p2,
                event_id=m.get("eventId", ""),
                event_name=ev.get("eventName", ""),
                lan=bool(ev.get("lan", False)),
                prize_pool=_parse_prize(ev.get("prizePool")),
                maps=maps,
                t1_won=t1_won,
                best_of=best_of,
                team1_countries=tuple(p.get("countryIso") or "" for p in m["team1Players"]),
                team2_countries=tuple(p.get("countryIso") or "" for p in m["team2Players"]),
                event_region=ev.get("region") or "",
                forfeited=bool(m.get("forfeited", False)),
            )
        )
    out.sort(key=lambda x: x.time)
    return out
