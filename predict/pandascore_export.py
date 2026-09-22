"""Export Counter-Strike matches from PandaScore into Valve's matchdata JSON schema.

Free-plan endpoints only (GET /csgo/matches/past and GET /csgo/teams). Requires a token:
  export PANDASCORE_TOKEN=...            (or --token, or a .pandascore_token file in the repo root)

Usage:
  .venv/bin/python -m predict.pandascore_export --since 2023-09-01            # full backfill
  .venv/bin/python -m predict.pandascore_export --since 2026-09-01 --until 2026-09-22
  .venv/bin/python -m predict.pandascore_export --rosters                     # also snapshot team rosters

Outputs (all gitignored):
  data/pandascore/matches_raw.jsonl     one PandaScore match per line, append-only cache
  data/pandascore/teams_raw.json        latest /csgo/teams snapshot (only with --rosters)
  data/matchdata_pandascore.json        Valve-schema file that predict.data.load_matches() reads

What the free plan does and does not give us:
  - per-game winner (so per-map binary updates work), series winner, number_of_games, forfeit flag
  - tournament id/name/prizepool/tier/region  ->  events[] entries
  - team ids and names                        ->  team1Id/team2Id
  - NO round scores: maps are written as 1-0 so the loader treats them as "no margin data"
  - NO per-match lineups: players are written as a single synthetic "team:<id>" player, so the
    player-level models degrade to team-level models. --rosters attaches the current roster from
    /csgo/teams to matches played after the snapshot date only; it is never back-applied.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "pandascore"
RAW_MATCHES = RAW_DIR / "matches_raw.jsonl"
RAW_TEAMS = RAW_DIR / "teams_raw.json"
OUT = ROOT / "data" / "matchdata_pandascore.json"
API = "https://api.pandascore.co"

MAX_PER_MINUTE = 55  # free tier burst limit is 60/min, 1000/hour


def get_token(cli: str | None) -> str:
    tok = cli or os.environ.get("PANDASCORE_TOKEN")
    f = ROOT / ".pandascore_token"
    if not tok and f.exists():
        tok = f.read_text().strip()
    if not tok:
        sys.exit("no PandaScore token: set PANDASCORE_TOKEN, pass --token, or create .pandascore_token")
    return tok


class Client:
    def __init__(self, token: str):
        self.token = token
        self.calls: list[float] = []

    def _throttle(self):
        now = time.time()
        self.calls = [t for t in self.calls if now - t < 60]
        if len(self.calls) >= MAX_PER_MINUTE:
            wait = 60 - (now - self.calls[0]) + 0.5
            print(f"  throttling {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
        self.calls.append(time.time())

    def get(self, path: str, params: dict) -> tuple[list | dict, dict]:
        url = f"{API}{path}?{urllib.parse.urlencode(params)}"
        for attempt in range(6):
            self._throttle()
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.token}", "Accept": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.loads(r.read()), dict(r.headers)
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = int(e.headers.get("Retry-After", "60"))
                    print(f"  429, sleeping {wait}s", file=sys.stderr)
                    time.sleep(wait)
                    continue
                if e.code in (401, 403):
                    sys.exit(f"HTTP {e.code} from PandaScore: check the token / plan ({e.read()[:200]!r})")
                if e.code >= 500:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise
        raise RuntimeError(f"gave up on {url}")


def iso(d: str, end=False) -> str:
    t = dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc)
    if end:
        t += dt.timedelta(days=1)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_matches(client: Client, since: str, until: str | None) -> int:
    """Page through /csgo/matches/past within [since, until) and append new matches to the cache."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    seen = set()
    if RAW_MATCHES.exists():
        with RAW_MATCHES.open() as f:
            for line in f:
                seen.add(json.loads(line)["id"])
    until = until or dt.date.today().isoformat()
    params = {
        "sort": "begin_at",
        "per_page": 100,
        "range[begin_at]": f"{iso(since)},{iso(until, end=True)}",
        "filter[finished]": "true",
    }
    added = 0
    page = 1
    with RAW_MATCHES.open("a") as out:
        while True:
            params["page"] = page
            data, headers = client.get("/csgo/matches/past", params)
            if not data:
                break
            for m in data:
                if m["id"] in seen:
                    continue
                seen.add(m["id"])
                out.write(json.dumps(m, separators=(",", ":")) + "\n")
                added += 1
            total = headers.get("X-Total") or headers.get("x-total") or "?"
            print(f"  page {page}: {len(data)} matches (total in range {total}), new so far {added}", file=sys.stderr)
            if len(data) < 100:
                break
            page += 1
    return added


def fetch_teams(client: Client) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    teams, page = [], 1
    while True:
        data, _ = client.get("/csgo/teams", {"per_page": 100, "page": page, "sort": "id"})
        if not data:
            break
        teams.extend(data)
        print(f"  teams page {page}: {len(data)}", file=sys.stderr)
        if len(data) < 100:
            break
        page += 1
    RAW_TEAMS.write_text(json.dumps({"snapshot_at": dt.datetime.now(dt.timezone.utc).isoformat(), "teams": teams}))


def parse_prize(s) -> str:
    """'5000 United States Dollar' -> '$5000' (Valve's loader only reads USD digit strings)."""
    if not s:
        return ""
    parts = s.replace(",", "").split()
    if parts and parts[0].isdigit() and "Dollar" in s:
        return f"${parts[0]}"
    return ""


def synthetic_players(team_id, name, location=None):
    """One stand-in 'player' per team. Its countryIso carries the team's PandaScore location so
    region-aware models work without lineups."""
    return [{"playerId": f"team:{team_id}", "nick": name, "country": "", "countryIso": (location or "world").lower(), "steamIds": []}]


def real_players(roster):
    return [
        {"playerId": str(p["id"]), "nick": p.get("name") or "", "country": "", "countryIso": (p.get("nationality") or "world").lower(), "steamIds": []}
        for p in roster
    ]


def convert() -> tuple[int, int]:
    rosters, snapshot_ts = {}, None
    if RAW_TEAMS.exists():
        snap = json.loads(RAW_TEAMS.read_text())
        snapshot_ts = dt.datetime.fromisoformat(snap["snapshot_at"]).timestamp()
        for t in snap["teams"]:
            if len(t.get("players") or []) >= 5:
                rosters[t["id"]] = t["players"][:5]

    matches, events = [], {}
    with RAW_MATCHES.open() as f:
        for line in f:
            m = json.loads(line)
            if m.get("draw") or len(m.get("opponents") or []) != 2 or not m.get("winner_id") or not m.get("begin_at"):
                continue
            t1 = m["opponents"][0]["opponent"]
            t2 = m["opponents"][1]["opponent"]
            start = int(dt.datetime.fromisoformat(m["begin_at"].replace("Z", "+00:00")).timestamp())
            maps = []
            for g in sorted(m.get("games") or [], key=lambda g: g["position"]):
                w = (g.get("winner") or {}).get("id")
                if not g.get("finished") or w is None:
                    continue
                maps.append({"mapName": "de_default", "team1Score": 1 if w == t1["id"] else 0, "team2Score": 1 if w == t2["id"] else 0})
            if not maps:
                continue
            tour = m.get("tournament") or {}
            ev_id = str(m.get("tournament_id") or tour.get("id") or m.get("serie_id"))
            if ev_id not in events:
                league = (m.get("league") or {}).get("name", "")
                serie = (m.get("serie") or {}).get("full_name", "")
                events[ev_id] = {
                    "eventId": ev_id,
                    "eventName": " ".join(x for x in (league, serie, tour.get("name", "")) if x),
                    "prizePool": parse_prize(tour.get("prizepool")),
                    "lan": False,
                    "tier": tour.get("tier"),
                    "region": tour.get("region"),
                    "finished": True,
                    "winnerTeamId": str(tour["winner_id"]) if tour.get("winner_id") else None,
                    "prizeDistribution": [],
                }

            def players_for(t):
                if snapshot_ts and start >= snapshot_ts and t["id"] in rosters:
                    return real_players(rosters[t["id"]])
                return synthetic_players(t["id"], t["name"], t.get("location"))

            matches.append(
                {
                    "matchStartTime": start,
                    "team1Id": str(t1["id"]),
                    "team2Id": str(t2["id"]),
                    "team1Name": t1["name"],
                    "team2Name": t2["name"],
                    "team1Players": players_for(t1),
                    "team2Players": players_for(t2),
                    "eventId": ev_id,
                    "maps": maps,
                    "winningTeam": 1 if m["winner_id"] == t1["id"] else 2,
                    "forfeited": bool(m.get("forfeit")),
                    "bestOf": m.get("number_of_games"),
                    "pandascoreId": m["id"],
                }
            )
    matches.sort(key=lambda x: x["matchStartTime"])
    OUT.write_text(json.dumps({"matches": matches, "events": list(events.values())}, separators=(",", ":")))
    return len(matches), len(events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token")
    ap.add_argument("--since", default="2023-09-01")
    ap.add_argument("--until", default=None)
    ap.add_argument("--rosters", action="store_true", help="also snapshot /csgo/teams rosters")
    ap.add_argument("--convert-only", action="store_true", help="skip the API, just rebuild the Valve-schema file")
    args = ap.parse_args()

    if not args.convert_only:
        client = Client(get_token(args.token))
        print(f"fetching matches {args.since} .. {args.until or 'today'}", file=sys.stderr)
        added = fetch_matches(client, args.since, args.until)
        print(f"added {added} new matches to {RAW_MATCHES}", file=sys.stderr)
        if args.rosters:
            fetch_teams(client)
    n, e = convert()
    print(f"wrote {OUT}: {n} matches, {e} events")


if __name__ == "__main__":
    main()
