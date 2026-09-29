"""Price scheduled matches: fresh Liquipedia pages for ongoing and upcoming tournaments, priced by the shipped model.

Fetches the current wikitext of every CS2 tournament page whose dates overlap [now - 2 days, now + --days] (plus
any page new to the category since the last full export) into a cache directory stamped with the current hour,
so the historical cache in data/liquipedia/cache stays untouched and a rerun within the hour costs nothing. A
match is upcoming when both opponents are known, it has no map results and it starts within the window.

Lineups come from the page's TeamCard / TeamParticipants (with per-match stand-ins), as in the export; a team
without one is priced with its latest lineup in the data ("recent"), or as its synthetic team entity. The model
is `rankings.best_model` trained on everything in --data.

Usage:
  .venv/bin/python -m predict.upcoming                      # next 3 days
  .venv/bin/python -m predict.upcoming --days 7 --tier S,A --csv upcoming.csv
  .venv/bin/python -m predict.upcoming --offline            # reuse the latest fetch, no network
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import re
import sys

from .data import load_matches
from .liquipedia import Client
from .liquipedia_export import (CATEGORY, COUNTRIES, LP_DIR, PAGES, TEAMS, convert, country_iso, lan, parse_page,
                                player_id, prize, resolve_teams)
from .rankings import best_model, current_lineups, implied_map_prob, synthetic_rosters

DAY = 86400


def _date(s: str) -> dt.date | None:
    m = re.match(r"\s*(\d{4})-(\d{2})-(\d{2})", s or "")
    if not m:
        return None
    try:
        return dt.date(*map(int, m.groups()))
    except ValueError:
        return None


def live_titles(pages: dict[str, str], today: dt.date, days: int) -> list[str]:
    out = []
    for title, text in pages.items():
        info = parse_page(title, text)["info"]
        s, e = _date(info.get("sdate") or info.get("startdate")), _date(info.get("edate") or info.get("enddate"))
        if e is None and s is None:
            continue
        s, e = s or e, e or s
        if e >= today - dt.timedelta(days=2) and s <= today + dt.timedelta(days=days):
            out.append(title)
    return out


def fetch(now: dt.datetime, days: int, offline: bool, contact: str | None):
    """(title -> fresh wikitext of the live tournament pages, client), cached under data/liquipedia/live/<UTC hour>/."""
    stamp = now.strftime("%Y-%m-%dT%H")
    if offline:  # the latest fetch, whatever hour it was
        stamp = max((d.name for d in (LP_DIR / "live").iterdir() if d.is_dir()), default=stamp)
    client = Client(contact, cache=LP_DIR / "live" / stamp, offline=offline)
    client.stamp = stamp
    known = json.loads(PAGES.read_text())
    titles, cont = [], {}
    while True:
        d = client.get(action="query", list="categorymembers", cmtitle=CATEGORY, cmlimit="500", cmnamespace="0", **cont)
        titles += [p["title"] for p in d["query"]["categorymembers"]]
        if "continue" not in d:
            break
        cont = {"cmcontinue": d["continue"]["cmcontinue"]}
    todo = sorted(set(live_titles(known, now.date(), days)) | (set(titles) - set(known)))
    pages = client.wikitext(todo)
    # a new page may be a finished event (added late to the category); keep only live ones
    live = set(live_titles(pages, now.date(), days))
    print(f"{len(titles)} tournament pages, {len(todo)} fetched ({len(set(titles) - set(known))} new), "
          f"{len(live)} live; {client.requests} API requests", file=sys.stderr)
    return {t: pages[t] for t in live}, client


def upcoming(pages, team_title, countries, start, end):
    """Unplayed matches with both opponents known, starting in [start, end)."""
    out = []
    for title, text in pages.items():
        pg = parse_page(title, text)
        info = pg["info"]
        lineup = {}
        for c in pg["cards"]:
            if len(c["players"]) == 5:
                lineup.setdefault(team_title.get(c["team"], c["team"]).lower(), c["players"])
        for m in pg["matches"]:
            if m["walkover"] or m["maps"] or m["time"] is None or not start <= m["time"] < end:
                continue
            if m["winner"] in ("1", "2"):
                continue
            t1, t2 = team_title.get(m["t1"], m["t1"]), team_title.get(m["t2"], m["t2"])
            sides = []
            for team, subs in ((t1, m["subs1"]), (t2, m["subs2"])):
                pl = lineup.get(team.lower())
                if pl is None:
                    sides.append((team, None, None))
                    continue
                pl = list(pl)
                for s in subs:  # announced stand-ins for this match
                    outp = {s["out"]["nick"].lower(), s["out"]["link"].lower()}
                    i = next((i for i, x in enumerate(pl) if x["nick"].lower() in outp or x["link"].lower() in outp), None)
                    if i is not None and s["games"] is None:
                        pl[i] = s["in"]
                ids = tuple(player_id(x) for x in pl)
                ccs = tuple(x["flag"] or country_iso(countries.get(x["link"].lower(), "")) or "world" for x in pl)
                sides.append((team, ids, ccs))
            out.append({"time": m["time"], "event": title, "tier": info.get("liquipediatier", ""),
                        "lan": lan(info), "prize": prize(info), "best_of": m["bestof"] or 3, "sides": sides})
    return sorted(out, key=lambda x: x["time"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/matchdata_liquipedia_sides.json")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--tier", help="comma list of Liquipedia tiers to show, e.g. S,A")
    ap.add_argument("--csv", help="also write the prices here")
    ap.add_argument("--offline", action="store_true", help="reuse the latest fetch, no network")
    ap.add_argument("--contact", help="contact for the User-Agent (default: git user.email)")
    args = ap.parse_args()

    now = dt.datetime.now(dt.timezone.utc)
    pages, client = fetch(now, args.days, args.offline, args.contact)
    team_title = json.loads(TEAMS.read_text())
    keys = set()
    for title, text in pages.items():
        pg = parse_page(title, text)
        keys |= {c["team"] for c in pg["cards"]} | {k for m in pg["matches"] for k in (m["t1"], m["t2"])}
    if not args.offline and keys - set(team_title):
        team_title = resolve_teams(client, keys)   # new team keys; this writes the shared team_titles.json cache
    countries = json.loads(COUNTRIES.read_text()) if COUNTRIES.exists() else {}
    t0 = int(now.timestamp())
    todo = upcoming(pages, team_title, countries, t0 - 6 * 3600, t0 + args.days * DAY)
    if args.tier:
        tiers = {t.strip().upper() for t in args.tier.split(",")}
        todo = [x for x in todo if x["tier"].split("-")[0].upper() in tiers]
    if not todo:
        print("no upcoming matches in the window")
        return

    matches = load_matches(args.data)
    model, glicko, batch = best_model(synthetic_rosters(matches))
    print(f"training on {len(matches)} matches...", file=sys.stderr)
    for m in matches:
        model.update(m)
    # results on the fresh pages that the data file does not have yet (it lags the live pages by days)
    fresh_path = LP_DIR / "live" / client.stamp / "finished.json"
    fresh_path.write_text(json.dumps(convert(pages, team_title, countries)[0]))
    fresh = [m for m in load_matches(fresh_path) if matches[-1].time < m.time < t0]
    for m in fresh:
        model.update(m)
    matches += fresh
    print(f"+ {len(fresh)} finished matches from the live pages", file=sys.stderr)
    batch.fit(matches[-1].time + 1)
    recent = {tid.lower(): v for tid, v in current_lineups(matches).items()}
    last_data = matches[-1].time
    print(f"data through {dt.datetime.fromtimestamp(last_data, dt.timezone.utc):%Y-%m-%d}; "
          f"pricing {len(todo)} matches (P = team1 wins the series)\n")

    rows = []
    for x in todo:
        (n1, p1, c1), (n2, p2, c2) = x["sides"]
        src = []
        resolved = []
        for name, pl, cc in ((n1, p1, c1), (n2, p2, c2)):
            if pl is not None:
                src.append("event")
            elif name.lower() in recent:
                _, pl, cc, _ = recent[name.lower()]
                src.append("recent")
            else:
                pl, cc = (f"team:{name.lower()}",), ("world",)
                src.append("synthetic")
            resolved.append((pl, cc))
        (p1, c1), (p2, c2) = resolved
        m = dataclasses.replace(matches[-1], time=max(x["time"], last_data + 1), team1_id=n1, team2_id=n2,
                                team1_name=n1, team2_name=n2, team1_players=p1, team2_players=p2,
                                team1_countries=c1, team2_countries=c2, best_of=x["best_of"], event_id=x["event"],
                                event_name=x["event"], lan=x["lan"],
                                prize_pool=float(x["prize"].strip("$") or 0), maps=[], extra={})
        p = model.predict(m)
        when = dt.datetime.fromtimestamp(x["time"], dt.timezone.utc)
        rows.append({"start_utc": f"{when:%Y-%m-%d %H:%M}", "event": x["event"], "tier": x["tier"], "best_of": x["best_of"],
                     "team1": n1, "team2": n2, "p_team1": round(p, 4), "p_map": round(implied_map_prob(p, x["best_of"]), 4),
                     "lineups": "/".join(src)})
    print(f"{'start (UTC)':16s} {'tier':7s} {'bo':>2s}  {'team1':24s} {'team2':24s} {'P1':>6s} {'map':>6s}  lineups  event")
    for r in rows:
        print(f"{r['start_utc']:16s} {r['tier'][:7]:7s} {r['best_of']:2d}  {r['team1'][:24]:24s} {r['team2'][:24]:24s} "
              f"{r['p_team1']:6.3f} {r['p_map']:6.3f}  {r['lineups']:15s} {r['event']}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
