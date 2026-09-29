"""Export CS2 matches from Liquipedia tournament pages into Valve's matchdata JSON schema.

Source: the current wikitext of every page in Category:CS2 Tournaments, fetched through the MediaWiki API
by `liquipedia.Client` (rate-limited, cached). Liquipedia content is CC-BY-SA 3.0; credit Liquipedia.

What a tournament page gives, and what goes into the file:
  - {{Infobox league}}: dates, prize pool (USD), Liquipedia tier, Online/Offline  ->  events[]
  - {{Match}}: opponents (team template keys), start time with timezone, and per map the map name and
    round scores by half (+ overtime)  ->  maps[] with real round scores, bestOf, hltvId
  - {{TeamCard}}: each participant's lineup (p1..p5, with |pNlink= disambiguation) at that event.
    TeamCards are written for every participant when the event happens, so whether a match has lineups
    does not depend on how the team does later (unlike Valve's standings pages, see live_rosters.py).
Team template keys (`faze`, `washington (british team)`) and TeamCard names are resolved to the team's
page title with {{Team|key}} through action=expandtemplates, in batches spaced like action=parse.
A side whose team has no TeamCard on the page keeps a synthetic `team:<page title>` player. Player
nationalities come from the lead section of each player's page ({{Infobox player|country=}}), 50 per request.

  .venv/bin/python -m predict.liquipedia_export --fetch       # pages, team names, player countries (cached)
  .venv/bin/python -m predict.liquipedia_export               # rebuild data/matchdata_liquipedia.json from cache
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import urllib.parse
from collections import Counter
from pathlib import Path

from .liquipedia import ROOT, Client

LP_DIR = ROOT / "data" / "liquipedia"
PAGES = LP_DIR / "pages.json"
TEAMS = LP_DIR / "team_titles.json"
COUNTRIES = LP_DIR / "player_countries.json"
OUT = ROOT / "data" / "matchdata_liquipedia.json"
CATEGORY = "Category:CS2 Tournaments"
EXPAND_INTERVAL = 30.0   # expandtemplates is as heavy as parse: at most one call per 30 s

TZ = {  # hours east of UTC for the {{Abbr/..}} zones used in match dates
    "UTC": 0, "GMT": 0, "WET": 0, "BST": 1, "WEST": 1, "CET": 1, "CEST": 2, "EET": 2, "EEST": 3, "MSK": 3,
    "TRT": 3, "AST": 3, "GST": 4, "PKT": 5, "IST": 5.5, "ICT": 7, "WIB": 7, "SGT": 8, "PHT": 8, "CST": 8,
    "HKT": 8, "AWST": 8, "KST": 9, "JST": 9, "ACST": 9.5, "AEST": 10, "ACDT": 10.5, "AEDT": 11, "NZST": 12,
    "NZDT": 13, "BRT": -3, "ART": -3, "CLT": -4, "CLST": -3, "EDT": -4, "EST": -5, "CDT": -5, "MDT": -6,
    "MST": -7, "PDT": -7, "PST": -8, "COT": -5, "PET": -5,
}
# country name (lowercase) -> ISO code as regions.py expects; seeded from the Valve sample's players
# plus Liquipedia spellings, extended from the names the player pages actually use
COUNTRY_ISO = {
    "algeria": "dz", "argentina": "ar", "armenia": "am", "australia": "au", "austria": "at",
    "azerbaijan": "az", "belarus": "by", "belgium": "be", "bosnia and herzegovina": "ba", "brazil": "br",
    "bulgaria": "bg", "cambodia": "kh", "canada": "ca", "chile": "cl", "china": "cn", "colombia": "co",
    "croatia": "hr", "czech republic": "cz", "czechia": "cz", "denmark": "dk", "ecuador": "ec",
    "england": "gb", "estonia": "ee", "finland": "fi", "france": "fr", "georgia": "ge", "germany": "de",
    "greece": "gr", "guatemala": "gt", "hong kong": "hk", "hungary": "hu", "iceland": "is", "india": "in",
    "indonesia": "id", "iraq": "iq", "ireland": "ie", "israel": "il", "italy": "it", "jordan": "jo",
    "kazakhstan": "kz", "korea": "kr", "kosovo": "xk", "kyrgyzstan": "kg", "latvia": "lv", "lebanon": "lb",
    "lithuania": "lt", "luxembourg": "lu", "malaysia": "my", "malta": "mt", "mexico": "mx", "moldova": "md",
    "mongolia": "mn", "montenegro": "me", "morocco": "ma", "netherlands": "nl", "new zealand": "nz",
    "north macedonia": "mk", "northern ireland": "gb", "norway": "no", "pakistan": "pk", "palestine": "ps",
    "paraguay": "py", "peru": "pe", "philippines": "ph", "poland": "pl", "portugal": "pt", "romania": "ro",
    "russia": "ru", "saudi arabia": "sa", "scotland": "gb", "serbia": "rs", "slovakia": "sk",
    "slovenia": "si", "solomon islands": "sb", "south africa": "za", "south korea": "kr", "spain": "es",
    "sri lanka": "lk", "sweden": "se", "switzerland": "ch", "syria": "sy", "taiwan": "tw", "thailand": "th",
    "tunisia": "tn", "turkey": "tr", "turkiye": "tr", "türkiye": "tr", "uae": "ae", "ukraine": "ua",
    "united arab emirates": "ae", "united kingdom": "gb", "united states": "us", "uruguay": "uy", "usa": "us",
    "uzbekistan": "uz", "venezuela": "ve", "vietnam": "vn", "wales": "gb", "iran": "ir", "singapore": "sg",
    "albania": "al", "japan": "jp", "kuwait": "kw",
}
# {{Abbr/CST}} is China Standard Time on Liquipedia unless the page says otherwise; US Central is rare
# enough in CS to ignore here.


# ---------------------------------------------------------------- wikitext template parsing

def templates(text: str, name: str):
    """Yield the inner text of every top-level-or-nested {{name ...}} in text (balanced braces)."""
    pat = re.compile(r"\{\{\s*" + re.escape(name) + r"\s*(?=[|}\n])")
    for m in pat.finditer(text):
        depth, i = 0, m.start()
        while i < len(text) - 1:
            two = text[i:i + 2]
            if two == "{{":
                depth += 1
                i += 2
                continue
            if two == "}}":
                depth -= 1
                i += 2
                if depth == 0:
                    yield text[m.end():i - 2]
                    break
                continue
            i += 1


def params(body: str) -> dict[str, str]:
    """Split a template body into named parameters at top-level pipes."""
    out, depth, cur, parts = {}, 0, [], []
    i = 0
    while i < len(body):
        two = body[i:i + 2]
        if two in ("{{", "[["):
            depth += 1
            cur.append(two)
            i += 2
            continue
        if two in ("}}", "]]"):
            depth -= 1
            cur.append(two)
            i += 2
            continue
        if body[i] == "|" and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(body[i])
        i += 1
    parts.append("".join(cur))
    n = 0
    for p in parts[1:] if parts and "=" not in parts[0] else parts:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            n += 1
            out[str(n)] = p.strip()
    return out


def strip_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def clean_key(s: str) -> str:
    """Team key without markup (<s>struck-out</s> teams, bold/italic quotes), lowercased."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>|'{2,}", "", s)).strip().lower()


def num(s: str | None) -> int | None:
    s = (s or "").strip()
    return int(s) if s.isdigit() else None


# ---------------------------------------------------------------- page -> records

DATE = re.compile(r"([A-Z][a-z]+ \d{1,2}, \d{4}|\d{4}-\d{2}-\d{2})(?:\s*-\s*(\d{1,2}):(\d{2}))?(?:.*?\{\{\s*Abbr/([A-Z]+)\s*\}\})?")


def parse_time(s: str) -> int | None:
    m = DATE.search(s or "")
    if not m:
        return None
    d = m.group(1)
    try:
        day = dt.datetime.strptime(d, "%B %d, %Y") if "," in d else dt.datetime.strptime(d, "%Y-%m-%d")
    except ValueError:
        return None
    if m.group(2):
        off = TZ.get(m.group(4) or "UTC", 0)
        t = day.replace(tzinfo=dt.timezone.utc) + dt.timedelta(hours=int(m.group(2)) - off, minutes=int(m.group(3)))
    else:
        t = day.replace(hour=12, tzinfo=dt.timezone.utc)
    return int(t.timestamp())


def opponent_key(s: str) -> tuple[str | None, str, list[dict]]:
    """{{TeamOpponent|key|score=..|substitutes=..}} -> (key, score, stand-ins)."""
    for body in templates(s, "TeamOpponent"):
        p = params(body)
        return clean_key(p.get("1") or p.get("template") or "") or None, p.get("score", ""), substitutes(p.get("substitutes", ""))
    return None, "", []


def person(s: str) -> dict:
    """A plain nick, [[link|nick]] or {{OpponentPlayer|nick|flag=|link=}} -> {nick, link, flag}."""
    for body in templates(s, "OpponentPlayer"):
        q = params(body)
        nick = (q.get("1") or "").strip()
        return {"nick": nick, "link": (q.get("link") or nick).strip(), "flag": (q.get("flag") or "").strip().lower()}
    m = re.match(r"\s*\[\[([^|\]]+)(?:\|([^\]]+))?\]\]", s)
    if m:
        return {"nick": (m.group(2) or m.group(1)).strip(), "link": m.group(1).strip(), "flag": ""}
    nick = re.sub(r"<[^>]*>|'{2,}", "", s).strip()
    return {"nick": nick, "link": nick, "flag": ""}


def substitutes(s: str) -> list[dict]:
    """{{PlayerSubstitutions|{{Substitution|in=|out=|games=}}...}}: per-match stand-ins. games= lists the
    maps the stand-in played ("2", "1 and 2", "2;3"); None means the whole match."""
    out = []
    for body in templates(s, "Substitution"):
        q = params(body)
        pin, pout = person(q.get("in", "")), person(q.get("out", ""))
        if q.get("link"):
            pin["link"] = q["link"].strip()
        if pin["nick"] and pout["nick"]:
            games = {int(x) for x in re.findall(r"\d+", q.get("games", ""))} or None
            out.append({"in": pin, "out": pout, "games": games})
    return out


def parse_map(s: str) -> dict | None:
    for body in templates(s, "Map"):
        p = params(body)
        if p.get("finished", "").lower() == "skip":
            return None
        name = p.get("map", "").strip()
        s1, s2 = num(p.get("score1")), num(p.get("score2"))
        if s1 is None or s2 is None:
            halves = [num(p.get(k)) for k in ("t1t", "t1ct", "t2t", "t2ct")]
            if any(h is None for h in halves):
                if p.get("winner") in ("1", "2") and name:
                    w = p["winner"] == "1"
                    return {"mapName": name, "team1Score": int(w), "team2Score": int(not w)}
                return None
            s1, s2 = halves[0] + halves[1], halves[2] + halves[3]
            for k in range(1, 10):
                ot = [num(p.get(f"o{k}{x}")) for x in ("t1t", "t1ct", "t2t", "t2ct")]
                if all(o is None for o in ot):
                    break
                s1 += (ot[0] or 0) + (ot[1] or 0)
                s2 += (ot[2] or 0) + (ot[3] or 0)
        if s1 == s2:
            return None
        return {"mapName": name or "unknown", "team1Score": s1, "team2Score": s2}
    return None


def parse_page(title: str, text: str) -> dict:
    text = strip_comments(text)
    info = {}
    for body in templates(text, "Infobox league"):
        info = params(body)
        break
    cards = []
    for body in templates(text, "TeamCard"):
        p = params(body)
        team = (p.get("team") or p.get("1") or "").strip()
        if not team:
            continue
        players = []
        for i in range(1, 6):
            nick = re.sub(r"\[\[|\]\]|\{\{.*?\}\}", "", p.get(f"p{i}", "")).strip()
            if not nick:
                break
            link = (p.get(f"p{i}link") or nick).strip()
            players.append({"nick": nick, "link": link, "flag": (p.get(f"p{i}flag") or "").strip().lower()})
        cards.append({"team": clean_key(team), "players": players})
    # Newer pages (most S/A events from 2025) list rosters as {{TeamParticipants|{{Opponent|<team>
    # |players={{Persons|{{Person|<nick>|link=|flag=|role=|status=|played=}}...}}}}...}} instead
    for tp in templates(text, "TeamParticipants"):
        for body in templates(tp, "Opponent"):
            p = params(body)
            team = (p.get("1") or "").strip()
            # starters first; a substitute (or a coach with csub=true) fills in for a starter marked
            # played=false
            ranked = []
            for pb in templates(p.get("players", ""), "Person"):
                q = params(pb)
                role = q.get("role", "").lower()
                nick = (q.get("1") or "").strip()
                if not nick or q.get("played") == "false" or q.get("type") == "staff" or q.get("status") == "former":
                    continue
                if "coach" in role:
                    rank = 2 if q.get("csub") == "true" else None
                elif role == "sub" or q.get("status") == "sub":
                    rank = 1
                else:
                    rank = 0
                if rank is not None:
                    ranked.append((rank, len(ranked), {"nick": nick, "link": (q.get("link") or nick).strip(),
                                                       "flag": (q.get("flag") or "").strip().lower()}))
            players = [x[2] for x in sorted(ranked)[:5]]
            if team and len(players) == 5:
                cards.append({"team": clean_key(team), "players": players})
    matches = []
    for body in templates(text, "Match"):
        p = params(body)
        k1, sc1, subs1 = opponent_key(p.get("opponent1", ""))
        k2, sc2, subs2 = opponent_key(p.get("opponent2", ""))
        if not k1 or not k2 or k1 == "tbd" or k2 == "tbd":
            continue
        maps = [m for i in range(1, 10) if (m := parse_map(p.get(f"map{i}", "")))]
        walkover = p.get("walkover", "") in ("1", "2") or {sc1.upper(), sc2.upper()} & {"W", "FF", "L", "DQ"}
        matches.append({
            "t1": k1, "t2": k2, "time": parse_time(p.get("date", "")), "maps": maps,
            "bestof": num(p.get("bestof")), "hltv": p.get("hltv", "").strip() or None, "walkover": bool(walkover),
            "winner": p.get("winner", "").strip(), "subs1": subs1, "subs2": subs2,
        })
    return {"title": title, "info": info, "cards": cards, "matches": matches}


# ---------------------------------------------------------------- fetching

def fetch_pages(client: Client) -> dict[str, str]:
    titles, cont = [], {}
    while True:
        d = client.get(action="query", list="categorymembers", cmtitle=CATEGORY, cmlimit="500", cmnamespace="0", **cont)
        titles += [p["title"] for p in d["query"]["categorymembers"]]
        if "continue" not in d:
            break
        cont = {"cmcontinue": d["continue"]["cmcontinue"]}
    pages = client.wikitext(titles)
    PAGES.write_text(json.dumps(pages))
    return pages


TEAM_LINK = re.compile(r"link=([^|\]]+)[|\]]")


def resolve_teams(client: Client, keys: set[str]) -> dict[str, str]:
    """Team template key -> team page title via {{Team|key}}; unknown keys resolve to themselves."""
    known = json.loads(TEAMS.read_text()) if TEAMS.exists() else {}
    todo = sorted(k for k in keys if k not in known and "{" not in k and "|" not in k and "~" not in k)
    batches, cur, size = [], [], 0
    for k in todo:
        cost = len(urllib.parse.quote(f"{{{{Team|{k}}}}}~~"))
        if cur and size + cost > 6000:  # URL-encoded length; the server rejects ~8 KB URLs
            batches.append(cur)
            cur, size = [], 0
        cur.append(k)
        size += cost
    if cur:
        batches.append(cur)
    for batch in batches:
        d = client.get(action="expandtemplates", prop="wikitext", _interval=EXPAND_INTERVAL,
                       text="~~".join(f"{{{{Team|{x}}}}}" for x in batch))
        for x, o in zip(batch, d["expandtemplates"]["wikitext"].split("~~")):
            m = TEAM_LINK.search(o)
            known[x] = m.group(1).strip() if m and "Missing template" not in o else x
        TEAMS.write_text(json.dumps(known))
        print(f"  resolved {len(known)} team keys", file=sys.stderr)
    return known


def fetch_player_countries(client: Client, links: set[str]) -> dict[str, str]:
    """Player page link -> country name from {{Infobox player|country=}}, reading only each page's lead
    section (rvsection=0), 50 pages per request. Players without a page are left out."""
    known = json.loads(COUNTRIES.read_text()) if COUNTRIES.exists() else {}
    todo = sorted(x for x in links if x.lower() not in known and not set(x) & set("#<>[]{}|"))
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        d = client.get(action="query", prop="revisions", rvprop="content", rvslots="main", rvsection="0",
                       redirects="1", titles="|".join(batch))
        q = d["query"]
        alias = {}  # final title -> requested links
        for x in batch:
            alias.setdefault(x, []).append(x)
        for step in ("normalized", "redirects"):
            for r in q.get(step, []):
                alias.setdefault(r["to"], []).extend(alias.pop(r["from"], []))
        for p in q.get("pages", []):
            country = ""
            if "revisions" in p:
                for body in templates(p["revisions"][0]["slots"]["main"]["content"], "Infobox player"):
                    country = strip_comments(params(body).get("country", "")).strip()
                    break
            for x in alias.get(p["title"], []):
                known[x.lower()] = country
        if i // 50 % 20 == 0:
            COUNTRIES.write_text(json.dumps(known))
            print(f"  player countries: {len(known)} of {len(known) + len(todo) - i - len(batch)}", file=sys.stderr)
    COUNTRIES.write_text(json.dumps(known))
    return known


def country_iso(name: str) -> str:
    return COUNTRY_ISO.get(strip_comments(name).strip().lower(), "")


# ---------------------------------------------------------------- export

def prize(info: dict) -> str:
    s = (info.get("prizepoolusd") or "").replace(",", "").strip()
    s = s.split(".")[0]
    return f"${s}" if s.isdigit() else ""


def lan(info: dict) -> bool:
    return "offline" in (info.get("type") or "").lower()


def player_id(pl: dict) -> str:
    return "lp:" + pl["link"].strip().lower().replace("_", " ")


def convert(pages: dict[str, str], team_title: dict[str, str], countries: dict[str, str] | None = None,
            lineups: bool = True, stand_ins: bool = True) -> tuple[dict, Counter]:
    countries = countries or {}
    st = Counter()
    events, out, seen = {}, [], set()
    for title in sorted(pages):
        pg = parse_page(title, pages[title])
        info = pg["info"]
        ev_id = title
        events[ev_id] = {
            "eventId": ev_id,
            "eventName": re.sub(r"&nbsp;", " ", info.get("name") or title),
            "prizePool": prize(info),
            "lan": lan(info),
            "tier": info.get("liquipediatier", ""),
            "region": info.get("country") or info.get("region") or "",
            "finished": True,
            "prizeDistribution": [],
        }
        lineup = {}
        for c in pg["cards"]:
            if len(c["players"]) == 5:
                lineup.setdefault(team_title.get(c["team"], c["team"]).lower(), c["players"])
        for m in pg["matches"]:
            st["parsed"] += 1
            if m["walkover"]:
                st["walkover"] += 1
                continue
            if m["time"] is None or not m["maps"]:
                st["no_time_or_maps"] += 1
                continue
            w1 = sum(x["team1Score"] > x["team2Score"] for x in m["maps"])
            w2 = len(m["maps"]) - w1
            if w1 == w2:
                st["unfinished"] += 1
                continue
            t1, t2 = team_title.get(m["t1"], m["t1"]), team_title.get(m["t2"], m["t2"])
            key = m["hltv"] or (m["time"] // 3600, *sorted((t1.lower(), t2.lower())))
            if key in seen:
                st["duplicate"] += 1
                continue
            seen.add(key)

            def side(team, subs):
                pl = lineup.get(team.lower()) if lineups else None
                if pl is None:
                    return [{"playerId": f"team:{team.lower()}", "nick": team, "country": "", "countryIso": "world", "steamIds": []}]
                pl = list(pl)
                for s in subs if stand_ins else ():
                    # one lineup per match: a stand-in counts if they played at least half of its maps
                    if s["games"] is not None and 2 * len(s["games"] & set(range(1, len(m["maps"]) + 1))) < len(m["maps"]):
                        st["stand_in_partial"] += 1
                        continue
                    if any(x["link"].lower() == s["in"]["link"].lower() for x in pl):
                        st["stand_in_listed"] += 1      # the event lineup already has the stand-in
                        continue
                    out = {s["out"]["nick"].lower(), s["out"]["link"].lower()}
                    i = next((i for i, x in enumerate(pl) if x["nick"].lower() in out or x["link"].lower() in out), None)
                    if i is None:
                        st["stand_in_unmatched"] += 1
                        continue
                    pl[i] = s["in"]
                    st["stand_in"] += 1
                return [{"playerId": player_id(x), "nick": x["nick"], "country": countries.get(x["link"].lower(), ""),
                         "countryIso": x["flag"] or country_iso(countries.get(x["link"].lower(), "")) or "world",
                         "steamIds": []} for x in pl]

            p1, p2 = side(t1, m["subs1"]), side(t2, m["subs2"])
            real = [not p[0]["playerId"].startswith("team:") for p in (p1, p2)]
            st["sides_real"] += sum(real)
            st["both_real"] += all(real)
            st["kept"] += 1
            out.append({
                "matchStartTime": m["time"], "team1Id": t1, "team2Id": t2, "team1Name": t1, "team2Name": t2,
                "team1Players": p1, "team2Players": p2, "eventId": ev_id, "maps": m["maps"],
                "winningTeam": 1 if w1 > w2 else 2, "forfeited": False,
                "bestOf": m["bestof"] or (2 * max(w1, w2) - 1), "hltvId": m["hltv"],
            })
    out.sort(key=lambda x: x["matchStartTime"])
    return {"matches": out, "events": list(events.values())}, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="fetch pages and resolve team names (cached)")
    ap.add_argument("--contact", help="contact for the User-Agent (default: git user.email)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-lineups", action="store_true", help="synthetic team players only (lineup-blind baseline)")
    ap.add_argument("--no-stand-ins", action="store_true", help="event lineups only, ignore per-match substitutes")
    args = ap.parse_args()
    LP_DIR.mkdir(parents=True, exist_ok=True)

    client = Client(args.contact, offline=not args.fetch)
    pages = fetch_pages(client) if args.fetch else json.loads(PAGES.read_text())
    keys = set()
    for title, text in pages.items():
        pg = parse_page(title, text)
        keys |= {c["team"] for c in pg["cards"]} | {k for m in pg["matches"] for k in (m["t1"], m["t2"])}
    team_title = resolve_teams(client, keys) if args.fetch else json.loads(TEAMS.read_text())
    if args.fetch:
        links = set()
        for text in pages.values():
            pg = parse_page("", text)
            links |= {x["link"] for c in pg["cards"] for x in c["players"]}
            links |= {s["in"]["link"] for m in pg["matches"] for s in m["subs1"] + m["subs2"]}
        countries = fetch_player_countries(client, links)
    else:
        countries = json.loads(COUNTRIES.read_text()) if COUNTRIES.exists() else {}
    data, st = convert(pages, team_title, countries, lineups=not args.no_lineups, stand_ins=not args.no_stand_ins)
    Path(args.out).write_text(json.dumps(data, separators=(",", ":")))
    print(f"{len(pages)} pages, {st['parsed']} match templates -> {st['kept']} matches "
          f"(walkover {st['walkover']}, no time/maps {st['no_time_or_maps']}, unfinished {st['unfinished']}, "
          f"duplicate {st['duplicate']}); sides with lineups {st['sides_real']} of {2 * st['kept']}, "
          f"both {st['both_real']}; stand-ins applied {st['stand_in']} (partial {st['stand_in_partial']}, already in the lineup {st['stand_in_listed']}, "
          f"out-player not in lineup {st['stand_in_unmatched']})")
    print(f"wrote {args.out} ({client.requests} API requests this run)")


if __name__ == "__main__":
    main()
