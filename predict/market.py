"""Market benchmark: the walk-forward model against pre-match prediction-market prices.

Sources (public, unauthenticated market-data APIs; no account, key, login or scraping):
  Polymarket  gamma-api.polymarket.com/markets/keyset   market metadata, outcomes, scheduled start
              clob.polymarket.com/prices-history         price history of the first outcome's token
  Kalshi      api.elections.kalshi.com/trade-api/v2       /markets (series KXCS2GAME) and 1-minute candlesticks

Only series-winner markets are used: Polymarket `sportsMarketType == "moneyline"` (the "Counter-Strike:
A vs B (BO3) - Event" market; map winners, handicaps and totals are other types), plus older Polymarket
"A vs B" markets that predate the sports tagging, and Kalshi's KXCS2GAME series ("<team> wins" the match).
Markets resolved 50-50 (cancelled, walkover) are dropped.

Pre-match price. The reference time is ref = min(market's scheduled start, Liquipedia's match time). The
price is the last observation at or before ref - OFFSET (default 60 minutes): the last point of the CLOB
price history (a per-minute series of the displayed price, i.e. the book midpoint, or the last trade when
the spread is wide) for Polymarket; the close mid (yes_bid + yes_ask) / 2 of the last 1-minute candle for
Kalshi. Taking the earlier of the two schedules and a full hour of margin keeps delayed or rescheduled
starts from leaking in-play prices; `--offsets` prints the market's log loss at other offsets so the jump
at kick-off is visible. Markets whose pre-match history never moves off its opening value (no trading, the
book never quoted) are flagged `flat` and excluded.

Everything fetched is cached under data/market/raw/ keyed by URL and never requested twice; uncached
requests are spaced MIN_INTERVAL apart. Listing pages are cached too, so markets resolved after the first
run need `--relist` (which only re-fetches listing pages).

Usage: .venv/bin/python -m predict.market                   # fetch what is missing, join, report
       .venv/bin/python -m predict.market --offline         # report from the cache only
       .venv/bin/python -m predict.market --offset 30 --spot 30
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from .liquipedia import ROOT, default_contact

CACHE = ROOT / "data" / "market"
PREDS = ROOT / "data" / "compare" / "matchdata_liquipedia_sides.base.npz"
DATA = ROOT / "data" / "matchdata_liquipedia_sides.json"
MIN_INTERVAL = 0.4          # seconds between uncached requests (< 3 per second)
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
POLY_TAGS = {"100780": "counter-strike-2", "100677": "cs2", "100602": "counter-strike", "100635": "csgo"}
KALSHI_SERIES = ("KXCS2GAME", "KXCSGOGAME")
WINDOW = 36 * 3600          # max |market start - Liquipedia time| for a join
TUNE, CONFIRM = "2024-07-01", "2025-07-01"


# ---------------------------------------------------------------- http

class Http:
    def __init__(self, offline: bool = False, relist: bool = False):
        self.offline, self.relist = offline, relist
        self.last = 0.0
        self.requests = 0
        contact = default_contact()
        self.ua = f"cs-model-research/0.1 (CS2 prediction backtest; {contact})" if contact else "cs-model-research/0.1"

    def _path(self, url: str) -> Path:
        host = urllib.parse.urlparse(url).netloc.split(".")[-2]
        return CACHE / "raw" / host / (hashlib.sha1(url.encode()).hexdigest() + ".json.gz")

    def get(self, base: str, path: str, listing: bool = False, **params):
        url = base + path + ("?" + urllib.parse.urlencode(params) if params else "")
        f = self._path(url)
        if f.exists() and not (listing and self.relist):
            return json.loads(gzip.decompress(f.read_bytes()))["body"]
        if self.offline:
            return None
        for attempt in range(6):
            wait = self.last + MIN_INTERVAL - time.time()
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": self.ua, "Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    body = json.loads(r.read())
            except urllib.error.HTTPError as e:
                if e.code == 429 or e.code >= 500:
                    print(f"  HTTP {e.code} on {url[:120]}, backing off", file=sys.stderr)
                    time.sleep(20 * (attempt + 1))
                    continue
                if e.code in (400, 404):      # e.g. no candlesticks for a market: cache the miss
                    body = {"_error": e.code}
                else:
                    raise
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                print(f"  {e!r}, retrying", file=sys.stderr)
                time.sleep(10 * (attempt + 1))
                continue
            self.requests += 1
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(gzip.compress(json.dumps({"url": url, "fetched": int(time.time()), "body": body}).encode()))
            return body
        raise RuntimeError(f"gave up on {url}")


# ---------------------------------------------------------------- market records

@dataclass
class Rec:
    source: str            # "polymarket" | "kalshi"
    id: str
    a: str                 # team whose price we hold
    b: str
    start: int | None      # scheduled start (unix), None for old Polymarket markets
    a_won: bool
    title: str
    best_of: int | None
    volume: float
    key: str               # CLOB token id of `a` (Polymarket) or market ticker of `a` (Kalshi)
    kind: str = ""         # "moneyline", "legacy", "kalshi"
    extra: dict = field(default_factory=dict)


def _ts(s: str | None) -> int | None:
    if not s:
        return None
    s = s.replace(" ", "T").replace("Z", "+00:00")
    if s.endswith("+00"):
        s += ":00"
    return int(dt.datetime.fromisoformat(s).timestamp())


def _bo(title: str) -> int | None:
    m = re.search(r"\(BO(\d)\)", title, re.I)
    return int(m.group(1)) if m else None


_NOT_SERIES = re.compile(r"map \d|game \d|o/u|over/under|handicap|total|rounds|winner\b|\bwin\b|pistol|first|kills?", re.I)


_OTHER_GAME = re.compile(r"\s*(lol|league of legends|dota|valorant|r6|rainbow|overwatch|cod|call of duty|mlbb|honor of kings)\b", re.I)


def polymarket(http: Http) -> list[Rec]:
    seen, out = set(), []
    for tag in POLY_TAGS:
        cursor, pages = None, 0
        while True:
            params = {"tag_id": tag, "closed": "true", "limit": 100}
            if tag == "100780":      # the sports tag: ~20 prop markets per match, so ask for series winners only
                params["sports_market_types"] = "moneyline"
            if cursor:
                params["after_cursor"] = cursor
            d = http.get(GAMMA, "/markets/keyset", listing=True, **params)
            if d is None:
                break
            pages += 1
            for m in d.get("markets", []):
                if m["id"] in seen:
                    continue
                seen.add(m["id"])
                r = _poly_rec(m)
                if r:
                    out.append(r)
            cursor = d.get("next_cursor")
            if not cursor:
                break
        print(f"  polymarket tag {POLY_TAGS[tag]}: {pages} pages", file=sys.stderr)
    return out


def _poly_rec(m: dict) -> Rec | None:
    try:
        outcomes, prices = json.loads(m["outcomes"]), [float(x) for x in json.loads(m["outcomePrices"])]
        tokens = json.loads(m.get("clobTokenIds") or "[]")
    except (KeyError, ValueError, TypeError):
        return None
    if len(outcomes) != 2 or len(tokens) != 2 or sorted(prices) != [0.0, 1.0]:
        return None                      # unresolved, or resolved 50-50 (cancelled / walkover)
    q = m.get("question", "")
    smt = m.get("sportsMarketType")
    if smt == "moneyline":
        kind = "moneyline"
    elif smt is None and " vs" in q and not _NOT_SERIES.search(q) and not _OTHER_GAME.match(q) \
            and not {o.lower() for o in outcomes} & {"yes", "no", "over", "under"}:
        kind = "legacy"
    else:
        return None
    ev = (m.get("events") or [{}])[0]
    return Rec("polymarket", m["id"], outcomes[0], outcomes[1], _ts(m.get("gameStartTime")) or _ts(m.get("eventStartTime")),
               prices[0] == 1.0, q, _bo(q), float(m.get("volumeNum") or m.get("volume") or 0), tokens[0], kind,
               {"end": _ts(m.get("endDate")), "event": ev.get("title", ""), "created": _ts(m.get("createdAt")),
                "fee_rate": float((m.get("feeSchedule") or {}).get("rate", 0)) if m.get("feesEnabled") else 0.0})


_SCHED = re.compile(r"scheduled for (\w{3} \d{1,2}, \d{4}) at (\d{1,2}:\d{2} [AP]M) (E[DS]T)")


def kalshi(http: Http) -> list[Rec]:
    out = []
    for series in KALSHI_SERIES:
        by_event: dict[str, list] = {}
        for endpoint in ("/markets", "/historical/markets"):
            cursor, pages = None, 0
            while True:
                params = {"series_ticker": series, "limit": 1000}
                if endpoint == "/markets":
                    params["status"] = "settled"
                if cursor:
                    params["cursor"] = cursor
                d = http.get(KALSHI, endpoint, listing=True, **params)
                if d is None or "_error" in d:
                    break
                pages += 1
                for m in d.get("markets", []):
                    by_event.setdefault(m["event_ticker"], {})[m["ticker"]] = m
                cursor = d.get("cursor")
                if not cursor or not d.get("markets"):
                    break
            print(f"  kalshi {series}{endpoint}: {pages} pages", file=sys.stderr)
        for ev, ms in by_event.items():
            ms = sorted(ms.values(), key=lambda m: m["ticker"])
            if len(ms) != 2 or {m.get("result") for m in ms} != {"yes", "no"}:
                continue
            a, b = ms
            mt = _SCHED.search(a.get("rules_primary", ""))
            if mt:
                off = 4 if mt.group(3) == "EDT" else 5
                start = int((dt.datetime.strptime(f"{mt.group(1)} {mt.group(2)}", "%b %d, %Y %I:%M %p")
                             .replace(tzinfo=dt.timezone.utc) + dt.timedelta(hours=off)).timestamp())
            else:        # early markets give the date only: anchor the join at noon ET, price at Liquipedia's time
                start = None
                md = re.search(r"scheduled for (\w{3} \d{1,2}, \d{4})", a.get("rules_primary", ""))
                noon = int((dt.datetime.strptime(md.group(1), "%b %d, %Y").replace(tzinfo=dt.timezone.utc)
                            + dt.timedelta(hours=16)).timestamp()) if md else None
            title = a.get("rules_primary", "")[:200]
            vol = sum(float(m.get("volume_fp") or m.get("volume") or 0) for m in ms)
            out.append(Rec("kalshi", ev, a.get("yes_sub_title") or a["title"], b.get("yes_sub_title") or b["title"],
                           start, a["result"] == "yes", title, _bo(title), vol, a["ticker"], "kalshi",
                           {"series": series, "b_ticker": b["ticker"], "created": _ts(a.get("open_time")),
                            "end": None if start else noon}))
    return out


# ---------------------------------------------------------------- prices

def poly_history(http: Http, token: str, ref: int) -> list[tuple[int, float]] | None:
    d = http.get(CLOB, "/prices-history", market=token, startTs=ref - 24 * 3600, endTs=ref + 3 * 3600, fidelity=1)
    if d is None or "_error" in d:
        return None
    return [(int(x["t"]), float(x["p"])) for x in d.get("history", [])]


def kalshi_history(http: Http, rec: Rec, ref: int) -> list[tuple[int, float, float]] | None:
    """(end of minute, mid, spread) from 1-minute candles of the `a` market; minutes without a two-sided quote are skipped."""
    series = rec.extra["series"]
    d = http.get(KALSHI, f"/series/{series}/markets/{rec.key}/candlesticks",
                 start_ts=ref - 12 * 3600, end_ts=ref + 3 * 3600, period_interval=1)
    if d is None or "_error" in d:
        d = http.get(KALSHI, f"/historical/markets/{rec.key}/candlesticks",
                     start_ts=ref - 12 * 3600, end_ts=ref + 3 * 3600, period_interval=1)
        if d is None or "_error" in d:
            return None
    out = []
    for c in d.get("candlesticks", []):
        try:
            yb, ya = c["yes_bid"], c["yes_ask"]          # live endpoint: close_dollars; historical: close
            bid, ask = float(yb.get("close_dollars", yb.get("close"))), float(ya.get("close_dollars", ya.get("close")))
        except (KeyError, TypeError, ValueError):
            continue
        if 0 < bid < ask < 1:
            out.append((int(c["end_period_ts"]), (bid + ask) / 2, ask - bid))
    return out


def price_at(hist, t: int, max_age: int = 12 * 3600):
    """Last observation at or before t (and not older than max_age), or None."""
    best = None
    for h in hist:
        if h[0] <= t:
            best = h
        else:
            break
    if best is None or t - best[0] > max_age:
        return None
    return best


# ---------------------------------------------------------------- names and the join

_DROP = {"esports", "esport", "gaming", "team", "clan", "club", "gg", "the", "org", "cs", "cs2", "csgo"}
# normalised market name -> normalised Liquipedia name (add pairs here when the spot check shows a miss)
ALIASES = {
    "nip": "ninjasinpyjamas", "navi": "natusvincere", "vp": "virtuspro", "col": "complexity",
    "ef": "eternalfire", "gl": "gamerlegion", "mongolz": "mongolz", "fal": "falcons", "tf": "falcons",
    "faze": "faze", "vit": "vitality", "liquid": "liquid", "c9": "cloud9", "mouz": "mouz", "mousesports": "mouz",
    "imp": "imperial", "bb": "betboom", "betboomteam": "betboom", "lv": "lynnvision", "tyl": "tyloo",
    "ra": "rareatom", "pv": "parivision", "nrg": "nrg", "g2": "g2", "vitalitybee": "vitalitybee",
    "themongolz": "mongolz", "wbt": "whitebit", "wbtacademy": "whitebitacademy", "1win": "1w", "ex1win": "ex1w",
    "furiafe": "furiafemale", "mibrfe": "mibrfemale", "imperialfe": "imperialfemale", "bigacademy": "bigacademy",
}


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\(.*?\)", " ", s).replace("&", " and ").replace("e-sports", "esports")
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t and t not in _DROP]
    n = "".join(toks) or re.sub(r"[^a-z0-9]", "", s)
    return ALIASES.get(n, n)


def sim(x: str, y: str) -> float:
    if x == y:
        return 1.0
    if min(len(x), len(y)) >= 4 and (x.startswith(y) or y.startswith(x)):
        return 0.9
    return SequenceMatcher(None, x, y).ratio()


@dataclass
class Preds:
    t: np.ndarray
    p: np.ndarray
    y: np.ndarray
    team1: np.ndarray
    team2: np.ndarray
    event: np.ndarray
    best_of: np.ndarray
    tier: np.ndarray


def load_preds(path: Path = PREDS, data: Path = DATA) -> Preds:
    if not path.exists():
        sys.exit(f"{path} missing: run `.venv/bin/python -m predict.compare base` first")
    d = dict(np.load(path))
    if "team1" not in d:
        # old cache without names: walk_forward keeps match order, so rows are the data file's matches from the
        # first prediction on; attach names only if times and outcomes agree row for row
        from .data import load_matches
        ms = [m for m in load_matches(data) if m.time >= d["t"].min()]
        if len(ms) != len(d["t"]) or any(m.time != t or float(m.t1_won) != y for m, t, y in zip(ms, d["t"], d["y"])):
            sys.exit(f"{path} has no team names and does not line up with {data.name}: delete it and rerun "
                     "`.venv/bin/python -m predict.compare base`")
        d.update(team1=np.array([m.team1_name for m in ms]), team2=np.array([m.team2_name for m in ms]),
                 event=np.array([m.event_id for m in ms]), best_of=np.array([m.best_of for m in ms]))
    tiers = {e["eventId"]: e.get("tier", "") for e in json.loads(data.read_text())["events"]}
    fix = {"S-Tier": "S", "A-Tier": "A", "B-Tier": "B", "C-Tier": "C", "C-tier": "C", "C": "C", "4": "D"}
    tier = np.array([fix.get(tiers.get(str(e), ""), "?") for e in d["event"]])
    return Preds(d["t"].astype(np.int64), d["p"], d["y"], d["team1"], d["team2"], d["event"], d["best_of"], tier)


def join(recs: list[Rec], P: Preds) -> list[dict]:
    """Best Liquipedia match within WINDOW for each market, both team names close; one market per (source, match)."""
    order = np.argsort(P.t)
    ts = P.t[order]
    n1 = [norm(x) for x in P.team1]
    n2 = [norm(x) for x in P.team2]
    best: dict[tuple, dict] = {}
    for r in recs:
        anchor = r.start or r.extra.get("end")
        if anchor is None:
            continue
        a, b = norm(r.a), norm(r.b)
        lo, hi = np.searchsorted(ts, anchor - WINDOW), np.searchsorted(ts, anchor + WINDOW, side="right")
        cand = None
        for j in order[lo:hi]:
            for flip, (x, y) in enumerate(((a, b), (b, a))):
                s1, s2 = sim(x, n1[j]), sim(y, n2[j])
                if min(s1, s2) < 0.8 or max(s1, s2) < 1.0:     # one name exact, the other at least close
                    continue
                score = (s1 + s2, -abs(int(P.t[j]) - anchor))
                if cand is None or score > cand["score"]:
                    cand = {"score": score, "j": int(j), "a_is_team1": flip == 0, "exact": s1 == s2 == 1.0}
        if cand is None:
            continue
        k = (r.source, cand["j"])
        if k not in best or cand["score"] > best[k]["score"]:
            best[k] = {**cand, "rec": r}
    return list(best.values())


def build(http: Http, recs: list[Rec], P: Preds, offsets: list[int]) -> list[dict]:
    """Joined rows with the market price for team1 at each offset (minutes before ref; negative = after)."""
    rows = []
    joined = join(recs, P)
    for i, m in enumerate(sorted(joined, key=lambda m: P.t[m["j"]])):
        r, j = m["rec"], m["j"]
        tl = int(P.t[j])
        ref = min(r.start, tl) if r.start else tl
        if r.source == "polymarket":
            h = poly_history(http, r.key, ref)
        else:
            h = kalshi_history(http, r, ref)
        if (i + 1) % 200 == 0:
            print(f"  prices {i + 1}/{len(joined)} ({http.requests} requests)", file=sys.stderr)
        if h is None:
            rows.append({"m": m, "ref": ref, "hist": None})
            continue
        pre = [x[1] for x in h if x[0] <= ref]
        flat = not pre or max(pre) - min(pre) <= 0.01     # never moved more than a tick or two: not a real quote
        px = {}
        for o in offsets:
            q = price_at(h, ref - 60 * o)
            if q is not None:
                pa = q[1]
                px[o] = {"p": pa if m["a_is_team1"] else 1 - pa, "spread": q[2] if len(q) > 2 else None, "age": ref - 60 * o - q[0]}
        rows.append({"m": m, "ref": ref, "hist": len(h), "flat": flat, "px": px, "n_pre": len(pre)})
    return rows


# ---------------------------------------------------------------- metrics

def _ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _pm(d):
    return f"{d.mean():+.4f} ± {d.std(ddof=1) / math.sqrt(len(d)):.4f}" if len(d) > 1 else "n/a"


def scores(pm, pk, y, label, width=22):
    """One table row: n, market LL/Brier/acc, model LL/Brier/acc, model - market paired differences."""
    d_ll = _ll(pk, y) - _ll(pm, y)
    d_br = (pk - y) ** 2 - (pm - y) ** 2
    return (f"{label:{width}s} {len(y):6d}  {_ll(pm, y).mean():.4f} {((pm - y) ** 2).mean():.4f} {((pm > .5) == y).mean():.3f}"
            f"   {_ll(pk, y).mean():.4f} {((pk - y) ** 2).mean():.4f} {((pk > .5) == y).mean():.3f}   {_pm(d_ll):>17s}  {_pm(d_br):>17s}")


HEADER = (f"{'':22s} {'n':>6s}  {'market LL  Brier  acc':22s}   {'model LL  Brier  acc':22s}   "
          f"{'LL model-market':>17s}  {'Brier model-market':>17s}")


def fit_logistic(X, y, l2=1e-6, iters=50):
    """Newton-Raphson logistic regression (intercept in X); returns (coef, standard errors)."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p) - l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ w))
    H = (X * (p * (1 - p))[:, None]).T @ X
    return w, np.sqrt(np.diag(np.linalg.inv(H)))


def combo(t, pm, pk, y, out):
    zm, zk = _logit(pm), _logit(pk)
    X = np.column_stack([np.ones_like(zm), zk, zm])
    w, se = fit_logistic(X, y)
    out.append(f"  in-sample fit (all n={len(y)}): logit p = {w[0]:+.3f} (±{se[0]:.3f}) + {w[1]:.3f} (±{se[1]:.3f}) z_model"
               f" + {w[2]:.3f} (±{se[2]:.3f}) z_market")
    o = np.argsort(t, kind="stable")
    half = o[len(o) // 2:]
    tr = o[: len(o) // 2]
    w2, se2 = fit_logistic(X[tr], y[tr])
    pc = 1 / (1 + np.exp(-X[half] @ w2))
    cut = dt.datetime.fromtimestamp(int(t[half[0]]), dt.timezone.utc).date()
    out.append(f"  time split at {cut}: train n={len(tr)} coef b0={w2[0]:+.3f} a(model)={w2[1]:.3f}±{se2[1]:.3f} "
               f"b(market)={w2[2]:.3f}±{se2[2]:.3f}; test n={len(half)}")
    for name, q in (("market", pm[half]), ("model", pk[half]), ("combination", pc)):
        d = _ll(q, y[half]) - _ll(pm[half], y[half])
        out.append(f"    test LL {name:12s} {_ll(q, y[half]).mean():.4f}   minus market {_pm(d)}")
    # expanding window, refit at each month start on everything before it (>= 300 rows)
    months = np.array([dt.datetime.fromtimestamp(int(x), dt.timezone.utc).strftime("%Y-%m") for x in t])
    pe = np.full(len(y), np.nan)
    for mo in sorted(set(months)):
        tr = t < t[months == mo].min()
        if tr.sum() < 300:
            continue
        we, _ = fit_logistic(X[tr], y[tr])
        pe[months == mo] = 1 / (1 + np.exp(-X[months == mo] @ we))
    k = ~np.isnan(pe)
    if k.sum():
        d = _ll(pe[k], y[k]) - _ll(pm[k], y[k])
        out.append(f"  expanding monthly refit: n={k.sum()}  combination {_ll(pe[k], y[k]).mean():.4f} vs market "
                   f"{_ll(pm[k], y[k]).mean():.4f}   combination - market {_pm(d)}")


def bets(pm, pk, y, ask_up, ask_dn, fee_rate, thresholds, out, rng):
    """Flat 1-unit stake on the side the model likes when |p_model - p_market| > threshold.
    Gross: filled at the market mid. Net: filled at the ask (mid + half-spread) plus a per-share taker fee of
    fee_rate * q * (1 - q) (Kalshi's published formula with rate 0.07; used for Polymarket with its feeSchedule rate)."""
    out.append(f"  {'thresh':>6s} {'bets':>5s} {'win%':>5s} {'avg px':>6s}  {'ROI gross (95% CI)':>26s}  {'ROI net (95% CI)':>26s}")
    for th in thresholds:
        up, dn = pk - pm > th, pm - pk > th
        sel = up | dn
        if sel.sum() < 5:
            out.append(f"  {th:6.2f} {sel.sum():5d}")
            continue
        won = np.where(up, y, 1 - y)[sel]
        px = np.where(up, pm, 1 - pm)[sel]
        pnet = np.clip(np.where(up, ask_up, ask_dn)[sel], 0.01, 0.99)
        pnet = np.clip(pnet + fee_rate[sel] * pnet * (1 - pnet), 0.01, 0.999)
        res = []
        for q in (px, pnet):
            prof = won / q - 1
            boot = np.array([prof[rng.integers(0, len(prof), len(prof))].mean() for _ in range(2000)])
            res.append(f"{prof.mean():+7.1%} [{np.percentile(boot, 2.5):+6.1%}, {np.percentile(boot, 97.5):+6.1%}]")
        out.append(f"  {th:6.2f} {sel.sum():5d} {won.mean():5.1%} {px.mean():6.3f}  {res[0]:>26s}  {res[1]:>26s}")


# ---------------------------------------------------------------- report

def _date(x):
    return dt.datetime.fromtimestamp(int(x), dt.timezone.utc).strftime("%Y-%m-%d %H:%M")


def table(rows, P, offset):
    """Usable rows at `offset` as arrays: priced, not flat, outcome agrees with Liquipedia."""
    keep = [r for r in rows if r.get("hist") and not r["flat"] and offset in r["px"]]
    agree = []
    for r in keep:
        m = r["m"]
        t1_won = m["rec"].a_won if m["a_is_team1"] else not m["rec"].a_won
        agree.append(t1_won == bool(P.y[m["j"]]))
    agree = np.array(agree, bool)
    keep = [r for r, a in zip(keep, agree) if a]
    j = np.array([r["m"]["j"] for r in keep], int)
    A = {
        "rows": keep, "j": j, "t": P.t[j], "y": P.y[j], "pk": P.p[j], "tier": P.tier[j],
        "pm": np.array([r["px"][offset]["p"] for r in keep]),
        "spread": np.array([r["px"][offset]["spread"] if r["px"][offset]["spread"] is not None else np.nan for r in keep]),
        "vol": np.array([r["m"]["rec"].volume for r in keep]),
        "fee": np.array([r["m"]["rec"].extra.get("fee_rate", 0.07 if r["m"]["rec"].source == "kalshi" else 0) for r in keep]),
        "kind": np.array([r["m"]["rec"].kind for r in keep]),
    }
    return A, int((~agree).sum())


def report(recs, rows, P, offset, offsets, spot, rng, half_spread, liquid):
    out = []
    ts0, ts1 = int(P.t.min()), int(P.t.max())
    for src in ("polymarket", "kalshi"):
        rs = [r for r in recs if r.source == src]
        R = [r for r in rows if r["m"]["rec"].source == src]
        if not rs:
            continue
        inr = [r for r in rs if ts0 - WINDOW <= (r.start or r.extra.get("end") or 0) <= ts1 + 3600]
        joined_ids = {r["m"]["rec"].id for r in R}
        anchors = [a for a in (r.start or r.extra.get("end") for r in rs) if a]
        out.append(f"\n=== {src}: {len(rs)} resolved series markets ({_date(min(anchors))[:10]} "
                   f"to {_date(max(anchors))[:10]}), {len(inr)} inside the Liquipedia prediction window")
        out.append(f"joined to a Liquipedia match: {len(R)} ({len(R) / max(1, len(inr)):.1%} of in-window markets; "
                   f"{sum(r['m']['exact'] for r in R)} with both names exact after normalisation)")
        nh = sum(1 for r in R if not r.get("hist"))
        fl = sum(1 for r in R if r.get("hist") and r["flat"])
        no = sum(1 for r in R if r.get("hist") and not r["flat"] and offset not in r["px"])
        A, bad = table(rows=R, P=P, offset=offset)
        out.append(f"dropped: {nh} no price history, {fl} flat (pre-start price range <= 0.01: never really quoted), {no} no price "
                   f"{offset} min before start (listed later), {bad} market outcome != Liquipedia outcome -> n = {len(A['y'])}")
        if len(A["y"]) < 20:
            continue
        pm, pk, y = A["pm"], A["pk"], A["y"]
        ext = ((pm < 0.03) | (pm > 0.97)).mean()
        dt_ref = np.array([(r["m"]["rec"].start - int(P.t[r["m"]["j"]])) / 3600 for r in A["rows"] if r["m"]["rec"].start])
        out.append(f"price {offset} min before ref = min(market start, Liquipedia time); share of prices outside [0.03, 0.97]: {ext:.1%}; "
                   f"market start - Liquipedia time: median {np.median(dt_ref) if len(dt_ref) else float('nan'):+.2f} h, "
                   f"|diff| > 1 h in {(np.abs(dt_ref) > 1).mean() if len(dt_ref) else 0:.1%}")
        if src == "kalshi":
            sp = A["spread"]
            out.append(f"Kalshi bid-ask spread at that minute (= overround of the yes/no pair): mean {np.nanmean(sp):.3f}, "
                       f"median {np.nanmedian(sp):.3f}")
        # offsets diagnostic on rows priced at every offset
        both = [r for r in R if r.get("hist") and not r["flat"] and all(o in r["px"] for o in offsets)]
        if both:
            yy = np.array([P.y[r["m"]["j"]] for r in both])
            cells = "  ".join(f"{-o:+d}m {_ll(np.array([r['px'][o]['p'] for r in both]), yy).mean():.4f}" for o in offsets)
            out.append(f"market log loss by minutes from ref (n={len(both)} priced at all offsets; a drop near 0 = in-play): {cells}")
        out.append("\n" + HEADER)
        out.append(scores(pm, pk, y, "all"))
        for w, lo, hi in (("tune window", TUNE, CONFIRM), ("confirm window", CONFIRM, "2100-01-01")):
            k = (A["t"] >= _iso(lo)) & (A["t"] < _iso(hi))
            if k.sum() >= 20:
                out.append(scores(pm[k], pk[k], y[k], w))
        for tier in "SABCD?":
            k = A["tier"] == tier
            if k.sum() >= 20:
                out.append(scores(pm[k], pk[k], y[k], f"tier {tier}"))
        quarter = np.array([f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in
                            (dt.datetime.fromtimestamp(int(x), dt.timezone.utc) for x in A["t"])])
        for qt in sorted(set(quarter)):
            k = quarter == qt
            if k.sum() >= 20:
                out.append(scores(pm[k], pk[k], y[k], qt))
        for lo, hi in ((0, 1e3), (1e3, 1e4), (1e4, 1e5), (1e5, 1e12)):
            k = (A["vol"] >= lo) & (A["vol"] < hi)
            if k.sum() >= 20:
                out.append(scores(pm[k], pk[k], y[k], f"volume {lo:.0e}-{hi:.0e}".replace("+0", "")))
        for kind in ("moneyline", "legacy"):
            k = A["kind"] == kind
            if src == "polymarket" and 20 <= k.sum() < len(k):
                out.append(scores(pm[k], pk[k], y[k], f"kind {kind}"))
        liq = A["vol"] >= liquid
        if liq.sum() >= 20:
            out.append(f"\nliquid subset, lifetime volume >= {liquid:,.0f} (volume includes in-play trading):")
            out.append(scores(pm[liq], pk[liq], y[liq], f"volume >= {liquid:.0e}".replace("+0", "")))
            for tier in "SABCD?":
                k = liq & (A["tier"] == tier)
                if k.sum() >= 20:
                    out.append(scores(pm[k], pk[k], y[k], f"  tier {tier}"))
        if src == "kalshi":
            sp = np.where(np.isnan(A["spread"]), 2 * half_spread, A["spread"])
            up, dn = pm + sp / 2, 1 - pm + sp / 2
            note = "net = at the recorded Kalshi ask + 0.07 q(1-q) fee"
        else:
            up, dn = pm + half_spread, 1 - pm + half_spread
            note = f"net = mid + {half_spread} assumed half-spread + feeSchedule rate x q(1-q) where fees were enabled"
        for label, k in (("all", np.ones(len(y), bool)), (f"volume >= {liquid:,.0f}", liq)):
            if k.sum() < 50:
                continue
            out.append(f"\n[{label}] combination logit p = b0 + a z_model + b z_market (z = logit):")
            combo(A["t"][k], pm[k], pk[k], y[k], out)
            out.append(f"[{label}] betting simulation, 1 unit per bet on the model's side ({note}):")
            bets(pm[k], pk[k], y[k], up[k], dn[k], A["fee"][k], (0.03, 0.05, 0.08, 0.10, 0.15, 0.20), out, rng)
        # spot check
        out.append(f"\nspot check ({spot} random joined rows): market teams | Liquipedia teams | market start | Liquipedia time | "
                   f"p_market(team1) | p_model | team1 won")
        for i in rng.choice(len(A["rows"]), min(spot, len(A["rows"])), replace=False):
            r = A["rows"][i]
            m, j = r["m"], r["m"]["j"]
            rec = m["rec"]
            out.append(f"  {rec.a} vs {rec.b} | {P.team1[j]} vs {P.team2[j]} | {_date(rec.start) if rec.start else 'n/a':16s} | "
                       f"{_date(P.t[j])} | {A['pm'][i]:.3f} | {A['pk'][i]:.3f} | {int(A['y'][i])}{'' if m['exact'] else '  (fuzzy)'}")
        miss = [r for r in inr if r.id not in joined_ids]
        near = _near(miss, P)
        out.append(f"\nunmatched in-window markets: {len(miss)}; for {near} of them neither team name appears in any Liquipedia "
                   f"match within ±36 h (the match or the team is missing from the export), sample:")
        for i in rng.choice(len(miss), min(spot, len(miss)), replace=False):
            r = miss[i]
            out.append(f"  {r.a} vs {r.b} ({norm(r.a)} / {norm(r.b)}) {_date(r.start or r.extra['end'])}  {r.title[:70]}")
    # both markets on the same matches
    Ap, _ = table([r for r in rows if r["m"]["rec"].source == "polymarket"], P, offset)
    Ak, _ = table([r for r in rows if r["m"]["rec"].source == "kalshi"], P, offset)
    common = sorted(set(Ap["j"]) & set(Ak["j"]))
    if len(common) >= 20:
        ip = {j: i for i, j in enumerate(Ap["j"])}
        ik = {j: i for i, j in enumerate(Ak["j"])}
        pp = np.array([Ap["pm"][ip[j]] for j in common])
        pkal = np.array([Ak["pm"][ik[j]] for j in common])
        yy = P.y[common]
        pk = P.p[common]
        out.append(f"\n=== matches priced by both markets (n={len(common)}): LL Polymarket {_ll(pp, yy).mean():.4f}, "
                   f"Kalshi {_ll(pkal, yy).mean():.4f}, model {_ll(pk, yy).mean():.4f}; mean |Poly - Kalshi| {np.abs(pp - pkal).mean():.3f}")
        out.append(HEADER)
        out.append(scores(pp, pk, yy, "vs Polymarket"))
        out.append(scores(pkal, pk, yy, "vs Kalshi"))
    return "\n".join(out)


def _near(miss, P):
    names = {}
    for j, (x, y) in enumerate(zip(P.team1, P.team2)):
        for n in (norm(x), norm(y)):
            names.setdefault(n, []).append(int(P.t[j]))
    c = 0
    for r in miss:
        anc = r.start or r.extra.get("end")
        hit = any(abs(t - anc) <= WINDOW for n in (norm(r.a), norm(r.b)) for t in names.get(n, []))
        c += not hit
    return c


def _iso(d):
    return int(dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc).timestamp())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preds", default=str(PREDS))
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--offset", type=int, default=60, help="minutes before the reference start (default 60)")
    ap.add_argument("--offsets", default="360,180,60,30,10,0,-30,-60", help="diagnostic offsets in minutes")
    ap.add_argument("--sources", default="polymarket,kalshi")
    ap.add_argument("--offline", action="store_true", help="cache only, no network")
    ap.add_argument("--relist", action="store_true", help="re-fetch the market listings (new resolutions)")
    ap.add_argument("--half-spread", type=float, default=0.01, help="assumed Polymarket half-spread for net betting ROI")
    ap.add_argument("--liquid", type=float, default=1e4, help="volume threshold for the liquid-market subset")
    ap.add_argument("--spot", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    http = Http(offline=args.offline, relist=args.relist)
    P = load_preds(Path(args.preds), Path(args.data))
    recs = []
    if "polymarket" in args.sources:
        recs += polymarket(http)
    if "kalshi" in args.sources:
        recs += kalshi(http)
    offsets = sorted({args.offset, *[int(x) for x in args.offsets.split(",")]}, reverse=True)
    rows = build(http, recs, P, offsets)
    print(f"({http.requests} network requests this run)", file=sys.stderr)
    print(f"model: {Path(args.preds).name}, {len(P.t)} walk-forward predictions {_date(P.t.min())[:10]} to {_date(P.t.max())[:10]}")
    print(report(recs, rows, P, args.offset, offsets, args.spot, np.random.default_rng(args.seed), args.half_spread, args.liquid))


if __name__ == "__main__":
    main()
