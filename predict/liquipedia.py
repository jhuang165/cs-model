"""Polite, cached client for Liquipedia's Counter-Strike MediaWiki API.

Follows https://liquipedia.net/api-terms-of-use: API endpoint only, at most one request per 2 seconds
(this client waits 2.5 s; no action=parse), gzip, one reused connection, a User-Agent with contact info,
and every response cached on disk so nothing is ever requested twice. Liquipedia content is CC-BY-SA 3.0:
credit Liquipedia as the source of anything derived from it.

The contact defaults to `git config user.email`; override with LIQUIPEDIA_CONTACT or --contact.
"""
from __future__ import annotations

import gzip
import hashlib
import http.client
import json
import os
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "liquipedia" / "cache"
HOST = "liquipedia.net"
PATH = "/counterstrike/api.php"
MIN_INTERVAL = 2.5


def default_contact() -> str:
    c = os.environ.get("LIQUIPEDIA_CONTACT")
    if c:
        return c
    try:
        return subprocess.run(["git", "config", "user.email"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except OSError:
        return ""


class Client:
    def __init__(self, contact: str | None = None, cache: Path = CACHE, offline: bool = False):
        contact = contact or default_contact()
        if not contact and not offline:
            sys.exit("Liquipedia requires contact info in the User-Agent: set LIQUIPEDIA_CONTACT or pass --contact")
        self.ua = f"cs-model-research/0.1 (CS2 match prediction experiments; {contact})"
        self.cache, self.offline = cache, offline
        self.cache.mkdir(parents=True, exist_ok=True)
        self.conn: http.client.HTTPSConnection | None = None
        self.last = 0.0
        self.requests = 0

    def _key(self, params: dict) -> Path:
        s = json.dumps(sorted(params.items()), ensure_ascii=False)
        return self.cache / (hashlib.sha1(s.encode()).hexdigest() + ".json")

    def get(self, _interval: float = MIN_INTERVAL, **params) -> dict:
        """One API call, served from the disk cache when possible; uncached calls are spaced at
        least `_interval` seconds apart (use 30 s for heavy actions such as expandtemplates)."""
        params = {"format": "json", "formatversion": "2", **params}
        path = self._key(params)
        if path.exists():
            return json.loads(path.read_text())
        if self.offline:
            raise KeyError(f"not cached: {params}")
        url = PATH + "?" + urllib.parse.urlencode(params)
        for attempt in range(5):
            wait = self.last + max(_interval, MIN_INTERVAL) - time.time()
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                if self.conn is None:
                    self.conn = http.client.HTTPSConnection(HOST, timeout=60)
                self.conn.request("GET", url, headers={"User-Agent": self.ua, "Accept-Encoding": "gzip"})
                r = self.conn.getresponse()
                body = r.read()
            except (http.client.HTTPException, OSError) as e:
                print(f"  connection error {e!r}, retrying", file=sys.stderr)
                self.conn = None
                time.sleep(10 * (attempt + 1))
                continue
            self.requests += 1
            if r.status == 429 or r.status >= 500:
                print(f"  HTTP {r.status}, backing off", file=sys.stderr)
                time.sleep(60 * (attempt + 1))
                continue
            if r.status != 200:
                sys.exit(f"HTTP {r.status} from Liquipedia for {url}: {body[:300]!r}")
            if r.getheader("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            data = json.loads(body)
            if "error" in data:
                sys.exit(f"Liquipedia API error for {url}: {data['error']}")
            path.write_text(json.dumps(data))
            return data
        raise RuntimeError(f"gave up on {url}")

    def wikitext(self, titles: list[str]) -> dict[str, str]:
        """Current wikitext of up to 50 pages per request, following redirects; title -> text."""
        out = {}
        for i in range(0, len(titles), 50):
            batch = sorted(titles[i:i + 50])
            d = self.get(action="query", prop="revisions", rvprop="content", rvslots="main",
                         redirects="1", titles="|".join(batch))
            for p in d["query"].get("pages", []):
                if "revisions" in p:
                    out[p["title"]] = p["revisions"][0]["slots"]["main"]["content"]
        return out
