"""Country -> region lookup, read from Valve's model/util/region.js so the two stay in sync.

Valve collapses its table to three buckets (Europe / Americas / Asia). For a rating prior
that is too coarse: CIS and Western Europe rarely meet outside tier-1, and NA and SA are
separate online scenes. So we keep Valve's fine labels and additionally split CIS out of EU.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

_JS = Path(__file__).resolve().parent.parent / "model" / "util" / "region.js"
CIS = {"ru", "ua", "by", "kz", "kg", "uz", "tj", "tm", "md"}
UNKNOWN = "??"


def _load() -> dict[str, str]:
    table = {}
    for cc, reg in re.findall(r"countrycode\s*:\s*'(\w+)'\s*,\s*region\s*:\s*'(\w+)'", _JS.read_text()):
        table[cc.lower()] = reg
    for cc in CIS:
        table[cc] = "CIS"
    return table


COUNTRY_REGION = _load()
REGIONS = sorted(set(COUNTRY_REGION.values()))


def country_region(cc: str | None) -> str:
    if not cc:
        return UNKNOWN
    return COUNTRY_REGION.get(cc.lower(), UNKNOWN)


def team_region(countries) -> str:
    """Plurality region of a lineup's player countries; unknown if nobody has a country."""
    c = Counter(country_region(x) for x in countries if x and x.lower() != "world")
    if not c:
        return UNKNOWN
    return c.most_common(1)[0][0]
