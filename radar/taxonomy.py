"""India-focused event taxonomy.

Each category has
  * a GDELT DOC 2.0 query (used for the 2017+ daily volume/tone backfill and live article lists)
  * keyword rules used to tag headlines from RSS feeds and GDELT article lists
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    gdelt_query: str
    keywords: tuple
    color: str  # UI accent


CATEGORIES: list[Category] = [
    Category(
        "monetary", "RBI & rates",
        '("Reserve Bank of India" OR RBI) (repo OR "interest rate" OR "monetary policy" OR inflation)',
        ("rbi", "reserve bank", "repo rate", "monetary policy", "mpc", "rate cut", "rate hike", "cpi inflation"),
        "#5B8DEF",
    ),
    Category(
        "fiscal", "Budget & fiscal",
        '("Union Budget" OR "fiscal deficit" OR "finance ministry") India',
        ("budget", "fiscal deficit", "finance minister", "gst", "tax", "disinvestment"),
        "#9B7BEA",
    ),
    Category(
        "flows", "Foreign flows",
        '(FII OR FPI OR "foreign investors" OR "foreign portfolio") India (outflow OR selling OR inflow OR buying)',
        ("fii", "fpi", "foreign investor", "foreign portfolio", "outflow", "inflow", "dii"),
        "#2BB3A3",
    ),
    Category(
        "regulation", "SEBI & regulation",
        'SEBI (ban OR probe OR order OR circular OR regulation OR penalty)',
        ("sebi", "regulator", "probe", "penalty", "circular", "f&o rules"),
        "#E0A43A",
    ),
    Category(
        "global_rates", "Fed & global rates",
        '("Federal Reserve" OR FOMC OR "Treasury yields") (rates OR inflation OR hike OR cut)',
        ("federal reserve", "fed ", "fomc", "treasury yield", "bond yield", "us inflation", "jobs report"),
        "#4FA3D9",
    ),
    Category(
        "commodities", "Crude & commodities",
        '("crude oil" OR Brent OR OPEC) (price OR prices OR supply)',
        ("crude", "brent", "opec", "oil price", "gold price"),
        "#C98A5B",
    ),
    Category(
        "geopolitics", "Geopolitics",
        'India (war OR attack OR border OR sanctions OR conflict OR tensions)',
        ("war", "attack", "border", "sanction", "conflict", "tension", "missile", "ceasefire", "tariff"),
        "#E05D5D",
    ),
    Category(
        "market_stress", "Market moves",
        '(Sensex OR Nifty) (crash OR plunge OR selloff OR tumble OR rally OR surge)',
        ("sensex", "nifty", "crash", "plunge", "sell-off", "selloff", "tumble", "rout", "rally", "surge", "bloodbath"),
        "#D9534F",
    ),
    Category(
        "corporate", "Corporate & earnings",
        'India (earnings OR "quarterly results" OR profit) (bank OR IT OR Reliance OR HDFC OR Infosys OR TCS)',
        ("q1", "q2", "q3", "q4", "earnings", "results", "net profit", "revenue", "guidance", "downgrade", "upgrade"),
        "#7FB069",
    ),
]

CAT_BY_KEY = {c.key: c for c in CATEGORIES}

# Compact finance lexicon (Loughran–McDonald style) for headline tone.
NEG = {
    "crash", "crashes", "plunge", "plunges", "plunged", "tumble", "tumbles", "slump", "slumps", "fall", "falls",
    "fell", "drop", "drops", "dropped", "decline", "declines", "loss", "losses", "selloff", "sell-off", "rout",
    "fear", "fears", "worry", "worries", "concern", "concerns", "risk", "risks", "war", "attack", "sanction",
    "sanctions", "downgrade", "downgrades", "default", "probe", "penalty", "ban", "outflow", "outflows", "weak",
    "weaker", "slowdown", "recession", "inflation", "hike", "hikes", "crisis", "volatile", "volatility", "panic",
    "bloodbath", "lowest", "slide", "slides", "sinks", "sink", "tension", "tensions", "conflict", "tariff", "tariffs",
    "miss", "misses", "cut", "warns", "warning", "fraud", "selling",
}
POS = {
    "rally", "rallies", "surge", "surges", "surged", "gain", "gains", "rise", "rises", "rose", "jump", "jumps",
    "record", "high", "highs", "strong", "stronger", "beat", "beats", "upgrade", "upgrades", "inflow", "inflows",
    "growth", "boost", "boosts", "recovery", "rebound", "rebounds", "optimism", "buying", "eases", "ease",
    "ceasefire", "relief", "positive", "outperform", "profit", "profits",
}
NEGATORS = {"not", "no", "never", "without", "despite"}
_TOKEN = re.compile(r"[a-z][a-z\-&]+")


def headline_tone(text: str) -> float:
    """Return tone in [-1, 1] for a headline. Lexicon based with simple negation."""
    toks = _TOKEN.findall(text.lower())
    score, n = 0.0, 0
    for i, t in enumerate(toks):
        s = 1.0 if t in POS else -1.0 if t in NEG else 0.0
        if s == 0.0:
            continue
        if i > 0 and toks[i - 1] in NEGATORS:
            s = -s
        score += s
        n += 1
    return 0.0 if n == 0 else max(-1.0, min(1.0, score / max(n, 1.5)))


def tag_headline(text: str) -> list[str]:
    t = f" {text.lower()} "
    hits = [c.key for c in CATEGORIES if any(k in t for k in c.keywords)]
    return hits or ["other"]
