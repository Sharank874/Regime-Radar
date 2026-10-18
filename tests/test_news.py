"""News parsing and tagging (network calls mocked)."""
from datetime import datetime

import pandas as pd

from radar import news as N
from radar.taxonomy import headline_tone, tag_headline


def test_tagging_and_tone():
    assert "monetary" in tag_headline("RBI holds repo rate steady")
    assert "flows" in tag_headline("FPIs pull out Rs 5,000 crore")
    assert headline_tone("Sensex crashes as FII selling intensifies") < 0
    assert headline_tone("Nifty rallies to record high") > 0
    assert headline_tone("Markets not worried despite weak data") != headline_tone("Markets worried about weak data")


def test_timeline_parsing(monkeypatch):
    payload = {"timeline": [{"series": "Article Count", "data": [
        {"date": "20240102T000000Z", "value": 10, "norm": 1000},
        {"date": "20240102T001500Z", "value": 5, "norm": 900},
        {"date": "20240103T000000Z", "value": 7, "norm": 800}]}]}
    monkeypatch.setattr(N, "_get", lambda params, retries=3: payload)
    df = N._timeline("q", "timelinevolraw", datetime(2024, 1, 1), datetime(2024, 1, 4))
    assert df.loc[pd.Timestamp("2024-01-02"), "value"] == 15      # 15-min bins summed per day
    assert df.loc[pd.Timestamp("2024-01-02"), "norm"] == 1900


def test_article_list_parsing(monkeypatch):
    payload = {"articles": [{"title": "RBI keeps repo rate unchanged", "url": "https://x", "domain": "x.com",
                             "seendate": "20261016T083000Z"}]}
    monkeypatch.setattr(N, "_get", lambda params, retries=3: payload)
    monkeypatch.setattr(N.time, "sleep", lambda s: None)
    items = N.gdelt_articles(24)
    assert items and items[0]["published_utc"].startswith("2026-10-16T08:30")
    assert all(i["categories"] for i in items)
