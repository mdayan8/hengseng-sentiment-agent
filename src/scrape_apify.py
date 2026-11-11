"""Scrape social posts with Apify (sentiment source 1 and 2).

Source 1: LIHKG   - Hong Kong's biggest retail-investor forum.
Source 2: Telegram - public HK trading channels, keyword search.

Both are scraped through ready-made Apify actors. The actor IDs live in
config.py, and we print the actor IDs we used because the employer asked
for them in the submission email (so they can reproduce the run).

Every post is normalised to one simple shape:
    {source, ticker, time_utc, text, likes, reposts, views, followers, author}
"""

import os
from datetime import datetime, timezone, timedelta

import pandas as pd
from apify_client import ApifyClient

import config


def _client() -> ApifyClient:
    token = os.environ.get("APIFY_TOKEN")
    if not token:
        raise RuntimeError("APIFY_TOKEN is not set. Put it in your .env file.")
    return ApifyClient(token)


def _normalise(item: dict, source: str, ticker: str) -> dict:
    """Map whatever an actor returns onto our common post shape.

    Actor outputs differ a little, so we check a few likely field names.
    """
    def first(*names, default=None):
        for n in names:
            if item.get(n) not in (None, ""):
                return item[n]
        return default

    ts = first("time", "timestamp", "created_at", "createdAt", "date",
               "published_at", "postTime", default="")
    text = first("text", "content", "title", "message", "body", default="")
    if first("title") and first("content"):
        text = f"{item['title']} {item['content']}"
    return {
        "source": source,
        "ticker": ticker,
        "time_utc": str(ts),
        "text": str(text)[:2000],
        "likes": first("likes", "like_count", "likeCount", default=0) or 0,
        "reposts": first("reposts", "shares", "shareCount", default=0) or 0,
        "views": first("views", "view_count", "viewCount", "pageview", default=0) or 0,
        "followers": first("followers", "follower_count", "authorFollowers", default=0) or 0,
        "author": str(first("author", "username", "user", "nickname", default="")),
    }


def _run_actor(client, actor_id: str, run_input: dict) -> list:
    """Run one Apify actor and return its dataset items."""
    print(f"  -> running Apify actor: {actor_id}")
    run = client.actor(actor_id).call(run_input=run_input)
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    print(f"     got {len(items)} raw items")
    return items


def scrape_lihkg(tickers=None, days: int = config.SOCIAL_DAYS) -> pd.DataFrame:
    """Search LIHKG for each ticker keyword, keep the last `days` days."""
    client = _client()
    tickers = tickers or config.TICKERS
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = []
    for t in tickers:
        try:
            items = _run_actor(client, config.APIFY_LIHKG_ACTOR, {
                "keywords": t["keywords"],
                "dateFrom": cutoff.date().isoformat(),
                "maxItems": 50,
                "maxReplies": 20,   # include some replies: that is where sentiment lives
            })
        except Exception as e:  # actor input schemas vary; fail soft, keep going
            print(f"     LIHKG actor failed for {t['ticker']}: {e}")
            continue
        for it in items:
            row = _normalise(it, "lihkg", t["ticker"])
            row["asset_name"] = t["name"]
            rows.append(row)
    df = pd.DataFrame(rows)
    return _filter_recent(df, cutoff)


def scrape_telegram(tickers=None, days: int = config.SOCIAL_DAYS) -> pd.DataFrame:
    """Keyword-search public Telegram channels for each stock."""
    client = _client()
    tickers = tickers or config.TICKERS
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = []
    for t in tickers:
        code = t["ticker"].split(".")[0]  # '0700.HK' -> '0700'
        try:
            items = _run_actor(client, config.APIFY_TELEGRAM_ACTOR, {
                "mode": "keyword",
                "keywords": [f"{code} HK stock", f"{t['name']} stock"],
                "afterDate": cutoff.date().isoformat(),
                "maxResultsPerKeyword": 50,
            })
        except Exception as e:
            print(f"     Telegram actor failed for {t['ticker']}: {e}")
            continue
        for it in items:
            row = _normalise(it, "telegram", t["ticker"])
            row["asset_name"] = t["name"]
            rows.append(row)
    df = pd.DataFrame(rows)
    return _filter_recent(df, cutoff)


def _filter_recent(df: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
    """Parse timestamps to UTC and keep posts from the last N days."""
    if df.empty:
        return df
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce", utc=True)
    before = len(df)
    df = df[(df["time_utc"].isna()) | (df["time_utc"] >= cutoff)]
    print(f"  kept {len(df)}/{before} posts inside the {config.SOCIAL_DAYS}-day window")
    return df.reset_index(drop=True)


def scrape_all() -> pd.DataFrame:
    """Run both sources and stack the results."""
    parts = []
    print("Source 1: LIHKG")
    parts.append(scrape_lihkg())
    print("Source 2: Telegram")
    parts.append(scrape_telegram())
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    print(f"Total social posts collected: {len(df)}")
    return df
