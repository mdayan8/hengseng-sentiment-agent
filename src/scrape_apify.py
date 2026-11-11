"""Scrape social posts with Apify (sentiment source 1 and 2).

Source 1: LIHKG   - Hong Kong's biggest retail-investor forum.
Source 2: Telegram - public HK channels, keyword search (HK region bias).

Both are scraped through ready-made Apify actors. The actor IDs live in
config.py, and we print the actor IDs we used because the employer asked
for them in the submission email (so they can reproduce the run).

Every post is normalised to one simple shape:
    {source, ticker, time_utc, text, likes, reposts, views, followers, author}

Two honest data-quality steps happen here, before anything reaches the model:
  1. Keyword search on Telegram matches a lot of irrelevant channels (e.g. a
     drama-upload channel whose episodes are produced by Tencent Video). A
     post is only kept if its text actually mentions the company AND talks
     in a finance context (see FIN_TERMS below).
  2. Duplicate texts (re-uploaded episodes, reposted market wraps) are
     dropped so one loud channel cannot dominate the sentiment score.
"""

import os
from datetime import datetime, timezone, timedelta

import pandas as pd
from apify_client import ApifyClient

import config

# Finance-context terms (English + Chinese). A post must mention the company
# AND at least one of these, otherwise it is noise from keyword matching.
FIN_TERMS = [
    "stock", "share", "shares", "buy", "sell", "bull", "bear", "target",
    "price", "earnings", "dividend", "market", "hang seng", "hsi",
    "valuation", "rally", "crash", "investor", "hk$", "hkd",
    "股", "買", "賣", "升", "跌", "牛市", "熊市", "恒指", "恆指", "股息", "成交",
]


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
    author = first("author", "username", "user", "nickname", default="")
    followers = first("followers", "follower_count", "authorFollowers", default=0) or 0
    if source == "telegram":
        # Telegram keyword search: the channel is the useful "author", and
        # its subscriber count sits in a nested channel_info dict.
        author = item.get("source_name") or author
        info = item.get("channel_info") or {}
        if isinstance(info, dict) and info.get("subscribers") is not None:
            followers = info["subscribers"]
    return {
        "source": source,
        "ticker": ticker,
        "time_utc": str(ts),
        "text": str(text)[:2000],
        "likes": first("likes", "like_count", "likeCount", default=0) or 0,
        "reposts": first("reposts", "shares", "shareCount", default=0) or 0,
        "views": first("views", "view_count", "viewCount", "pageview", default=0) or 0,
        "followers": followers,
        "author": str(author),
    }


def _run_actor(client, actor_id: str, run_input: dict) -> list:
    """Run one Apify actor and return its dataset items."""
    print(f"  -> running Apify actor: {actor_id}")
    run = client.actor(actor_id).call(run_input=run_input)
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    print(f"     got {len(items)} raw items")
    return items


def _is_relevant(text: str, ticker_cfg: dict) -> bool:
    """Keep a post only if it mentions the company in a finance context."""
    t = text.lower()
    company_pats = ([ticker_cfg["name"].lower(),
                     ticker_cfg["ticker"].split(".")[0],
                     ticker_cfg["cashtag"].lower()]
                    + [k.lower() for k in ticker_cfg["keywords"]])
    return any(p in t for p in company_pats) and any(f in t for f in FIN_TERMS)


def scrape_lihkg(tickers=None, days: int = config.SOCIAL_DAYS) -> pd.DataFrame:
    """Search LIHKG for each ticker keyword, keep the last `days` days."""
    client = _client()
    tickers = tickers or config.TICKERS
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = []
    for t in tickers:
        try:
            items = _run_actor(client, config.APIFY_LIHKG_ACTOR, {
                "mode": "search",            # keyword search across thread titles/replies
                "keywords": t["keywords"],
                "dateFrom": "2w",            # actor-side window; we re-filter to `days` below
                "order": "desc_reply_time",  # freshest discussion first
                "maxItems": 50,
                "maxReplies": 10,            # a few replies: that is where sentiment lives
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
    """Keyword-search public Telegram channels for each stock (HK bias)."""
    client = _client()
    tickers = tickers or config.TICKERS
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = []
    for t in tickers:
        code = t["ticker"].split(".")[0]  # '0700.HK' -> '0700'
        try:
            items = _run_actor(client, config.APIFY_TELEGRAM_ACTOR, {
                "mode": "keyword",
                "keywords": [t["name"], t["keywords"][2], f"{code} HK"],
                "afterDate": cutoff.date().isoformat(),
                "maxResultsPerKeyword": 50,
                "countryCode": "hk",       # bias search toward HK channels
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
    """Run both sources, filter for real finance chatter, stack the results."""
    parts = []
    print("Source 1: LIHKG")
    parts.append(scrape_lihkg())
    print("Source 2: Telegram")
    parts.append(scrape_telegram())
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    before = len(df)
    df = df[df.apply(lambda r: _is_relevant(str(r["text"]),
                     next(t for t in config.TICKERS if t["ticker"] == r["ticker"])), axis=1)]
    print(f"Relevance filter: kept {len(df)}/{before} posts that mention the "
          f"company in a finance context")
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Dedupe: kept {len(df)}/{before} unique posts")
    print(f"Total social posts collected: {len(df)}")
    return df
