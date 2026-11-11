"""Pull recent news per asset with exa.com (complements social sentiment)."""

import os
from datetime import datetime, timezone, timedelta

import pandas as pd

import config


def fetch_news(tickers=None, days: int = config.SOCIAL_DAYS) -> pd.DataFrame:
    key = os.environ.get("EXA_API_KEY")
    if not key:
        raise RuntimeError("EXA_API_KEY is not set. Put it in your .env file.")
    from exa_py import Exa
    exa = Exa(api_key=key)
    tickers = tickers or config.TICKERS
    start = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    rows = []
    for t in tickers:
        query = f"{t['name']} stock {t['ticker'].split('.')[0]} Hong Kong"
        print(f"  -> Exa search: {query}")
        try:
            res = exa.search_and_contents(
                query,
                type="auto",
                category="news",
                start_published_date=start,
                num_results=10,
                text={"max_characters": 1000},
            )
        except Exception as e:
            print(f"     Exa failed for {t['ticker']}: {e}")
            continue
        for r in res.results:
            rows.append({
                "source": "exa_news",
                "ticker": t["ticker"],
                "asset_name": t["name"],
                "time_utc": str(getattr(r, "published_date", "") or ""),
                "text": f"{getattr(r, 'title', '')} {(getattr(r, 'text', '') or '')[:800]}",
                "url": getattr(r, "url", ""),
                "likes": 0, "reposts": 0, "views": 0, "followers": 0,
                "author": "",
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce", utc=True)
    print(f"Total news items collected: {len(df)}")
    return df
