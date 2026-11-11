"""Score every post and align daily sentiment with the price bars.

For each (ticker, trading day) we compute:
  mentions            - how many posts mentioned the stock
  sentiment_mean      - average sentiment score of those posts
  sentiment_weighted  - same, but louder accounts (followers/views) count more
  news_sentiment      - average sentiment of that day's news

These become the model's features, and the daily alignment is what lets us
compare "what people said on day t" with "what the price did on day t+1".
"""

import numpy as np
import pandas as pd

import sentiment


def score_posts(posts: pd.DataFrame) -> pd.DataFrame:
    if posts.empty:
        return posts
    posts = posts.copy()
    posts["sentiment"] = posts["text"].map(sentiment.score_text)
    posts["date"] = pd.to_datetime(posts["time_utc"], errors="coerce", utc=True).dt.date
    return posts


def daily_features(posts: pd.DataFrame) -> pd.DataFrame:
    if posts.empty:
        return pd.DataFrame(columns=["ticker", "date", "mentions",
                                     "sentiment_mean", "sentiment_weighted",
                                     "engagement"])
    posts = posts.dropna(subset=["date"])
    rows = []
    for (sym, day), g in posts.groupby(["ticker", "date"]):
        weights = np.log1p(g["views"].fillna(0).astype(float)
                           + g["likes"].fillna(0).astype(float) * 5
                           + g["followers"].fillna(0).astype(float)) + 1.0
        wmean = float(np.average(g["sentiment"], weights=weights))
        rows.append({
            "ticker": sym,
            "date": day,
            "mentions": len(g),
            "sentiment_mean": round(float(g["sentiment"].mean()), 4),
            "sentiment_weighted": round(wmean, 4),
            "engagement": float(g["views"].fillna(0).astype(float).sum()
                                + g["likes"].fillna(0).astype(float).sum()),
        })
    return pd.DataFrame(rows)


def merge_with_prices(daily: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Join sentiment features onto price bars (per ticker and date)."""
    px = prices.copy()
    px["ret_1d"] = px.groupby("ticker")["close"].pct_change()
    px["close_next"] = px.groupby("ticker")["close"].shift(-1)  # target: n+1
    if daily.empty:
        daily = pd.DataFrame(columns=["ticker", "date", "mentions",
                                      "sentiment_mean", "sentiment_weighted",
                                      "engagement"])
    merged = px.merge(daily, on=["ticker", "date"], how="left")
    for col, val in [("mentions", 0), ("sentiment_mean", 0.0),
                     ("sentiment_weighted", 0.0), ("engagement", 0.0)]:
        merged[col] = merged[col].fillna(val)
    return merged
