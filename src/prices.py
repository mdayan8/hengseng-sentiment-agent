"""Download daily price bars with yfinance and compute per-ticker stats.

No API key needed. We pull ~90 days of bars so the model has history to
learn from, and we compute simple support/resistance levels from the last
20 trading days (recent low / recent high).
"""

from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf

import config


def download_prices(tickers=None, days: int = config.PRICE_HISTORY_DAYS) -> pd.DataFrame:
    tickers = tickers or config.TICKERS
    symbols = [t["ticker"] for t in tickers]
    start = (datetime.now(timezone.utc) - timedelta(days=days)).date()
    print(f"Downloading price bars for {symbols} since {start}")
    raw = yf.download(symbols, start=start, interval="1d",
                      group_by="ticker", auto_adjust=True, progress=False)
    frames = []
    for t in tickers:
        sym = t["ticker"]
        try:
            df = raw[sym].dropna(how="all").reset_index()
        except KeyError:
            print(f"  no price data for {sym}")
            continue
        df = df.rename(columns=str.lower)
        df["ticker"] = sym
        df["asset_name"] = t["name"]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        frames.append(df[["date", "ticker", "asset_name", "open", "high",
                          "low", "close", "volume"]])
    out = pd.concat(frames, ignore_index=True)
    print(f"Downloaded {len(out)} price bars")
    return out


def ticker_snapshot(prices: pd.DataFrame) -> pd.DataFrame:
    """Current price + 20-day support/resistance per ticker."""
    rows = []
    for sym, g in prices.groupby("ticker"):
        g = g.sort_values("date")
        last = g.iloc[-1]
        recent = g.tail(20)
        rows.append({
            "ticker": sym,
            "asset_name": last["asset_name"],
            "current_price": round(float(last["close"]), 2),
            "support": round(float(recent["low"].min()), 2),
            "resistance": round(float(recent["high"].max()), 2),
        })
    return pd.DataFrame(rows)
