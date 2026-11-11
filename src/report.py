"""Build the visual report: charts + one HTML page.

The report sorts tickers by social mentions and sentiment (as the brief
asks) and shows, per ticker: current price, sentiment + direction,
support/resistance, and the n+1 prediction with out-of-sample metrics.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import sentiment as sent


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def chart_mentions(summary: pd.DataFrame) -> str:
    s = summary.sort_values("mentions", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(s["asset_name"], s["mentions"], color="#2563eb")
    ax.set_title("Social mentions per ticker (last 10 days)")
    ax.set_xlabel("number of posts")
    return _fig_to_b64(fig)


def chart_sentiment(summary: pd.DataFrame) -> str:
    s = summary.sort_values("sentiment", ascending=True)
    colors = ["#dc2626" if v < 0 else "#16a34a" for v in s["sentiment"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(s["asset_name"], s["sentiment"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Average sentiment score per ticker (-1 bearish .. +1 bullish)")
    return _fig_to_b64(fig)


def chart_price_vs_sentiment(prices: pd.DataFrame, daily: pd.DataFrame,
                             ticker: str, name: str) -> str:
    px = prices[prices["ticker"] == ticker].sort_values("date")
    fig, ax1 = plt.subplots(figsize=(8, 3.6))
    ax1.plot(px["date"], px["close"], color="#111827", label="close price")
    ax1.set_ylabel("price (HKD)")
    ax1.legend(loc="upper left")
    d = daily[daily["ticker"] == ticker].sort_values("date") if not daily.empty else pd.DataFrame()
    if not d.empty:
        ax2 = ax1.twinx()
        ax2.bar(d["date"], d["sentiment_mean"],
                color=["#16a34a" if v >= 0 else "#dc2626" for v in d["sentiment_mean"]],
                alpha=0.45, label="daily sentiment")
        ax2.set_ylabel("daily sentiment")
        ax2.axhline(0, color="gray", linewidth=0.5)
        ax2.legend(loc="upper right")
    ax1.set_title(f"{name} ({ticker}) - price vs social sentiment")
    fig.autofmt_xdate()
    return _fig_to_b64(fig)


def chart_forecast(forecasts: pd.DataFrame) -> str:
    s = forecasts.sort_values("predicted_move_pct", ascending=True)
    colors = ["#dc2626" if v < 0 else "#16a34a" for v in s["predicted_move_pct"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(s["asset_name"], s["predicted_move_pct"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Predicted next-day move (n+1), % vs last close")
    return _fig_to_b64(fig)


def build_report(posts: pd.DataFrame, daily: pd.DataFrame,
                 prices: pd.DataFrame, snapshot: pd.DataFrame,
                 metrics: pd.DataFrame, forecasts: pd.DataFrame,
                 exchange: str) -> str:
    # Per-ticker summary table, sorted by mentions then sentiment (the brief)
    if posts.empty:
        summary = snapshot.assign(mentions=0, sentiment=0.0)
    else:
        summary = (posts.groupby(["ticker", "asset_name"])
                   .agg(mentions=("text", "count"), sentiment=("sentiment", "mean"))
                   .reset_index())
        summary = snapshot.merge(summary, on=["ticker", "asset_name"], how="left")
        summary[["mentions", "sentiment"]] = summary[["mentions", "sentiment"]].fillna(0)
    summary["direction"] = summary["sentiment"].map(sent.direction)
    summary["sentiment"] = summary["sentiment"].round(3)
    summary = summary.sort_values(["mentions", "sentiment"], ascending=False)

    rows = []
    for _, r in summary.iterrows():
        fc = forecasts[forecasts["ticker"] == r["ticker"]]
        mt = metrics[metrics["ticker"] == r["ticker"]]
        fc_html = ("no forecast" if fc.empty else
                   f"{fc.iloc[0]['predicted_next_close']} HKD "
                   f"({fc.iloc[0]['predicted_move_pct']:+.2f}%, {fc.iloc[0]['predicted_direction']})")
        mt_html = ("no metrics" if mt.empty else
                   f"MAE {mt.iloc[0]['mae_hkd']} HKD, "
                   f"direction {mt.iloc[0]['direction_accuracy_pct']}% "
                   f"over {mt.iloc[0]['oos_points']} out-of-sample days")
        rows.append(f"""
        <div class="card">
          <h2>{r['asset_name']} ({r['ticker']})</h2>
          <table>
            <tr><td>Current price</td><td>{r['current_price']} HKD</td></tr>
            <tr><td>Sentiment</td><td>{r['sentiment']} ({r['direction']}), {int(r['mentions'])} mentions</td></tr>
            <tr><td>Support / resistance (20d)</td><td>{r['support']} / {r['resistance']} HKD</td></tr>
            <tr><td>Predicted next close (n+1)</td><td>{fc_html}</td></tr>
            <tr><td>Out-of-sample check</td><td>{mt_html}</td></tr>
          </table>
          <img src="data:image/png;base64,{chart_price_vs_sentiment(prices, daily, r['ticker'], r['asset_name'])}"/>
        </div>""")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Hang Seng sentiment report</title>
<style>
 body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 2rem auto; color: #111827; }}
 h1 {{ margin-bottom: 0; }} .sub {{ color: #6b7280; margin-top: 0.2rem; }}
 .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 1rem 1.4rem; margin: 1.2rem 0; }}
 table {{ border-collapse: collapse; }} td {{ padding: 3px 14px 3px 0; vertical-align: top; }}
 td:first-child {{ color: #6b7280; white-space: nowrap; }}
 img {{ max-width: 100%; margin-top: 0.6rem; }}
</style></head><body>
<h1>Social sentiment price prediction - {exchange}</h1>
<p class="sub">Sources: LIHKG + Telegram (Apify) and exa.com news. Prices: Yahoo Finance.
Sorted by social mentions and sentiment. Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}.</p>
<h2>Overview</h2>
<img src="data:image/png;base64,{chart_mentions(summary)}"/>
<img src="data:image/png;base64,{chart_sentiment(summary)}"/>
{('<img src="data:image/png;base64,' + chart_forecast(forecasts) + '"/>') if not forecasts.empty else ''}
{''.join(rows)}
</body></html>"""
    return html
