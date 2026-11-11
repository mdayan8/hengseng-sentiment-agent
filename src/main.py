"""Run the whole pipeline end to end.

    python main.py              # full run: scrape + news + model + report
    python main.py --offline    # skip Apify/Exa (uses cached CSVs in output/)

Outputs land in ../output:
    posts.csv, daily_features.csv, prices.csv,
    predictions.csv, forecast_metrics.csv, report.html
"""

import argparse
import os
import sys

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

import config
import features
import model
import news_exa
import prices as prices_mod
import report as report_mod
import scrape_apify

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", config.OUTPUT_DIR)
os.makedirs(OUT, exist_ok=True)


def path(name):
    return os.path.join(OUT, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="reuse cached posts/news instead of calling Apify/Exa")
    args = ap.parse_args()

    # 1. Social + news ingestion -------------------------------------------
    if args.offline:
        posts = pd.read_csv(path("posts.csv"))
        news = pd.read_csv(path("news.csv"))
        print("Offline mode: loaded cached posts and news")
    else:
        posts = scrape_apify.scrape_all()
        posts.to_csv(path("posts.csv"), index=False)
        news = news_exa.fetch_news()
        news.to_csv(path("news.csv"), index=False)

    # 2. Sentiment scoring ---------------------------------------------------
    all_text = pd.concat([posts, news], ignore_index=True)
    all_text = features.score_posts(all_text)
    all_text.to_csv(path("all_scored.csv"), index=False)

    # 3. Daily features + prices --------------------------------------------
    daily = features.daily_features(all_text)
    daily.to_csv(path("daily_features.csv"), index=False)
    px = prices_mod.download_prices()
    px.to_csv(path("prices.csv"), index=False)
    snapshot = prices_mod.ticker_snapshot(px)
    merged = features.merge_with_prices(daily, px)

    # 4. Predict n+1 with out-of-sample validation ---------------------------
    metrics, forecasts = model.predict_next_day(merged)
    metrics.to_csv(path("forecast_metrics.csv"), index=False)
    forecasts.to_csv(path("predictions.csv"), index=False)
    print("\nOut-of-sample metrics:")
    print(metrics.to_string(index=False))
    print("\nNext-day forecasts:")
    print(forecasts.to_string(index=False))

    # 5. Visual report ---------------------------------------------------------
    scored_posts = all_text[all_text["source"] != "exa_news"]
    html = report_mod.build_report(posts=scored_posts, daily=daily, prices=px,
                                   snapshot=snapshot, metrics=metrics,
                                   forecasts=forecasts, exchange=config.EXCHANGE)
    with open(path("report.html"), "w") as f:
        f.write(html)
    print(f"\nReport written to {path('report.html')}")


if __name__ == "__main__":
    main()
