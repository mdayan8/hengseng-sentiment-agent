# Hang Seng Sentiment Price Prediction Agent

Intern position assessment for CrowdWisdomTrading (TZURONI LTD.).

A Python pipeline that reads what people say online about 5 Hang Seng
stocks, turns it into sentiment scores, and predicts each stock's next
trading-day (n+1) close price. Results are checked with strict
out-of-sample validation and shown in a visual report.

## What it does, step by step

1. **Scrape social posts** (`src/scrape_apify.py`) - two Hong Kong sources,
   both via Apify actors:
   - **LIHKG** - the biggest HK retail-investor forum
   - **Telegram** - public HK trading channels (keyword search)

   For each of 5 Hang Seng stocks it collects the last 10 days of posts:
   the text, the UTC timestamp, likes/reposts/views and the author's
   follower count.
2. **Pull recent news** (`src/news_exa.py`) - exa.com news search per stock,
   last 10 days, to complement social sentiment.
3. **Score sentiment** (`src/sentiment.py`) - VADER plus a small finance
   word list. Each post gets a score from -1 (very bearish) to +1 (very
   bullish).
4. **Align with prices** (`src/features.py`, `src/prices.py`) - Yahoo
   Finance daily bars; posts are grouped per stock per day so we can
   compare "what people said on day t" with "what the price did on t+1".
5. **Predict n+1** (`src/model.py`) - one Ridge regression per stock.
   Validation is walk-forward: the model only ever trains on days *before*
   the day it predicts. We report MAE and direction accuracy.
6. **Visual report** (`src/report.py`) - `output/report.html`: tickers
   sorted by mentions and sentiment, per-stock price-vs-sentiment charts,
   support/resistance levels and the next-day forecast.

## The 5 stocks

| Ticker  | Company       |
|---------|---------------|
| 0700.HK | Tencent       |
| 9988.HK | Alibaba       |
| 0005.HK | HSBC          |
| 0941.HK | China Mobile  |
| 1299.HK | AIA           |

## How to run it

```bash
pip install -r requirements.txt
cp .env.example .env      # then paste your free Apify and Exa tokens into .env
cd src
python main.py            # full run
python main.py --offline  # reuse the last scraped data (no API calls)
```

Outputs are written to `output/`: `report.html` (the visual report),
`predictions.csv`, `forecast_metrics.csv`, plus the raw data CSVs.

## Latest run (11 Nov 2026, live data)

Everything in `output/` comes from a real run on 11 Nov 2026:

- **71 social posts** kept after quality filtering (70 Telegram + 1 LIHKG)
  plus **50 Exa news items** across the 5 stocks.
- Predictions and out-of-sample metrics are in `output/report.html`.
  Honest headline numbers: direction accuracy ranges from **36.6% (AIA,
  worse than a coin flip) to 53.7% (China Mobile, Alibaba)** over 41
  out-of-sample days each. MAPE runs from 1.1% to 7.0%.

What the raw data looked like, honestly:

- Telegram keyword search matches a lot of junk - a drama-upload channel
  matched "Tencent" because Tencent Video makes the show. Every post is
  kept only if its text mentions the company AND uses finance language
  (see `FIN_TERMS` in `src/scrape_apify.py`), then deduplicated.
- LIHKG search returned very little real finance discussion for these 5
  keywords this week (most hits were entertainment news, filtered out).
- China Mobile had **zero** usable social posts in the window; its signal
  is news-only, and the report card says exactly that.

## Tokens used (for reproduction)

- **Apify** (free plan): token in `.env` as `APIFY_TOKEN`
  - Actor 1 (LIHKG): `claude_code_reviewer/lihkg-scraper-en`
  - Actor 2 (Telegram): `lofomachines/telegram-keyword-search-scraper`
- **Exa** (free credits): key in `.env` as `EXA_API_KEY`

The actual token values are shared in the submission email, as requested.

## Notes and honesty

- Predictions are small-sample and noisy; the out-of-sample metrics in the
  report show exactly how well (or badly) the model did. Nothing is hidden.
- Prices are daily bars, so "n+1" means the next trading day.
- Sentiment is VADER plus a small finance word list, running on mostly
  Cantonese text. VADER is an English model, so scores on Chinese posts
  lean on the finance word list and are rough. Treat them as a signal,
  not a measurement.
