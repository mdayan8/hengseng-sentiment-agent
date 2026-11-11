"""Central settings for the HengSeng sentiment price-prediction pipeline.

Everything you might want to change (which stocks, which days, which
Apify actors) lives here so the rest of the code stays clean.
"""

# --- Assets: 5 large Hang Seng (HKEX) stocks -------------------------------
# Each entry: Yahoo Finance ticker, company name, and the search words people
# actually use when talking about the stock online (ticker, cashtag, keyword).
TICKERS = [
    {"ticker": "0700.HK", "name": "Tencent",
     "keywords": ["0700", "Tencent", "騰訊"], "cashtag": "$0700.HK"},
    {"ticker": "9988.HK", "name": "Alibaba",
     "keywords": ["9988", "Alibaba", "阿里巴巴"], "cashtag": "$9988.HK"},
    {"ticker": "0005.HK", "name": "HSBC",
     "keywords": ["0005", "HSBC", "匯豐"], "cashtag": "$0005.HK"},
    {"ticker": "0941.HK", "name": "China Mobile",
     "keywords": ["0941", "China Mobile", "中國移動"], "cashtag": "$0941.HK"},
    {"ticker": "1299.HK", "name": "AIA",
     "keywords": ["1299", "AIA", "友邦"], "cashtag": "$1299.HK"},
]

EXCHANGE = "HKEX (Hang Seng)"

# --- Time window -------------------------------------------------------------
SOCIAL_DAYS = 10        # brief: scrape the last 10 days of social posts
PRICE_HISTORY_DAYS = 90 # extra price history so the model has data to learn from

# --- Apify actors (free tier works for these small runs) --------------------
# Source 1: LIHKG, the big HK retail-trader forum
APIFY_LIHKG_ACTOR = "claude_code_reviewer/lihkg-scraper-en"
# Source 2: Telegram, keyword search across public HK trading channels
APIFY_TELEGRAM_ACTOR = "lofomachines/telegram-keyword-search-scraper"

# --- Files -------------------------------------------------------------------
OUTPUT_DIR = "output"
