"""Turn raw post / news text into a sentiment score.

We use VADER (a small, rule-based sentiment model that runs locally, no API
key needed) and then nudge the score with a short list of finance words,
because "breakout" and "crash" mean more than VADER thinks.

Score range: -1 (very bearish) to +1 (very bullish).
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

# Simple finance lexicon: word -> score bump.
BULL_WORDS = {
    "breakout": 0.3, "upgrade": 0.25, "beat": 0.2, "rally": 0.3,
    "bullish": 0.35, "surge": 0.3, "soar": 0.3, "outperform": 0.25,
    "buy": 0.15, "moon": 0.25, "undervalued": 0.2, "growth": 0.1,
}
BEAR_WORDS = {
    "crash": -0.4, "downgrade": -0.25, "miss": -0.2, "plunge": -0.35,
    "bearish": -0.35, "dump": -0.3, "sell": -0.15, "overvalued": -0.2,
    "fraud": -0.4, "warning": -0.2, "lawsuit": -0.3, "collapse": -0.4,
}


def score_text(text: str) -> float:
    """Return a sentiment score in [-1, 1] for one piece of text."""
    if not text:
        return 0.0
    base = _analyzer.polarity_scores(text)["compound"]
    lower = text.lower()
    bump = sum(w for word, w in BULL_WORDS.items() if word in lower)
    bump += sum(w for word, w in BEAR_WORDS.items() if word in lower)
    return max(-1.0, min(1.0, base + bump))


def direction(score: float) -> str:
    """Label a score as bullish / bearish / neutral."""
    if score >= 0.15:
        return "bullish"
    if score <= -0.15:
        return "bearish"
    return "neutral"
