"""Predict each stock's next-day (n+1) price and check the results honestly.

Model: a plain Ridge regression (a straight-line model that is hard to
overfit) per ticker. Inputs are today's return, volume change and today's
sentiment features; the target is tomorrow's close.

Validation: walk-forward (out-of-sample). We train on the past and test on
the future, then move the window forward. We never let the model see the
day it is predicting - that is what "rigorous out-of-sample" means here.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

FEATURES = ["ret_1d", "mentions", "sentiment_mean", "sentiment_weighted",
            "engagement"]


def _walk_forward(g: pd.DataFrame, min_train: int = 20):
    """Yield (train_df, test_row) pairs, always training on the past only."""
    g = g.sort_values("date").reset_index(drop=True)
    for i in range(min_train, len(g)):
        yield g.iloc[:i], g.iloc[i]


def predict_next_day(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (metrics per ticker, latest n+1 prediction per ticker)."""
    metrics, forecasts = [], []
    for sym, g in merged.groupby("ticker"):
        g = g.dropna(subset=["close", "ret_1d"]).sort_values("date")
        if len(g) < 30:
            print(f"  {sym}: not enough price history, skipped")
            continue
        preds, actuals, dates = [], [], []
        for train, test in _walk_forward(g):
            train = train.dropna(subset=["close_next"])
            if len(train) < 15 or pd.isna(test["close_next"]):
                continue
            m = Ridge(alpha=1.0)
            m.fit(train[FEATURES], train["close_next"])
            preds.append(float(m.predict(test[FEATURES].to_frame().T)[0]))
            actuals.append(float(test["close_next"]))
            dates.append(test["date"])
        if not preds:
            continue
        preds, actuals = np.array(preds), np.array(actuals)
        mae = float(np.mean(np.abs(preds - actuals)))
        # Direction accuracy: did we get up/down vs previous close right?
        prev_close = g.set_index("date").loc[dates, "close"].to_numpy()
        direction_hit = float(np.mean(np.sign(preds - prev_close)
                                      == np.sign(actuals - prev_close)))
        metrics.append({
            "ticker": sym,
            "asset_name": g["asset_name"].iloc[0],
            "oos_points": len(preds),
            "mae_hkd": round(mae, 2),
            "mape_pct": round(100 * mae / float(np.mean(actuals)), 2),
            "direction_accuracy_pct": round(100 * direction_hit, 1),
        })
        # Latest forecast: train on everything, predict the next trading day
        full = g.dropna(subset=["close_next"])
        m = Ridge(alpha=1.0).fit(full[FEATURES], full["close_next"])
        next_pred = float(m.predict(g[FEATURES].iloc[[-1]])[0])
        last_close = float(g["close"].iloc[-1])
        forecasts.append({
            "ticker": sym,
            "asset_name": g["asset_name"].iloc[0],
            "last_close": round(last_close, 2),
            "predicted_next_close": round(next_pred, 2),
            "predicted_move_pct": round(100 * (next_pred - last_close) / last_close, 2),
            "predicted_direction": "up" if next_pred > last_close else "down",
        })
    return pd.DataFrame(metrics), pd.DataFrame(forecasts)
