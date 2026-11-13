"""Deterministic trend and relative-strength overlay for the final rating.

The fundamental theses are 3-6 month views; the five-day tape is dominated by
momentum. This module gives the executor a hard consistency check: a value
Overweight below the 200-day is not a buy, and an Underweight above the 200-day
is not a short. It also maps the five-tier rating to a target weight so a
long-only book can express a relative view instead of sitting in cash.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from tradingagents.dataflows.vendors.yahoo.market import get_closes
from tradingagents.memory.settlement import resolve_benchmark

_WEIGHTS = {
    "Buy": 1.50,
    "Overweight": 1.25,
    "Hold": 1.00,
    "Underweight": 0.50,
    "Sell": 0.00,
}


@dataclass(frozen=True)
class TrendState:
    ticker: str
    as_of: str
    price: float | None
    sma50: float | None
    sma200: float | None
    rs20: float | None
    rs60: float | None
    state: str  # up / down / mixed / unknown

    @property
    def weak(self) -> bool:
        return self.price is not None and self.sma50 is not None and self.price < self.sma50

    @property
    def strong(self) -> bool:
        return self.state in ("up", "down")

    def render(self) -> str:
        def pct(value):
            return "n/a" if value is None else f"{value:+.2%}"

        def price(value):
            return "n/a" if value is None else f"{value:,.2f}"

        return (
            f"Trend/relative strength as of {self.as_of}: price {price(self.price)}; "
            f"50-day {price(self.sma50)}; 200-day {price(self.sma200)}; "
            f"20-day relative strength {pct(self.rs20)}; "
            f"60-day relative strength {pct(self.rs60)}; state {self.state}."
        )


def _closes(ticker: str, trade_date: str, days: int = 420):
    start = (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
    return get_closes(ticker, start, trade_date)


def _relative(stock, benchmark, sessions: int) -> float | None:
    if len(stock) <= sessions or len(benchmark) <= sessions:
        return None
    stock_return = float(stock.iloc[-1] / stock.iloc[-1 - sessions] - 1)
    bench_return = float(benchmark.iloc[-1] / benchmark.iloc[-1 - sessions] - 1)
    return stock_return - bench_return


def trend_state(ticker: str, trade_date: str, config: dict) -> TrendState:
    """Point-in-time trend state from closes available on ``trade_date``.

    A data failure is reported as ``unknown``; the overlay then leaves the
    model's rating alone rather than inventing a trend.
    """
    try:
        stock = _closes(ticker, trade_date)
        benchmark_ticker = resolve_benchmark(ticker, config)
        benchmark = _closes(benchmark_ticker, trade_date)
        if stock is None or len(stock) < 20:
            raise ValueError("not enough closes")
        price = float(stock.iloc[-1])
        sma50 = float(stock.tail(50).mean()) if len(stock) >= 50 else None
        sma200 = float(stock.tail(200).mean()) if len(stock) >= 200 else None
        rs20 = _relative(stock, benchmark, 20)
        rs60 = _relative(stock, benchmark, 60)
        if sma50 is None or sma200 is None or rs20 is None:
            state = "unknown"
        elif price > sma200 and sma50 > sma200 and rs20 > 0:
            state = "up"
        elif price < sma200 and sma50 < sma200 and rs20 < 0:
            state = "down"
        else:
            state = "mixed"
        return TrendState(ticker, trade_date, price, sma50, sma200, rs20, rs60, state)
    except Exception:
        return TrendState(ticker, trade_date, None, None, None, None, None, "unknown")


def apply_trend_overlay(rating: str, trend: TrendState | None) -> str:
    """Risk filter: do not buy a downtrend or sell an uptrend.

    Hold stays Hold unless it is weak (then Underweight). Underweight/Sell are
    lifted to Hold when the trend is not down, so an avoid call cannot become a
    short or a full exit against an uptrend. Buy/Overweight are downgraded when
    the trend is down or price is below the 50-day.
    """
    if trend is None or trend.state == "unknown":
        return rating
    up = trend.state == "up"
    down = trend.state == "down"
    weak = trend.weak
    if rating in ("Buy", "Overweight"):
        if up:
            return rating
        return "Underweight" if (down or weak) else "Hold"
    if rating == "Hold":
        return "Underweight" if weak else "Hold"
    if rating == "Underweight":
        return "Underweight" if (down or weak) else "Hold"
    if rating == "Sell":
        if down:
            return "Sell"
        return "Underweight" if weak else "Hold"
    return rating


def rating_weight(rating: str) -> float:
    """Target weight relative to an equal-weight 10-name basket."""
    return _WEIGHTS.get(rating, 0.0)


def trend_context(ticker: str, trade_date: str, config: dict) -> str:
    return trend_state(ticker, trade_date, config).render()
