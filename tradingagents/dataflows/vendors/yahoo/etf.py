"""Resolve an ETF into the tickers of its largest holdings."""

from __future__ import annotations


def top_holdings(etf: str, top_n: int = 10) -> list[str]:
    """Return up to ``top_n`` holding symbols of ``etf``, largest weight first.

    Yahoo exposes only the fund's top ~10 holdings, so larger values return
    what is available.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    import yfinance as yf

    try:
        holdings = yf.Ticker(etf).funds_data.top_holdings
    except Exception as exc:
        raise ValueError(f"Could not fetch holdings for {etf}: {exc}") from exc
    if holdings is None or len(holdings) == 0:
        raise ValueError(f"No holdings data available for {etf}")
    symbols: list[str] = []
    for symbol in holdings.sort_values("Holding Percent", ascending=False).index:
        symbol = str(symbol).strip()
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols[:top_n]
