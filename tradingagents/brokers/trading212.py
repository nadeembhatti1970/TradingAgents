"""Trading 212 limit-order execution for stock-only CLI runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from datetime import date
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any

import requests

from tradingagents.default_config import normalize_trade_mode

_BASE_URLS = {
    "practice": "https://demo.trading212.com/api/v0",
    "real": "https://live.trading212.com/api/v0",
}
_EXCHANGE_SUFFIXES = {
    ".L": "GB",
    ".HK": "HK",
    ".T": "JP",
    ".TO": "CA",
    ".AX": "AU",
    ".SS": "CN",
    ".SZ": "CN",
    ".NS": "IN",
    ".BO": "IN",
}


# Yahoo suffix -> ISIN country prefixes and currencies Trading 212 lists the stock under.
_SUFFIX_HINTS = {
    ".L": {"isin": ("GB", "IE", "JE", "GG"), "currencies": ("GBX", "GBP")},
    ".HK": {"isin": ("HK", "KY", "BM", "CN"), "currencies": ("HKD",)},
    ".T": {"isin": ("JP",), "currencies": ("JPY",)},
    ".TO": {"isin": ("CA",), "currencies": ("CAD",)},
    ".AX": {"isin": ("AU",), "currencies": ("AUD",)},
}
# The instrument list is large and the endpoint allows about one call per 50 seconds.
_INSTRUMENTS_TTL_SECONDS = 24 * 3600


class Trading212Error(RuntimeError):
    """A rejected or malformed Trading 212 API operation."""


class Trading212Client:
    def __init__(
        self,
        mode: str,
        api_key: str,
        api_secret: str,
        *,
        session: requests.Session | None = None,
        instruments_cache: str | Path | None = None,
    ):
        self.instruments_cache = Path(instruments_cache) if instruments_cache else None
        self.mode = normalize_trade_mode(mode)
        if not api_key or not api_secret:
            raise Trading212Error("Set T212_API_KEY and T212_API_SECRET to enable execution.")
        self.base_url = _BASE_URLS[self.mode]
        self.auth = (api_key, api_secret)
        self.session = session or requests.Session()

    @classmethod
    def from_env(
        cls,
        mode: str,
        *,
        session: requests.Session | None = None,
        instruments_cache: str | Path | None = None,
    ):
        return cls(
            mode,
            os.environ.get("T212_API_KEY", ""),
            os.environ.get("T212_API_SECRET", ""),
            session=session,
            instruments_cache=instruments_cache,
        )

    def _request(self, method: str, path: str, **kwargs):
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                auth=self.auth,
                timeout=kwargs.pop("timeout", 15),
                **kwargs,
            )
        except requests.RequestException as exc:
            raise Trading212Error(
                f"Trading 212 request failed ({type(exc).__name__}); check connectivity and "
                "verify the order status in the account before retrying."
            ) from exc
        if not response.ok:
            hint = (
                " (the API key may lack the permission scope this endpoint needs)"
                if response.status_code == 403
                else ""
            )
            detail = " ".join((response.text or "").split())[:300]
            if detail:
                hint += f" [response: {detail}]"
            raise Trading212Error(
                f"Trading 212 returned HTTP {response.status_code} for {method} {path}{hint}; "
                "verify the account and order status before retrying."
            )
        try:
            return response.json()
        except ValueError as exc:
            raise Trading212Error(f"Trading 212 returned invalid JSON for {method} {path}.") from exc

    def account_details(self) -> dict[str, Any]:
        summary = self._request("GET", "/equity/account/summary")
        if not isinstance(summary, dict):
            raise Trading212Error("Trading 212 account summary response was not an object.")
        currency = summary.get("currency")
        if not isinstance(currency, str) or not currency.strip():
            raise Trading212Error("Trading 212 account summary did not include its primary currency.")
        cash = summary.get("cash")
        if not isinstance(cash, dict):
            raise Trading212Error("Trading 212 account summary did not include available cash.")
        try:
            available = float(cash.get("availableToTrade"))
        except (TypeError, ValueError) as exc:
            raise Trading212Error(
                "Trading 212 account summary returned invalid available cash."
            ) from exc
        if not math.isfinite(available) or available < 0:
            raise Trading212Error(
                "Trading 212 account summary returned invalid available cash."
            )
        return {"currency": currency.upper(), "available_to_trade": available}

    @staticmethod
    def _expected_ticker(symbol: str) -> str:
        cleaned = symbol.strip().upper()
        suffix = next((suffix for suffix in _EXCHANGE_SUFFIXES if cleaned.endswith(suffix)), None)
        if suffix:
            base = cleaned[:-len(suffix)]
            exchange = _EXCHANGE_SUFFIXES[suffix]
        else:
            base, exchange = cleaned, "US"
        if (
            not base
            or cleaned.endswith(("-USD", "-USDT", "-USDC"))
            or not base.replace("-", "").isalnum()
        ):
            raise Trading212Error(f"{symbol!r} is not a supported stock symbol.")
        return f"{base}_{exchange}_EQ"

    def _instruments(self) -> list[Any]:
        """The instrument list, cached on disk because it is large and rate limited."""
        cache = self.instruments_cache
        if cache is not None:
            cache = cache.with_name(f"{cache.stem}_{self.mode}{cache.suffix}")
        cached = None
        if cache is not None and cache.is_file():
            try:
                cached = json.loads(cache.read_text())
            except (OSError, ValueError):
                cached = None
            if isinstance(cached, list) and time.time() - cache.stat().st_mtime < _INSTRUMENTS_TTL_SECONDS:
                return cached
        try:
            instruments = self._request("GET", "/equity/metadata/instruments", timeout=60)
        except Trading212Error:
            if isinstance(cached, list):
                return cached
            raise
        if not isinstance(instruments, list):
            raise Trading212Error("Trading 212 instruments response was not a list.")
        if cache is not None:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(instruments))
            except OSError:
                pass
        return instruments

    def resolve_instrument(self, symbol: str) -> dict[str, Any]:
        expected = self._expected_ticker(symbol)
        instruments = self._instruments()
        matches = [
            item for item in instruments
            if isinstance(item, dict)
            and item.get("ticker") == expected
            and item.get("type") == "STOCK"
        ]
        if not matches:
            matches = find_instrument_matches(symbol, instruments)
        if len(matches) != 1:
            candidates = ", ".join(str(item.get("ticker")) for item in matches[:5])
            raise Trading212Error(
                f"Could not resolve {symbol} to one Trading 212 stock "
                f"(tried {expected}{'; candidates: ' + candidates if candidates else ''}); "
                "no order was submitted."
            )
        if not isinstance(matches[0].get("ticker"), str) or not matches[0]["ticker"]:
            raise Trading212Error(f"Trading 212 returned an invalid instrument for {symbol}.")
        return matches[0]

    def position(self, ticker: str) -> dict[str, Any] | None:
        positions = self._request(
            "GET", "/equity/positions", params={"ticker": ticker}
        )
        if not isinstance(positions, list):
            raise Trading212Error("Trading 212 positions response was not a list.")
        matches = [
            item for item in positions
            if isinstance(item, dict)
            and (
                item.get("ticker")
                or (
                    item.get("instrument", {}).get("ticker")
                    if isinstance(item.get("instrument"), dict)
                    else None
                )
            ) == ticker
        ]
        if len(matches) > 1:
            raise Trading212Error(f"Trading 212 returned duplicate positions for {ticker}.")
        return matches[0] if matches else None

    def place_limit_order(self, ticker: str, quantity: float, limit_price: float):
        if not math.isfinite(quantity) or quantity == 0:
            raise Trading212Error("Order quantity must be a non-zero finite number.")
        if not math.isfinite(limit_price) or limit_price <= 0:
            raise Trading212Error("Order limit price must be a positive finite number.")
        result = self._request(
            "POST",
            "/equity/orders/limit",
            json={
                "ticker": ticker,
                "quantity": quantity,
                "limitPrice": limit_price,
                "timeValidity": "DAY",
            },
        )
        if not isinstance(result, dict):
            raise Trading212Error("Trading 212 order response was not an object.")
        return result

    def order_history(self, limit: int = 50) -> list[dict[str, Any]]:
        result = self._request("GET", "/equity/history/orders", params={"limit": limit})
        items = result.get("items") if isinstance(result, dict) else None
        return [item["order"] for item in items or [] if isinstance(item, dict) and "order" in item]


def find_instrument_matches(symbol: str, instruments: list[Any]) -> list[dict[str, Any]]:
    """Match a Yahoo-style symbol to Trading 212 stocks by short name, ISIN and currency.

    Trading 212 tickers do not follow the Yahoo pattern (London stocks look like
    ``HSBAl_EQ``), so this matches on the base symbol and prefers the instruments
    whose ISIN country or currency fits the symbol's exchange suffix.
    """
    cleaned = symbol.strip().upper()
    suffix = next((suffix for suffix in _SUFFIX_HINTS if cleaned.endswith(suffix)), "")
    base = cleaned[: -len(suffix)] if suffix else cleaned
    hints = _SUFFIX_HINTS.get(suffix)
    fitting, others = [], []
    for item in instruments:
        if not isinstance(item, dict) or item.get("type") != "STOCK":
            continue
        short = str(item.get("shortName", "")).upper()
        ticker_base = str(item.get("ticker", "")).split("_")[0].upper()
        names = {short, ticker_base}
        if suffix == ".L":
            names.add(ticker_base.removesuffix("L"))
        if base not in names:
            continue
        isin_country = str(item.get("isin", ""))[:2]
        if hints:
            fits = isin_country in hints["isin"] or item.get("currencyCode") in hints["currencies"]
        else:
            fits = isin_country == "US" or item.get("currencyCode") == "USD"
        (fitting if fits else others).append(item)
    pool = fitting or others
    if suffix == ".L" and len(pool) > 1:
        # London listings are ``<base>l_EQ``; this separates them from ADR lines.
        london = [i for i in pool if str(i.get("ticker", "")).upper() == f"{base}L_EQ"]
        pool = london or pool
    return pool


def _ensure_journal(db: sqlite3.Connection) -> None:
    columns = {row[1] for row in db.execute("PRAGMA table_info(trading212_orders)")}
    if columns and "value" not in columns:
        db.execute("ALTER TABLE trading212_orders ADD COLUMN value REAL NOT NULL DEFAULT 0")


def _journal_spent_today(path: Path, mode: str) -> float:
    """Buy value already claimed today; uncertain orders count as spent."""
    if not path.exists():
        return 0.0
    with sqlite3.connect(path) as db:
        _ensure_journal(db)
        row = db.execute(
            "SELECT COALESCE(SUM(value), 0) FROM trading212_orders "
            "WHERE mode = ? AND date(created_at) = date('now')",
            (mode,),
        ).fetchone()
    return float(row[0])


def _journal_claim(
    path: Path, execution_id: str, symbol: str, mode: str, value: float = 0.0
) -> bool:
    """Claim this ticker/date/mode once; an uncertain POST is never retried blindly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute(
            """CREATE TABLE IF NOT EXISTS trading212_orders (
                execution_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                order_id TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                value REAL NOT NULL DEFAULT 0
            )"""
        )
        _ensure_journal(db)
        cursor = db.execute(
            "INSERT OR IGNORE INTO trading212_orders "
            "(execution_id, symbol, mode, status, value) VALUES (?, ?, ?, 'submitting', ?)",
            (execution_id, symbol, mode, value),
        )
        if cursor.rowcount == 1:
            return True
        # A broker-confirmed rejection never reached the market, so it can be retried.
        cursor = db.execute(
            "UPDATE trading212_orders SET status = 'submitting', order_id = NULL, value = ?, "
            "created_at = CURRENT_TIMESTAMP WHERE execution_id = ? AND status IN "
            f"({','.join('?' * len(_RETRYABLE))})",
            (value, execution_id, *_RETRYABLE),
        )
        return cursor.rowcount == 1


_RETRYABLE = ("REJECTED", "CANCELLED")


def _journal_reconcile(path: Path, mode: str, client: Any) -> None:
    """Mark journalled orders that Trading 212 rejected or cancelled unfilled."""
    if not path.exists():
        return
    with sqlite3.connect(path) as db:
        _ensure_journal(db)
        pending = {
            row[0]: row[1]
            for row in db.execute(
                "SELECT order_id, execution_id FROM trading212_orders WHERE mode = ? "
                "AND status = 'submitted' AND order_id IS NOT NULL "
                "AND date(created_at) = date('now')",
                (mode,),
            )
        }
        if not pending:
            return
        try:
            history = client.order_history()
        except Trading212Error:
            return
        for order in history:
            execution_id = pending.get(str(order.get("id")))
            status = str(order.get("status") or "").upper()
            if (
                execution_id
                and status in _RETRYABLE
                and not float(order.get("filledQuantity") or 0)
            ):
                db.execute(
                    "UPDATE trading212_orders SET status = ?, value = 0 WHERE execution_id = ?",
                    (status, execution_id),
                )


def _journal_update(path: Path, execution_id: str, status: str, order_id: str | None = None) -> None:
    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE trading212_orders SET status = ?, order_id = ? WHERE execution_id = ?",
            (status, order_id, execution_id),
        )


def _quantity_for_buy(max_value: float, limit_price: float) -> float:
    if not math.isfinite(max_value) or max_value <= 0:
        raise Trading212Error(
            "Automatic buys are disabled. Set TRADINGAGENTS_T212_MAX_BUY_VALUE "
            "to a positive cap in your account's primary currency."
        )
    if not math.isfinite(limit_price) or limit_price <= 0:
        raise Trading212Error("Trader did not provide a valid positive limit price.")
    quantity = (Decimal(str(max_value)) / Decimal(str(limit_price))).quantize(
        Decimal("0.001"), rounding=ROUND_DOWN
    )
    if quantity <= 0:
        raise Trading212Error("The configured buy cap is too small for one fractional share.")
    return float(quantity)


def action_for_rating(rating: str) -> str | None:
    """Resolve the final portfolio-manager rating to a trade or no action."""
    return {
        "Buy": "Buy",
        "Overweight": "Buy",
        "Sell": "Sell",
        "Underweight": "Sell",
    }.get(rating)


def execute_trade(
    *,
    symbol: str,
    trade_date: str,
    action: str,
    limit_price: float | None,
    mode: str,
    max_buy_value: float,
    data_dir: str | Path,
    max_daily_buy_value: float = 0.0,
    entry_price: float | None = None,
    client: Trading212Client | None = None,
) -> dict[str, Any]:
    """Submit one guarded limit order; returns an explicit skip or order receipt."""
    canonical_mode = normalize_trade_mode(mode)
    # The limit is the most a buy may pay; resting at the proposal's lower entry
    # keeps the pullback entry the debate agreed on instead of filling at the cap.
    if (
        action == "Buy"
        and isinstance(entry_price, (int, float))
        and not isinstance(entry_price, bool)
        and math.isfinite(entry_price)
        and entry_price > 0
        and limit_price is not None
        and math.isfinite(limit_price)
        and entry_price < limit_price
    ):
        limit_price = float(entry_price)
    if action not in ("Buy", "Sell"):
        return {"status": "skipped", "reason": "The final decision is Hold or has no trade action."}
    if limit_price is None or not math.isfinite(limit_price) or limit_price <= 0:
        return {"status": "skipped", "reason": "No valid action limit price was provided."}
    if trade_date != date.today().isoformat():
        return {"status": "skipped", "reason": "Orders are only allowed for today's analysis date."}
    if action == "Buy" and (
        not math.isfinite(max_buy_value) or max_buy_value <= 0
    ):
        return {
            "status": "skipped",
            "reason": (
                "Automatic buys are disabled. Set TRADINGAGENTS_T212_MAX_BUY_VALUE "
                "to a positive cap in your account's primary currency."
            ),
        }
    if client is None and not (
        os.environ.get("T212_API_KEY") and os.environ.get("T212_API_SECRET")
    ):
        return {
            "status": "skipped",
            "reason": "Set T212_API_KEY and T212_API_SECRET to enable Trading 212 execution.",
        }
    client = client or Trading212Client.from_env(
        canonical_mode,
        instruments_cache=Path(data_dir) / "trading212_instruments.json",
    )
    _journal_reconcile(Path(data_dir) / "trading212_orders.sqlite", canonical_mode, client)
    instrument = client.resolve_instrument(symbol)
    ticker = instrument["ticker"]
    currency = (instrument.get("currencyCode") or "").upper()
    account = client.account_details()
    account_currency = account["currency"]
    # GBX instruments are priced in pence, so values in GBP use price / 100.
    price_divisor = 100.0 if currency == "GBX" else 1.0
    if currency == "GBX":
        currency = "GBP"
    if action == "Buy" and (not currency or currency != account_currency):
        return {
            "status": "skipped",
            "reason": (
                f"{symbol} trades in {currency or 'an unknown currency'}, but Trading 212 orders "
                f"require the primary account currency ({account_currency})."
            ),
        }

    if action == "Buy":
        affordable_value = min(max_buy_value, account["available_to_trade"])
        if affordable_value <= 0:
            return {
                "status": "skipped",
                "reason": "The account has no cash currently available to trade.",
            }
        if max_daily_buy_value > 0:
            remaining = max_daily_buy_value - _journal_spent_today(
                Path(data_dir) / "trading212_orders.sqlite", canonical_mode
            )
            if remaining <= 0:
                return {
                    "status": "skipped",
                    "reason": "The daily buy cap has been reached; no order was submitted.",
                }
            affordable_value = min(affordable_value, remaining)
        quantity = _quantity_for_buy(affordable_value, limit_price / price_divisor)
        order_value = quantity * limit_price / price_divisor
        if order_value > max_buy_value:
            raise Trading212Error("Calculated buy order exceeds the configured maximum value.")
    else:
        position = client.position(ticker)
        if position is None:
            return {"status": "skipped", "reason": "There is no open long position to close."}
        try:
            available = float(position.get("quantityAvailableForTrading") or 0)
        except (TypeError, ValueError) as exc:
            raise Trading212Error(
                "Trading 212 returned an invalid quantity available for trading."
            ) from exc
        if not math.isfinite(available) or available <= 0:
            return {
                "status": "skipped",
                "reason": "No shares are currently available to sell; short selling is disabled.",
            }
        try:
            current_price = float(position.get("currentPrice"))
        except (TypeError, ValueError) as exc:
            raise Trading212Error(
                "Trading 212 did not return a valid current price for the open position."
            ) from exc
        if not math.isfinite(current_price) or current_price <= 0:
            raise Trading212Error(
                "Trading 212 did not return a valid current price for the open position."
            )
        quantity = -available
        order_value = 0.0

    # Sells have their own claim so an earlier buy never blocks closing a position.
    claim_key = f"{symbol.upper()}:{trade_date}:{canonical_mode}"
    if action == "Sell":
        claim_key += ":sell"
    execution_id = hashlib.sha256(claim_key.encode()).hexdigest()
    journal = Path(data_dir) / "trading212_orders.sqlite"
    if not _journal_claim(journal, execution_id, symbol, canonical_mode, order_value):
        return {
            "status": "skipped",
            "reason": (
                "An order was already submitted or may have been submitted for this "
                "symbol/date/mode; check Trading 212 before taking further action."
            ),
        }

    try:
        receipt = client.place_limit_order(ticker, quantity, float(limit_price))
    except Exception:
        _journal_update(journal, execution_id, "uncertain")
        raise
    order_id = str(receipt.get("id")) if receipt.get("id") is not None else None
    _journal_update(journal, execution_id, "submitted", order_id)
    return {
        "status": "submitted",
        "mode": canonical_mode,
        "ticker": ticker,
        "side": action,
        "quantity": quantity,
        "limit_price": float(limit_price),
        "order_id": order_id,
        "order_status": receipt.get("status"),
    }
