"""Trading 212 limit-order execution for stock-only CLI runs."""

from __future__ import annotations

import hashlib
import math
import os
import sqlite3
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
    ):
        self.mode = normalize_trade_mode(mode)
        if not api_key or not api_secret:
            raise Trading212Error("Set T212_API_KEY and T212_API_SECRET to enable execution.")
        self.base_url = _BASE_URLS[self.mode]
        self.auth = (api_key, api_secret)
        self.session = session or requests.Session()

    @classmethod
    def from_env(cls, mode: str, *, session: requests.Session | None = None):
        return cls(
            mode,
            os.environ.get("T212_API_KEY", ""),
            os.environ.get("T212_API_SECRET", ""),
            session=session,
        )

    def _request(self, method: str, path: str, **kwargs):
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                auth=self.auth,
                timeout=15,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise Trading212Error(
                f"Trading 212 request failed ({type(exc).__name__}); check connectivity and "
                "verify the order status in the account before retrying."
            ) from exc
        if not response.ok:
            raise Trading212Error(
                f"Trading 212 returned HTTP {response.status_code} for {method} {path}; "
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

    def resolve_instrument(self, symbol: str) -> dict[str, Any]:
        expected = self._expected_ticker(symbol)
        instruments = self._request("GET", "/equity/metadata/instruments")
        if not isinstance(instruments, list):
            raise Trading212Error("Trading 212 instruments response was not a list.")
        matches = [
            item for item in instruments
            if isinstance(item, dict)
            and item.get("ticker") == expected
            and item.get("type") == "STOCK"
        ]
        if len(matches) != 1:
            raise Trading212Error(
                f"Could not resolve {symbol} to one Trading 212 stock ({expected}); "
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
        return cursor.rowcount == 1


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
        Decimal("0.000001"), rounding=ROUND_DOWN
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
    client: Trading212Client | None = None,
) -> dict[str, Any]:
    """Submit one guarded limit order; returns an explicit skip or order receipt."""
    canonical_mode = normalize_trade_mode(mode)
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
    client = client or Trading212Client.from_env(canonical_mode)
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
        if limit_price < current_price:
            return {
                "status": "skipped",
                "reason": (
                    f"The sell limit ({limit_price:g}) is below Trading 212's current "
                    f"position price ({current_price:g}); no order was submitted."
                ),
            }
        quantity = -available
        order_value = 0.0

    execution_id = hashlib.sha256(
        f"{symbol.upper()}:{trade_date}:{canonical_mode}".encode()
    ).hexdigest()
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
