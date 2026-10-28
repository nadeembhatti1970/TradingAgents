from __future__ import annotations

from datetime import date, timedelta

import pytest

from tradingagents.brokers.trading212 import (
    Trading212Client,
    Trading212Error,
    action_for_rating,
    execute_trade,
)


class _Client:
    def __init__(self, *, currency="USD", position=None):
        self.instrument = {
            "ticker": "AAPL_US_EQ",
            "type": "STOCK",
            "currencyCode": currency,
        }
        self.currency = "USD"
        self.open_position = position
        self.orders = []

    def resolve_instrument(self, symbol):
        assert symbol == "AAPL"
        return self.instrument

    def account_details(self):
        return {"currency": self.currency, "available_to_trade": 1000}

    def position(self, ticker):
        assert ticker == "AAPL_US_EQ"
        return self.open_position

    def place_limit_order(self, ticker, quantity, limit_price):
        self.orders.append((ticker, quantity, limit_price))
        return {"id": 123, "status": "NEW"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "rating,expected",
    [("Buy", "Buy"), ("Overweight", "Buy"), ("Sell", "Sell"),
     ("Underweight", "Sell"), ("Hold", None), ("REVIEW", None)],
)
def test_final_rating_maps_to_order_action(rating, expected):
    assert action_for_rating(rating) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "symbol,expected",
    [("AAPL", "AAPL_US_EQ"), ("AZN.L", "AZN_GB_EQ"), ("0700.HK", "0700_HK_EQ")],
)
def test_trading212_stock_symbol_mapping(symbol, expected):
    assert Trading212Client._expected_ticker(symbol) == expected


@pytest.mark.unit
def test_market_or_non_stock_symbols_are_not_guessed():
    with pytest.raises(Trading212Error, match="not a supported stock symbol"):
        Trading212Client._expected_ticker("BTC-USD")


@pytest.mark.unit
def test_buy_uses_capped_fractional_quantity_and_journals_submission(tmp_path):
    client = _Client()
    options = {
        "symbol": "AAPL",
        "trade_date": date.today().isoformat(),
        "action": "Buy",
        "limit_price": 30,
        "mode": "practice",
        "max_buy_value": 100,
        "data_dir": tmp_path,
        "client": client,
    }

    result = execute_trade(**options)

    assert result == {
        "status": "submitted",
        "mode": "practice",
        "ticker": "AAPL_US_EQ",
        "side": "Buy",
        "quantity": 3.333333,
        "limit_price": 30,
        "order_id": "123",
        "order_status": "NEW",
    }
    assert client.orders == [("AAPL_US_EQ", 3.333333, 30)]
    duplicate = execute_trade(**options)
    assert duplicate["status"] == "skipped"
    assert "already submitted" in duplicate["reason"]
    assert len(client.orders) == 1


@pytest.mark.unit
def test_buy_never_exceeds_available_account_cash(tmp_path):
    client = _Client()
    client.account_details = lambda: {"currency": "USD", "available_to_trade": 25}
    result = execute_trade(
        symbol="AAPL",
        trade_date=date.today().isoformat(),
        action="Buy",
        limit_price=10,
        mode="practice",
        max_buy_value=100,
        data_dir=tmp_path,
        client=client,
    )
    assert result["quantity"] == 2.5
    assert result["quantity"] * result["limit_price"] <= 25


@pytest.mark.unit
def test_sell_closes_only_available_long_shares(tmp_path):
    client = _Client(position={"quantityAvailableForTrading": 2.5, "currentPrice": 35})
    result = execute_trade(
        symbol="AAPL",
        trade_date=date.today().isoformat(),
        action="Sell",
        limit_price=40,
        mode="practice",
        max_buy_value=0,
        data_dir=tmp_path,
        client=client,
    )
    assert result["quantity"] == -2.5
    assert client.orders == [("AAPL_US_EQ", -2.5, 40)]


@pytest.mark.unit
def test_sell_without_available_long_shares_does_not_submit(tmp_path):
    client = _Client(position={"quantityAvailableForTrading": 0, "currentPrice": 35})
    result = execute_trade(
        symbol="AAPL",
        trade_date=date.today().isoformat(),
        action="Sell",
        limit_price=40,
        mode="practice",
        max_buy_value=0,
        data_dir=tmp_path,
        client=client,
    )
    assert result["status"] == "skipped"
    assert "short selling is disabled" in result["reason"]
    assert client.orders == []


@pytest.mark.unit
def test_buy_is_disabled_until_an_explicit_value_cap_is_configured(tmp_path):
    client = _Client()
    result = execute_trade(
        symbol="AAPL",
        trade_date=date.today().isoformat(),
        action="Buy",
        limit_price=40,
        mode="practice",
        max_buy_value=0,
        data_dir=tmp_path,
        client=client,
    )
    assert result["status"] == "skipped"
    assert "buys are disabled" in result["reason"]
    assert client.orders == []


@pytest.mark.unit
def test_currency_mismatch_and_old_analysis_date_never_submit(tmp_path):
    client = _Client(currency="EUR")
    currency_result = execute_trade(
        symbol="AAPL",
        trade_date=date.today().isoformat(),
        action="Buy",
        limit_price=40,
        mode="practice",
        max_buy_value=100,
        data_dir=tmp_path,
        client=client,
    )
    assert currency_result["status"] == "skipped"
    assert "primary account currency" in currency_result["reason"]

    old_date_result = execute_trade(
        symbol="AAPL",
        trade_date=(date.today() - timedelta(days=1)).isoformat(),
        action="Buy",
        limit_price=40,
        mode="practice",
        max_buy_value=100,
        data_dir=tmp_path,
        client=client,
    )
    assert old_date_result["status"] == "skipped"
    assert "today's analysis date" in old_date_result["reason"]
    assert client.orders == []


@pytest.mark.unit
def test_no_credentials_means_no_execution(tmp_path, monkeypatch):
    monkeypatch.delenv("T212_API_KEY", raising=False)
    monkeypatch.delenv("T212_API_SECRET", raising=False)
    result = execute_trade(
        symbol="AAPL",
        trade_date=date.today().isoformat(),
        action="Sell",
        limit_price=40,
        mode="real",
        max_buy_value=0,
        data_dir=tmp_path,
    )
    assert result["status"] == "skipped"
    assert "T212_API_KEY" in result["reason"]


@pytest.mark.unit
def test_mode_selects_explicit_demo_or_live_endpoint():
    practice = Trading212Client("practice", "key", "secret")
    real = Trading212Client("real", "key", "secret")
    assert practice.base_url == "https://demo.trading212.com/api/v0"
    assert real.base_url == "https://live.trading212.com/api/v0"
