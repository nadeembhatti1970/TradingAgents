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


@pytest.mark.unit
def test_gbx_instrument_buys_in_gbp_account_using_pence_prices(tmp_path):
    client = _Client(currency="GBX")
    client.currency = "GBP"
    result = execute_trade(
        symbol="AAPL",
        trade_date=date.today().isoformat(),
        action="Buy",
        limit_price=1000,  # pence
        mode="practice",
        max_buy_value=50,
        data_dir=tmp_path,
        client=client,
    )
    assert result["status"] == "submitted"
    assert result["quantity"] == 5  # GBP 50 / GBP 10
    assert result["quantity"] * result["limit_price"] / 100 <= 50


@pytest.mark.unit
def test_daily_buy_cap_limits_total_spend_across_symbols(tmp_path):
    class _Multi(_Client):
        def resolve_instrument(self, symbol):
            return {**self.instrument, "ticker": f"{symbol}_US_EQ"}

    def buy(symbol):
        return execute_trade(
            symbol=symbol,
            trade_date=date.today().isoformat(),
            action="Buy",
            limit_price=10,
            mode="practice",
            max_buy_value=100,
            max_daily_buy_value=150,
            data_dir=tmp_path,
            client=_Multi(),
        )

    assert buy("AAA")["quantity"] == 10
    assert buy("BBB")["quantity"] == 5  # only 50 of the 150 cap remains
    capped = buy("CCC")
    assert capped["status"] == "skipped"
    assert "daily buy cap" in capped["reason"]


_LSE_INSTRUMENTS = [
    {"ticker": "HSBAl_EQ", "type": "STOCK", "shortName": "HSBA", "currencyCode": "GBX",
     "isin": "GB0005405286"},
    {"ticker": "HSBC_US_EQ", "type": "STOCK", "shortName": "HSBC", "currencyCode": "USD",
     "isin": "US4042804066"},
    {"ticker": "AAPL_US_EQ", "type": "STOCK", "shortName": "AAPL", "currencyCode": "USD",
     "isin": "US0378331005"},
    {"ticker": "HSBAl_ETF", "type": "ETF", "shortName": "HSBA", "currencyCode": "GBX",
     "isin": "GB0000000000"},
]


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def request(self, method, url, **kwargs):
        self.calls += 1

        class _Response:
            ok = True

            def json(_self):
                return self.payload

        return _Response()


@pytest.mark.unit
def test_london_symbol_resolves_to_trading212_ticker_by_short_name():
    from tradingagents.brokers.trading212 import find_instrument_matches

    assert [i["ticker"] for i in find_instrument_matches("HSBA.L", _LSE_INSTRUMENTS)] == ["HSBAl_EQ"]
    assert find_instrument_matches("XXX.L", _LSE_INSTRUMENTS) == []


@pytest.mark.unit
def test_client_resolves_when_yahoo_pattern_ticker_is_not_listed():
    client = Trading212Client("practice", "k", "s", session=_Session(_LSE_INSTRUMENTS))
    assert client.resolve_instrument("HSBA.L")["ticker"] == "HSBAl_EQ"
    assert client.resolve_instrument("AAPL")["ticker"] == "AAPL_US_EQ"
    with pytest.raises(Trading212Error, match="Could not resolve XXX.L"):
        client.resolve_instrument("XXX.L")


@pytest.mark.unit
def test_instrument_list_is_cached_on_disk(tmp_path):
    cache = tmp_path / "instruments.json"
    first = _Session(_LSE_INSTRUMENTS)
    Trading212Client(
        "practice", "k", "s", session=first, instruments_cache=cache
    ).resolve_instrument("HSBA.L")
    second = _Session([])
    again = Trading212Client("practice", "k", "s", session=second, instruments_cache=cache)
    assert again.resolve_instrument("HSBA.L")["ticker"] == "HSBAl_EQ"
    assert (first.calls, second.calls) == (1, 0)


@pytest.mark.unit
def test_london_symbol_prefers_london_line_over_adr_with_same_short_name():
    from tradingagents.brokers.trading212 import find_instrument_matches

    instruments = [
        {"ticker": "AZN_US_EQ", "type": "STOCK", "shortName": "AZN", "currencyCode": "GBX"},
        {"ticker": "AZNl_EQ", "type": "STOCK", "shortName": "AZN", "currencyCode": "GBX"},
    ]
    assert [i["ticker"] for i in find_instrument_matches("AZN.L", instruments)] == ["AZNl_EQ"]
