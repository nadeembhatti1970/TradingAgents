from __future__ import annotations

from datetime import date, timedelta

import pytest

from tradingagents.brokers.trading212 import (
    Trading212Client,
    Trading212Error,
    action_for_rating,
    execute_trade,
    symbol_from_t212,
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
        self.market_orders = []
        self.stop_orders = []
        self.history = []
        self.positions_data = []

    def order_history(self, limit=50):
        return self.history

    def positions(self):
        return list(self.positions_data)

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

    def place_market_order(self, ticker, quantity):
        self.market_orders.append((ticker, quantity))
        return {"id": 456, "status": "NEW"}

    def place_stop_order(self, ticker, quantity, stop_price):
        self.stop_orders.append((ticker, quantity, stop_price))
        return {"id": 789, "status": "NEW"}


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
        "quantity": 3.333,
        "limit_price": 30,
        "order_id": "123",
        "order_status": "NEW",
    }
    assert client.orders == [("AAPL_US_EQ", 3.333, 30)]
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


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ticker", "currency", "expected"),
    [
        ("RRl_EQ", "GBX", "RR.L"),
        ("BPl_EQ", "GBX", "BP.L"),
        ("AZNl_EQ", "GBX", "AZN.L"),
        ("AAPL_US_EQ", "USD", "AAPL"),
        ("0700_HK_EQ", "HKD", "0700.HK"),
        ("7203_T_EQ", "JPY", "7203.T"),
        ("SHOP_TO_EQ", "CAD", "SHOP.TO"),
        ("BHP_AX_EQ", "AUD", "BHP.AX"),
        ("AAPL", "USD", "AAPL"),
        ("AAPL", None, "AAPL"),
        ("RRl_EQ", None, "RR.L"),
    ],
)
def test_t212_listings_map_to_the_yahoo_symbols_the_agents_use(ticker, currency, expected):
    assert symbol_from_t212(ticker, currency) == expected


class _PortfolioClient(Trading212Client):
    def __init__(self):
        super().__init__("practice", "key", "secret")

    def account_details(self):
        return {"currency": "GBP", "available_to_trade": 100.0}

    def positions(self):
        return [
            {
                "instrument": {"ticker": "RRl_EQ", "type": "STOCK", "currencyCode": "GBX"},
                "quantity": 3.846,
                "averagePricePaid": 1273.798,
            },
            {
                "instrument": {"ticker": "AAPL_US_EQ", "type": "STOCK", "currencyCode": "USD"},
                "quantity": 2,
                "averagePricePaid": 150.0,
            },
        ]


@pytest.mark.unit
def test_live_portfolio_maps_broker_tickers_to_the_analysed_symbols():
    book = _PortfolioClient().portfolio_context()

    assert book.cash == 100.0 and book.currency == "GBP"
    rr = book.position_in("RR.L")
    assert rr is not None and rr.quantity == 3.846
    assert rr.broker_ticker == "RRl_EQ"
    assert "Current position in RR.L" in book.render("RR.L")
    assert "listed as RRl_EQ" in book.render("RR.L")
    assert book.position_in("AAPL").quantity == 2


@pytest.mark.unit
def test_sell_below_market_still_submits_so_a_loser_can_be_closed(tmp_path):
    client = _Client(position={"quantityAvailableForTrading": 2.5, "currentPrice": 35})
    result = execute_trade(
        symbol="AAPL",
        trade_date=date.today().isoformat(),
        action="Sell",
        limit_price=30,
        mode="practice",
        max_buy_value=0,
        data_dir=tmp_path,
        client=client,
    )
    assert result["status"] == "submitted"
    assert client.orders == [("AAPL_US_EQ", -2.5, 30)]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("entry", "expected"),
    [(25, 25), (35, 30), (None, 30), (0, 30), (float("nan"), 30)],
)
def test_buy_rests_at_a_lower_entry_price_only_when_explicitly_requested(tmp_path, entry, expected):
    client = _Client()
    execute_trade(
        symbol="AAPL",
        trade_date=date.today().isoformat(),
        action="Buy",
        limit_price=30,
        entry_price=entry,
        rest_at_entry=True,
        mode="practice",
        max_buy_value=100,
        data_dir=tmp_path,
        client=client,
    )
    assert client.orders[0][2] == expected


@pytest.mark.unit
def test_buy_uses_the_explicit_limit_by_default_even_when_an_entry_is_lower(tmp_path):
    """The entry is a target, not the order cap; lowering the limit to it made
    several pullback orders non-marketable and the broker rejected them."""
    client = _Client()
    execute_trade(
        symbol="AAPL",
        trade_date=date.today().isoformat(),
        action="Buy",
        limit_price=30,
        entry_price=25,
        mode="practice",
        max_buy_value=100,
        data_dir=tmp_path,
        client=client,
    )
    assert client.orders == [("AAPL_US_EQ", 3.333, 30)]


@pytest.mark.unit
def test_market_sell_uses_the_market_endpoint_and_needs_no_limit(tmp_path):
    client = _Client(position={"quantityAvailableForTrading": 2.5, "currentPrice": 35})
    result = execute_trade(
        symbol="AAPL",
        trade_date=date.today().isoformat(),
        action="Sell",
        limit_price=None,
        order_type="market",
        mode="practice",
        max_buy_value=0,
        data_dir=tmp_path,
        client=client,
    )
    assert result["status"] == "submitted"
    assert result["limit_price"] is None
    assert client.orders == []
    assert client.market_orders == [("AAPL_US_EQ", -2.5)]


@pytest.mark.unit
def test_a_requested_stop_is_placed_best_effort_after_the_buy(tmp_path):
    client = _Client()
    result = execute_trade(
        symbol="AAPL",
        trade_date=date.today().isoformat(),
        action="Buy",
        limit_price=30,
        stop_loss=25,
        mode="practice",
        max_buy_value=100,
        data_dir=tmp_path,
        client=client,
    )
    assert result["status"] == "submitted"
    assert client.orders == [("AAPL_US_EQ", 3.333, 30)]
    assert client.stop_orders == [("AAPL_US_EQ", -3.333, 25.0)]
    assert result["stop_order"]["stop_price"] == 25.0


@pytest.mark.unit
def test_a_failed_stop_does_not_hide_a_submitted_buy(tmp_path):
    client = _Client()

    def fail(*args, **kwargs):
        raise RuntimeError("stop rejected")

    client.place_stop_order = fail
    result = execute_trade(
        symbol="AAPL",
        trade_date=date.today().isoformat(),
        action="Buy",
        limit_price=30,
        stop_loss=25,
        mode="practice",
        max_buy_value=100,
        data_dir=tmp_path,
        client=client,
    )
    assert result["status"] == "submitted"
    assert "stop rejected" in result["stop_error"]


def _opts(tmp_path, client, **extra):
    return {
        "symbol": "AAPL",
        "trade_date": date.today().isoformat(),
        "action": "Buy",
        "limit_price": 30,
        "mode": "practice",
        "max_buy_value": 100,
        "data_dir": tmp_path,
        "client": client,
        **extra,
    }


@pytest.mark.unit
def test_rejected_order_is_released_and_can_be_retried(tmp_path):
    client = _Client()
    assert execute_trade(**_opts(tmp_path, client))["status"] == "submitted"
    assert execute_trade(**_opts(tmp_path, client))["status"] == "skipped"

    client.history = [{"id": 123, "status": "REJECTED", "filledQuantity": 0}]
    retry = execute_trade(**_opts(tmp_path, client))

    assert retry["status"] == "submitted"
    assert len(client.orders) == 2


@pytest.mark.unit
def test_filled_or_open_order_stays_claimed(tmp_path):
    client = _Client()
    execute_trade(**_opts(tmp_path, client))
    client.history = [{"id": 123, "status": "FILLED", "filledQuantity": 3.3}]

    assert execute_trade(**_opts(tmp_path, client))["status"] == "skipped"
    assert len(client.orders) == 1


@pytest.mark.unit
def test_rejected_order_stops_counting_toward_daily_cap(tmp_path):
    client = _Client()
    opts = _opts(tmp_path, client, max_daily_buy_value=100)
    assert execute_trade(**opts)["status"] == "submitted"
    client.history = [{"id": 123, "status": "REJECTED", "filledQuantity": 0}]
    other = _opts(tmp_path, client, max_daily_buy_value=100)
    # Same symbol is reclaimable, so the cap must have been freed to allow the order.
    assert execute_trade(**other)["status"] == "submitted"


@pytest.mark.unit
def test_sell_is_allowed_after_same_day_buy(tmp_path):
    client = _Client(position={"quantityAvailableForTrading": 3.333, "currentPrice": 30})
    assert execute_trade(**_opts(tmp_path, client))["status"] == "submitted"

    sold = execute_trade(**_opts(tmp_path, client, action="Sell"))

    assert sold["status"] == "submitted"
    assert client.orders[-1][1] == -3.333
    assert execute_trade(**_opts(tmp_path, client, action="Sell"))["status"] == "skipped"


@pytest.mark.unit
def test_executor_uses_the_portfolio_managers_typed_execution(tmp_path, monkeypatch):
    """The PM is the final decision; its levels and trade_now gate the order,
    not the Trader proposal that the debate may have overruled."""
    from cli import run
    from tradingagents.trend import TrendState

    captured = {}

    def fake_execute(**kwargs):
        captured.update(kwargs)
        return {"status": "submitted"}

    monkeypatch.setattr(
        run, "trend_state",
        lambda *args, **kwargs: TrendState(
            "AAPL", "2026-01-01", 100.0, 99.0, 98.0, 0.01, 0.02, "up"
        ),
    )
    monkeypatch.setattr(run, "execute_trade", fake_execute)
    result = run._execute_trading212(
        {
            "trade_mode": "practice",
            "t212_max_buy_value": 100,
            "t212_max_daily_buy_value": 0,
            "data_cache_dir": str(tmp_path),
        },
        {"asset_type": "stock", "ticker": "AAPL", "analysis_date": "2026-01-01"},
        {
            "final_rating": "Buy",
            "final_execution": {
                "trade_now": True,
                "order_type": "market",
                "limit_price": 105,
                "entry_price": 100,
                "stop_loss": 95,
            },
            "trader_proposal": {"action": "Sell", "limit_price": 90},
        },
    )

    assert result["status"] == "submitted"
    assert captured["action"] == "Buy"
    assert captured["limit_price"] == 105
    assert captured["stop_loss"] == 95
    assert captured["order_type"] == "market"
    assert captured["rest_at_entry"] is False


@pytest.mark.unit
def test_executor_places_nothing_when_the_pm_says_wait(monkeypatch):
    from cli import run
    from tradingagents.trend import TrendState

    called = []
    monkeypatch.setattr(
        run, "trend_state",
        lambda *args, **kwargs: TrendState(
            "AAPL", "2026-01-01", 100.0, 99.0, 98.0, 0.01, 0.02, "up"
        ),
    )
    monkeypatch.setattr(run, "execute_trade", lambda **kwargs: called.append(kwargs))
    result = run._execute_trading212(
        {"trade_mode": "practice"},
        {"asset_type": "stock", "ticker": "AAPL", "analysis_date": "2026-01-01"},
        {
            "final_rating": "Overweight",
            "final_execution": {"trade_now": False, "order_type": "limit", "limit_price": 100},
        },
    )

    assert result["status"] == "skipped"
    assert "wait" in result["reason"].lower()
    assert called == []


@pytest.mark.unit
def test_executor_skips_a_buy_against_a_downtrend(monkeypatch):
    from cli import run
    from tradingagents.trend import TrendState

    called = []
    monkeypatch.setattr(
        run, "trend_state",
        lambda *args, **kwargs: TrendState(
            "AAPL", "2026-01-01", 90.0, 99.0, 98.0, -0.02, -0.03, "down"
        ),
    )
    monkeypatch.setattr(run, "execute_trade", lambda **kwargs: called.append(kwargs))
    result = run._execute_trading212(
        {"trade_mode": "practice"},
        {"asset_type": "stock", "ticker": "AAPL", "analysis_date": "2026-01-01"},
        {
            "final_rating": "Buy",
            "final_execution": {"trade_now": True, "limit_price": 105, "entry_price": 100},
        },
    )

    assert result["status"] == "skipped"
    assert "Trend overlay" in result["reason"]
    assert result["adjusted_rating"] == "Underweight"
    assert called == []


@pytest.mark.unit
def test_executor_skips_a_sell_against_an_uptrend(monkeypatch):
    from cli import run
    from tradingagents.trend import TrendState

    called = []
    monkeypatch.setattr(
        run, "trend_state",
        lambda *args, **kwargs: TrendState(
            "AAPL", "2026-01-01", 110.0, 99.0, 98.0, 0.02, 0.03, "up"
        ),
    )
    monkeypatch.setattr(run, "execute_trade", lambda **kwargs: called.append(kwargs))
    result = run._execute_trading212(
        {"trade_mode": "practice"},
        {"asset_type": "stock", "ticker": "AAPL", "analysis_date": "2026-01-01"},
        {
            "final_rating": "Sell",
            "final_execution": {"trade_now": True, "order_type": "market"},
        },
    )

    assert result["status"] == "skipped"
    assert "Trend overlay" in result["reason"]
    assert result["adjusted_rating"] == "Hold"
    assert called == []


@pytest.mark.unit
def test_completed_report_guard_matches_only_the_same_day(tmp_path):
    from cli.run import _completed_report

    execution = tmp_path / "reports" / "AAPL_20260101_120000" / "6_execution"
    execution.mkdir(parents=True)
    (execution / "execution.md").write_text("{}", encoding="utf-8")

    assert _completed_report({"results_dir": str(tmp_path)}, "AAPL", "2026-01-01") is not None
    assert _completed_report({"results_dir": str(tmp_path)}, "AAPL", "2026-01-02") is None


@pytest.mark.unit
def test_client_exposes_account_pending_and_fill_details():
    summary_session = _Session({
        "totalValue": 100.0,
        "currency": "GBP",
        "cash": {"availableToTrade": 10.0},
        "investments": {"currentValue": 90.0},
    })
    client = Trading212Client("practice", "k", "s", session=summary_session)
    assert client.account_summary()["totalValue"] == 100.0
    assert client.account_details() == {"currency": "GBP", "available_to_trade": 10.0}

    pending_session = _Session([{"id": 1, "ticker": "AAPL_US_EQ", "status": "NEW"}])
    pending = Trading212Client("practice", "k", "s", session=pending_session)
    assert pending.pending_orders() == [{"id": 1, "ticker": "AAPL_US_EQ", "status": "NEW"}]

    details_session = _Session({
        "items": [{"order": {"id": 2, "status": "FILLED"}, "fill": {"price": 12.5}}]
    })
    details = Trading212Client("practice", "k", "s", session=details_session)
    assert details.order_history_details() == [
        {"order": {"id": 2, "status": "FILLED"}, "fill": {"price": 12.5}}
    ]
    assert details.order_history() == [{"id": 2, "status": "FILLED"}]
