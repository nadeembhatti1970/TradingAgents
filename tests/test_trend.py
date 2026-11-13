"""Deterministic trend/relative-strength overlay and target weights."""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.trend import (
    TrendState,
    apply_trend_overlay,
    rating_weight,
    trend_state,
)


def _series(start: float, step: float, n: int = 260):
    return pd.Series(
        [start + step * i for i in range(n)],
        index=pd.date_range("2025-01-01", periods=n, freq="D"),
    )


def _state(state, *, price=100.0, sma50=99.0, sma200=98.0, rs20=0.01, rs60=0.02):
    return TrendState("AAPL", "2026-01-01", price, sma50, sma200, rs20, rs60, state)


@pytest.mark.unit
def test_uptrend_is_up_and_not_weak(monkeypatch):
    def fake(ticker, start, end):
        return _series(100, 0) if ticker == "^FTSE" else _series(50, 0.5)

    monkeypatch.setattr("tradingagents.trend.get_closes", fake)
    state = trend_state("AAPL", "2026-01-01", {"benchmark_map": {"": "^FTSE"}})

    assert state.state == "up"
    assert state.strong
    assert not state.weak
    assert state.rs20 > 0


@pytest.mark.unit
def test_downtrend_is_down_and_weak(monkeypatch):
    def fake(ticker, start, end):
        return _series(100, 0) if ticker == "^FTSE" else _series(200, -0.5)

    monkeypatch.setattr("tradingagents.trend.get_closes", fake)
    state = trend_state("AAPL", "2026-01-01", {"benchmark_map": {"": "^FTSE"}})

    assert state.state == "down"
    assert state.strong
    assert state.weak


@pytest.mark.unit
def test_data_failure_is_unknown_and_leaves_the_rating_alone(monkeypatch):
    def fake(*args, **kwargs):
        raise RuntimeError("no prices")

    monkeypatch.setattr("tradingagents.trend.get_closes", fake)
    state = trend_state("AAPL", "2026-01-01", {})
    assert state.state == "unknown"
    assert apply_trend_overlay("Buy", state) == "Buy"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rating", "state", "expected"),
    [
        ("Buy", "up", "Buy"),
        ("Buy", "mixed", "Hold"),
        ("Buy", "down", "Underweight"),
        ("Overweight", "up", "Overweight"),
        ("Overweight", "down", "Underweight"),
        ("Hold", "up", "Hold"),
        ("Hold", "down", "Hold"),
        ("Hold", "mixed", "Hold"),
        ("Underweight", "up", "Hold"),
        ("Underweight", "down", "Underweight"),
        ("Sell", "up", "Hold"),
        ("Sell", "down", "Sell"),
    ],
)
def test_overlay_only_downgrades_a_buy_or_lifts_an_avoid(rating, state, expected):
    assert apply_trend_overlay(rating, _state(state)) == expected


@pytest.mark.unit
def test_a_hold_below_the_50_day_is_downgraded():
    trend = _state("mixed", price=90.0, sma50=99.0, sma200=98.0)
    assert trend.weak
    assert apply_trend_overlay("Hold", trend) == "Underweight"


@pytest.mark.unit
def test_target_weights():
    assert rating_weight("Buy") == 1.5
    assert rating_weight("Overweight") == 1.25
    assert rating_weight("Hold") == 1.0
    assert rating_weight("Underweight") == 0.5
    assert rating_weight("Sell") == 0.0
    assert rating_weight("unknown") == 0.0


@pytest.mark.unit
def test_trend_render_names_the_state_and_levels():
    text = _state("up").render()
    assert "price 100.00" in text
    assert "200-day 98.00" in text
    assert "state up" in text
