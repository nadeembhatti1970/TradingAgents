"""The live performance scoreboard: forward scoring, costs and the gate."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tradingagents.memory import TradingMemoryLog
from tradingagents.performance import (
    CostSummary,
    PerformanceReport,
    ScoreRow,
    WindowScore,
    build_report,
    evaluate_gate,
    scan_costs,
    score_window,
)


def _costs(**overrides):
    values = {
        "total_usd": 0.2,
        "records": 2,
        "average_usd": 0.1,
        "runs": 2,
        "unique": 2,
        "duplicates": 0,
        "estimated_usd": 0.2,
        "duplicate_usd": 0.0,
        "position_value": 50.0,
        "cost_per_position": 0.002,
        "round_trip": 0.007,
        "note": "",
        "by_key": {},
    }
    values.update(overrides)
    return CostSummary(**values)


def _window(days, count, *, rating="Overweight", alpha=0.01, net=0.003):
    rows = [
        ScoreRow("AAPL", "2026-01-01", rating, alpha, alpha, net)
        for _ in range(count)
    ]
    return WindowScore(days, rows=rows)


@pytest.mark.unit
def test_score_window_uses_the_stored_outcome_and_fetches_the_rest(monkeypatch):
    entries = [
        {
            "ticker": "AAPL", "date": "2026-01-01", "rating": "Overweight",
            "pending": False, "raw": "+2.0%", "alpha": "+1.0%", "holding": "5d",
            "resolved": "2026-01-08",
        },
        {"ticker": "MSFT", "date": "2026-01-02", "rating": "Underweight", "pending": True},
    ]

    def fake_fetch(ticker, date, holding_days, benchmark="SPY"):
        assert holding_days == 5
        return -0.02, -0.03, 5, "2026-01-09"

    monkeypatch.setattr("tradingagents.performance.fetch_returns", fake_fetch)
    costs = _costs(by_key={("AAPL", "2026-01-01"): 0.1, ("MSFT", "2026-01-02"): 0.1})
    window = score_window(entries, 5, {"benchmark_map": {"": "SPY"}}, costs)

    assert len(window.scored()) == 2
    aapl = next(row for row in window.rows if row.ticker == "AAPL")
    assert aapl.source == "log" and aapl.alpha == 0.01
    assert aapl.net_alpha == pytest.approx(0.01 - 0.007 - 0.1 / 50)
    msft = next(row for row in window.rows if row.ticker == "MSFT")
    assert msft.source == "prices" and msft.alpha == -0.03
    assert window.pending == 0


@pytest.mark.unit
def test_score_window_keeps_a_window_without_a_price_as_pending(monkeypatch):
    monkeypatch.setattr(
        "tradingagents.performance.fetch_returns",
        lambda *args, **kwargs: (None, None, None, None),
    )
    window = score_window(
        [{"ticker": "AAPL", "date": "2026-01-01", "rating": "Buy", "pending": True}],
        5,
        {"benchmark_map": {"": "SPY"}},
        _costs(),
    )
    assert window.pending == 1
    assert window.rows[0].source == "pending"


@pytest.mark.unit
def test_window_by_rating_reports_net_alpha_and_direction_hit():
    window = WindowScore(5, rows=[
        ScoreRow("AAPL", "2026-01-01", "Overweight", 0.02, 0.01, 0.003),
        ScoreRow("MSFT", "2026-01-01", "Underweight", -0.02, -0.01, -0.017),
        ScoreRow("NVDA", "2026-01-01", "Hold", 0.01, 0.005, -0.002),
    ])
    scores = window.by_rating()

    assert scores["Overweight"].hit_rate == 1.0
    assert scores["Underweight"].hit_rate == 1.0
    assert scores["Hold"].hit_rate is None
    assert scores["Overweight"].mean_net == pytest.approx(0.003)


@pytest.mark.unit
def test_gate_needs_thirty_directional_decisions_at_both_horizons():
    assert evaluate_gate([_window(5, 29), _window(20, 30)])[0] == "INSUFFICIENT DATA"
    assert evaluate_gate([_window(5, 30), _window(20, 29)])[0] == "INSUFFICIENT DATA"


@pytest.mark.unit
def test_gate_fails_on_negative_net_or_less_than_half_the_directions():
    assert evaluate_gate([_window(5, 30, net=-0.01), _window(20, 30)])[0] == "FAIL"
    assert evaluate_gate([_window(5, 30), _window(20, 30, net=-0.01)])[0] == "FAIL"
    assert evaluate_gate([_window(5, 30, alpha=-0.01), _window(20, 30)])[0] == "FAIL"


@pytest.mark.unit
def test_gate_passes_with_positive_net_and_a_direction_hit():
    gate, detail = evaluate_gate([_window(5, 30), _window(20, 30)])
    assert gate == "PASS"
    assert "net alpha" in detail


@pytest.mark.unit
def test_scan_costs_counts_duplicate_runs_and_the_position_cost(tmp_path):
    (tmp_path / "costs.jsonl").write_text(
        '{"ticker":"AAPL","date":"2026-01-01","cost_usd":0.2}\n'
        '{"ticker":"MSFT","date":"2026-01-01","cost_usd":0.4}\n',
        encoding="utf-8",
    )
    for name in ("AAPL_20260101_1", "AAPL_20260101_2", "MSFT_20260101_1"):
        (tmp_path / "reports" / name).mkdir(parents=True)

    costs = scan_costs(
        {"results_dir": str(tmp_path), "t212_max_buy_value": 50},
        [{"ticker": "AAPL"}, {"ticker": "MSFT"}],
    )

    assert costs.records == 2
    assert costs.total_usd == pytest.approx(0.6)
    assert costs.average_usd == pytest.approx(0.3)
    assert costs.runs == 3
    assert costs.unique == 2
    assert costs.duplicates == 1
    assert costs.estimated_usd == pytest.approx(0.9)
    assert costs.duplicate_usd == pytest.approx(0.3)
    assert costs.position_value == 50
    assert costs.cost_per_position == pytest.approx(0.006)
    assert costs.cost_for("AAPL", "2026-01-01") == pytest.approx(0.2)
    assert costs.cost_for("UNKNOWN", "2026-01-01") == pytest.approx(0.3)


@pytest.mark.unit
def test_build_report_scores_the_live_log_and_serializes(tmp_path, monkeypatch):
    memory = tmp_path / "memory.md"
    log = TradingMemoryLog({"memory_log_path": str(memory)})
    log.store_decision(
        "AAPL", "2026-01-01", "**Rating**: Overweight\n\nA decision.", "Overweight"
    )
    monkeypatch.setattr(
        "tradingagents.performance.fetch_returns",
        lambda *args, **kwargs: (0.01, 0.005, 1, "2026-01-02"),
    )
    config = {
        "memory_log_path": str(memory),
        "results_dir": str(tmp_path),
        "benchmark_map": {"": "SPY"},
        "t212_max_buy_value": 50,
    }

    report = build_report(config, (1, 5, 20))

    assert [window.holding_days for window in report.windows] == [1, 5, 20]
    assert report.windows[0].scored()[0].ticker == "AAPL"
    assert report.gate == "INSUFFICIENT DATA"
    assert "Gate" in report.render()
    assert json.dumps(report.as_dict())  # the JSON shape is serializable


@pytest.mark.unit
def test_render_reports_a_broker_that_could_not_be_read():
    report = PerformanceReport(
        as_of="2026-11-09 13:00 UTC",
        memory_log="memory.md",
        windows=[],
        costs=_costs(position_value=0.0, cost_per_position=None),
        gate="INSUFFICIENT DATA",
        gate_detail="Not enough decisions yet.",
        broker={"error": "T212_API_KEY is not set"},
    )
    assert "Unavailable: T212_API_KEY is not set" in report.render()


@pytest.mark.unit
def test_performance_command_prints_json_and_saves_the_report(tmp_path, monkeypatch):
    from cli import main as cli_main

    report = PerformanceReport(
        as_of="2026-11-09 13:00 UTC",
        memory_log=str(tmp_path / "memory.md"),
        windows=[],
        costs=_costs(position_value=0.0, cost_per_position=None),
        gate="INSUFFICIENT DATA",
        gate_detail="Not enough decisions yet.",
        broker=None,
    )
    seen = {}

    def fake_build(config, windows, position_value=None, broker=False):
        seen.update(config=config, windows=windows, position_value=position_value, broker=broker)
        return report

    monkeypatch.setattr(cli_main, "build_report", fake_build)
    result = CliRunner().invoke(cli_main.app, [
        "performance", "--json", "--memory-log", str(tmp_path / "memory.md"),
        "--results-dir", str(tmp_path), "--save",
    ])

    assert result.exit_code == 0, result.output
    assert "INSUFFICIENT DATA" in result.output
    assert seen["windows"] == (1, 5, 20)
    assert seen["config"]["memory_log_path"] == str(tmp_path / "memory.md")
    assert seen["config"]["results_dir"] == str(tmp_path)
    assert (tmp_path / "performance.md").is_file()
