"""Forward scoreboard for the live decision log, independent of settlement.

The memory log settles a decision only once its configured holding window has
traded, so a new live system has no visible track record for weeks and shows
only one horizon. This module scores every logged decision forward at several
windows, subtracts the estimated execution and LLM cost, and reports whether
the strategy has passed the measurement gate yet.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

from tradingagents.agents.rating import RATING_REVIEW
from tradingagents.brokers.trading212 import symbol_from_t212
from tradingagents.memory import TradingMemoryLog
from tradingagents.memory.settlement import fetch_returns, resolve_benchmark

_DIRECTION = {"Buy": 1, "Overweight": 1, "Hold": 0, "Underweight": -1, "Sell": -1}
_RATING_ORDER = ("Buy", "Overweight", "Hold", "Underweight", "Sell")
# 0.5% UK stamp duty on the buy plus ~0.2% spread/slippage over a round trip.
_ROUND_TRIP_COST = 0.007
_GATE_DECISIONS = 30


@dataclass
class RatingScore:
    count: int
    mean_raw: float
    mean_alpha: float
    mean_net: float
    hit_rate: float | None


@dataclass
class ScoreRow:
    ticker: str
    date: str
    rating: str
    raw: float | None = None
    alpha: float | None = None
    net_alpha: float | None = None
    resolution: str | None = None
    source: str = "pending"
    error: str | None = None


@dataclass
class WindowScore:
    holding_days: int
    rows: list[ScoreRow] = field(default_factory=list)
    pending: int = 0
    errors: int = 0
    unscored: int = 0

    def scored(self) -> list[ScoreRow]:
        return [r for r in self.rows if r.alpha is not None]

    def directional(self) -> list[ScoreRow]:
        return [r for r in self.scored() if _DIRECTION.get(r.rating, 0) != 0]

    def by_rating(self) -> dict[str, RatingScore]:
        scored = self.scored()
        result: dict[str, RatingScore] = {}
        for rating in _RATING_ORDER:
            group = [r for r in scored if r.rating == rating]
            if not group:
                continue
            alphas = [r.alpha for r in group if r.alpha is not None]
            nets = [r.net_alpha for r in group if r.net_alpha is not None]
            raws = [r.raw for r in group if r.raw is not None]
            direction = _DIRECTION.get(rating, 0)
            hit = (
                sum(1 for r in group if (r.alpha > 0) == (direction > 0)) / len(group)
                if direction else None
            )
            result[rating] = RatingScore(
                count=len(group),
                mean_raw=fmean(raws) if raws else 0.0,
                mean_alpha=fmean(alphas) if alphas else 0.0,
                mean_net=fmean(nets) if nets else 0.0,
                hit_rate=hit,
            )
        return result


@dataclass
class CostSummary:
    total_usd: float
    records: int
    average_usd: float
    runs: int
    unique: int
    duplicates: int
    estimated_usd: float
    duplicate_usd: float
    position_value: float
    cost_per_position: float | None
    round_trip: float
    note: str
    by_key: dict[tuple[str, str], float] = field(default_factory=dict, repr=False)

    def cost_for(self, ticker: str, date: str) -> float:
        return self.by_key.get((ticker, date), self.average_usd)


@dataclass
class PerformanceReport:
    as_of: str
    memory_log: str
    windows: list[WindowScore]
    costs: CostSummary
    gate: str
    gate_detail: str
    broker: dict | None = None

    def as_dict(self) -> dict:
        costs = asdict(self.costs)
        costs.pop("by_key", None)
        return {
            "as_of": self.as_of,
            "memory_log": self.memory_log,
            "windows": [
                {
                    "holding_days": w.holding_days,
                    "scored": len(w.scored()),
                    "pending": w.pending,
                    "errors": w.errors,
                    "unscored": w.unscored,
                    "by_rating": {k: asdict(v) for k, v in w.by_rating().items()},
                    "rows": [asdict(r) for r in w.rows],
                }
                for w in self.windows
            ],
            "costs": costs,
            "gate": self.gate,
            "gate_detail": self.gate_detail,
            "broker": self.broker,
        }

    def render(self) -> str:
        lines = [
            "# Performance scoreboard",
            "",
            f"- As of: {self.as_of}",
            f"- Memory log: {self.memory_log}",
            "",
        ]
        for window in self.windows:
            label = f"{window.holding_days} trading day" + (
                "s" if window.holding_days != 1 else ""
            )
            if window.holding_days == 5:
                label += " — gate horizon"
            elif window.holding_days == 20:
                label += " — strategy horizon"
            lines.extend([
                f"## {label}",
                (
                    f"Scored: {len(window.scored())} · pending: {window.pending} · "
                    f"errors: {window.errors} · unscored: {window.unscored}"
                ),
                "",
            ])
            scores = window.by_rating()
            if not scores:
                lines.extend(["No closed windows yet.", ""])
                continue
            lines.extend([
                "| Rating | n | mean raw | mean alpha | net of costs | direction hit |",
                "|---|---:|---:|---:|---:|---:|",
            ])
            for rating, score in scores.items():
                hit = f"{score.hit_rate:.0%}" if score.hit_rate is not None else "n/a"
                lines.append(
                    f"| {rating} | {score.count} | {score.mean_raw:+.2%} | "
                    f"{score.mean_alpha:+.2%} | {score.mean_net:+.2%} | {hit} |"
                )
            lines.append("")

        lines.extend(["## Gate", f"**{self.gate}** — {self.gate_detail}", ""])
        costs = self.costs
        lines.extend([
            "## Costs",
            (
                f"- Recorded LLM runs: {costs.records}; total ${costs.total_usd:.2f}; "
                f"average ${costs.average_usd:.3f}/run."
            ),
            (
                f"- Run directories: {costs.runs}; unique decisions: {costs.unique}; "
                f"duplicates: {costs.duplicates}."
            ),
            (
                f"- Estimated LLM spend: ${costs.estimated_usd:.2f} for all runs; "
                f"${costs.duplicate_usd:.2f} duplicated."
            ),
        ])
        if costs.cost_per_position is not None:
            lines.append(
                f"- Position value: {costs.position_value:,.2f}; LLM cost per position: "
                f"{costs.cost_per_position:.2%}."
            )
        else:
            lines.append("- Position value: not configured; only the round-trip estimate is applied.")
        lines.extend([
            (
                f"- Round-trip stamp/spread estimate: {costs.round_trip:.2%}; net alpha "
                "in the tables subtracts both this and the LLM share."
            ),
            f"- {costs.note}",
            "",
        ])
        if self.broker is not None:
            lines.extend(self._broker_lines())
        return "\n".join(lines)

    def _broker_lines(self) -> list[str]:
        broker = self.broker or {}
        lines = ["## Broker"]
        if "error" in broker:
            return lines + [f"Unavailable: {broker['error']}", ""]
        summary = broker.get("summary") or {}
        cash = summary.get("cash") or {}
        investments = summary.get("investments") or {}
        currency = summary.get("currency") or ""
        lines.append(
            f"- Account: {summary.get('totalValue')} {currency}; "
            f"cash {cash.get('availableToTrade')}; investments {investments.get('currentValue')}; "
            f"realised {investments.get('realizedProfitLoss')}; "
            f"unrealised {investments.get('unrealizedProfitLoss')}."
        )
        for position in broker.get("positions") or []:
            instrument = position.get("instrument") or {}
            ticker = str(instrument.get("ticker") or "")
            symbol = symbol_from_t212(ticker, instrument.get("currencyCode"))
            lines.append(
                f"- Position: {symbol} {position.get('quantity')} units at "
                f"{position.get('averagePricePaid')} (listed as {ticker})."
            )
        orders = broker.get("orders") or []
        pending = broker.get("pending") or []
        filled = [
            item for item in orders
            if str((item.get("order") or {}).get("status") or "").upper() == "FILLED"
        ]
        rejected = [
            item for item in orders
            if str((item.get("order") or {}).get("status") or "").upper() == "REJECTED"
        ]
        lines.append(
            f"- Orders: {len(orders)} history; {len(filled)} filled, "
            f"{len(rejected)} rejected, {len(pending)} pending."
        )
        for item in filled:
            order = item.get("order") or {}
            fill = item.get("fill") or {}
            impact = fill.get("walletImpact") or {}
            lines.append(
                f"- Filled: {order.get('side')} {order.get('ticker')} "
                f"{order.get('filledQuantity')} @ {fill.get('price')} "
                f"({str(order.get('createdAt') or '')[:10]}); "
                f"realised {impact.get('realisedProfitLoss')}."
            )
        for order in pending:
            lines.append(
                f"- Pending: {order.get('side')} {order.get('ticker')} "
                f"{order.get('quantity')} @ {order.get('limitPrice')}."
            )
        return lines + [""]


def _pct(value) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip().rstrip("%")) / 100
    except (TypeError, ValueError):
        return None


def _holding_days(value) -> int | None:
    if not value:
        return None
    text = str(value).strip().lower().removesuffix("d")
    try:
        return int(text)
    except ValueError:
        return None


def _net_alpha(alpha: float, ticker: str, date: str, costs: CostSummary) -> float:
    extra = 0.0
    if costs.position_value > 0:
        extra = costs.cost_for(ticker, date) / costs.position_value
    return alpha - costs.round_trip - extra


def score_window(
    entries: list[dict],
    holding_days: int,
    config: dict,
    costs: CostSummary,
) -> WindowScore:
    """Score every entry forward by ``holding_days`` trading days."""
    window = WindowScore(holding_days)
    for entry in entries:
        rating = str(entry.get("rating") or RATING_REVIEW)
        if rating == RATING_REVIEW:
            window.unscored += 1
            continue
        ticker = str(entry["ticker"])
        date = str(entry["date"])
        if not entry.get("pending") and _holding_days(entry.get("holding")) == holding_days:
            raw = _pct(entry.get("raw"))
            alpha = _pct(entry.get("alpha"))
            if alpha is not None:
                window.rows.append(ScoreRow(
                    ticker, date, rating, raw, alpha,
                    _net_alpha(alpha, ticker, date, costs),
                    entry.get("resolved"), "log",
                ))
                continue
        try:
            benchmark = resolve_benchmark(ticker, config)
            raw, alpha, _days, resolution = fetch_returns(
                ticker, date, holding_days, benchmark=benchmark
            )
        except Exception as exc:  # a price outage must not hide the other rows
            window.rows.append(ScoreRow(ticker, date, rating, source="error", error=str(exc)))
            window.errors += 1
            continue
        if raw is None or alpha is None:
            window.rows.append(ScoreRow(ticker, date, rating, source="pending"))
            window.pending += 1
            continue
        window.rows.append(ScoreRow(
            ticker, date, rating, raw, alpha,
            _net_alpha(alpha, ticker, date, costs),
            resolution, "prices",
        ))
    return window


def evaluate_gate(windows: list[WindowScore]) -> tuple[str, str]:
    """The measurement gate: 30 directional decisions at both 5 and 20 days."""
    by_days = {window.holding_days: window for window in windows}
    five = by_days.get(5).directional() if by_days.get(5) else []
    twenty = by_days.get(20).directional() if by_days.get(20) else []
    if len(five) < _GATE_DECISIONS or len(twenty) < _GATE_DECISIONS:
        return "INSUFFICIENT DATA", (
            f"5-day directional decisions {len(five)}/{_GATE_DECISIONS}; "
            f"20-day {len(twenty)}/{_GATE_DECISIONS}. Keep the measurement phase running."
        )

    def mean_net(rows: list[ScoreRow]) -> float:
        values = [row.net_alpha for row in rows if row.net_alpha is not None]
        return fmean(values) if values else 0.0

    def hit_rate(rows: list[ScoreRow]) -> float:
        return sum(
            1 for row in rows if (row.alpha > 0) == (_DIRECTION[row.rating] > 0)
        ) / len(rows)

    net_5, net_20, hit_5 = mean_net(five), mean_net(twenty), hit_rate(five)
    detail = (
        f"5-day net alpha {net_5:+.2%}; 20-day net alpha {net_20:+.2%}; "
        f"5-day direction hit {hit_5:.0%}. Need both net alpha values > 0 and hit >= 50%."
    )
    if net_5 <= 0 or net_20 <= 0 or hit_5 < 0.5:
        return "FAIL", detail
    return "PASS", detail


def scan_costs(config: dict, entries: list[dict], position_value: float | None = None) -> CostSummary:
    """Read the cost log and count run directories, including duplicates."""
    results = Path(config["results_dir"])
    by_key: dict[tuple[str, str], float] = {}
    total = 0.0
    records = 0
    costs_path = results / "costs.jsonl"
    if costs_path.is_file():
        for line in costs_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            ticker = str(record.get("ticker") or "")
            date = str(record.get("date") or "")
            try:
                cost = float(record.get("cost_usd") or 0)
            except (TypeError, ValueError):
                continue
            if ticker and date:
                by_key[(ticker, date)] = cost
            total += cost
            records += 1

    reports = results / "reports"
    runs = len([path for path in reports.iterdir() if path.is_dir()]) if reports.is_dir() else 0
    unique = len(entries)
    duplicates = max(0, runs - unique)
    average = total / records if records else 0.0
    value = (
        float(position_value)
        if position_value is not None and position_value > 0
        else float(config.get("t212_max_buy_value") or 0.0)
    )
    cost_per_position = average / value if value > 0 and average > 0 else None
    if records == 0:
        note = "No costs.jsonl records yet; LLM cost is not included in the net figures."
    elif records < runs:
        note = (
            f"Only {records} of {runs} run directories have recorded costs; the estimate "
            "applies their average to every run. LLM costs are USD; positions use the "
            "account currency."
        )
    else:
        note = "LLM costs are USD; positions use the account currency."
    return CostSummary(
        total_usd=total,
        records=records,
        average_usd=average,
        runs=runs,
        unique=unique,
        duplicates=duplicates,
        estimated_usd=average * runs,
        duplicate_usd=average * duplicates,
        position_value=value,
        cost_per_position=cost_per_position,
        round_trip=_ROUND_TRIP_COST,
        note=note,
        by_key=by_key,
    )


def scan_broker(config: dict) -> dict:
    """Best-effort Trading 212 account, position and order snapshot."""
    from tradingagents.brokers.trading212 import Trading212Client

    try:
        client = Trading212Client.from_env(config.get("trade_mode", "practice"))
        return {
            "summary": client.account_summary(),
            "positions": client.positions(),
            "orders": client.order_history_details(),
            "pending": client.pending_orders(),
        }
    except Exception as exc:
        return {"error": str(exc)}


def build_report(
    config: dict,
    holding_days: tuple[int, ...] = (1, 5, 20),
    position_value: float | None = None,
    broker: bool = False,
) -> PerformanceReport:
    """Build the live scoreboard from the memory log and the cost ledger."""
    log = TradingMemoryLog(config)
    entries = log.load_entries()
    costs = scan_costs(config, entries, position_value)
    windows = [score_window(entries, days, config, costs) for days in holding_days]
    gate, detail = evaluate_gate(windows)
    return PerformanceReport(
        as_of=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        memory_log=str(config.get("memory_log_path") or ""),
        windows=windows,
        costs=costs,
        gate=gate,
        gate_detail=detail,
        broker=scan_broker(config) if broker else None,
    )
