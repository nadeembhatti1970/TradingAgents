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
from tradingagents.trend import (
    TrendState,
    apply_trend_overlay,
    rating_weight,
    trend_state,
)

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
class BasketScore:
    label: str
    n: int
    weight: float
    mean_raw: float
    mean_alpha: float
    mean_net: float


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
    original_rating: str | None = None
    trend: str | None = None
    target_weight: float | None = None


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

    def basket_scores(self) -> dict[str, BasketScore]:
        """Equal-weight, target-weight, long-only and avoid baskets."""
        scored = self.scored()
        if not scored:
            return {}

        def make(label: str, rows: list[ScoreRow], weighted: bool) -> BasketScore | None:
            if not rows:
                return None
            weights = [
                (r.target_weight if r.target_weight is not None else rating_weight(r.rating))
                if weighted else 1.0
                for r in rows
            ]
            total = sum(weights)
            if total <= 0:
                return None

            def mean(attribute: str) -> float:
                pairs = [
                    (weight, getattr(row, attribute))
                    for weight, row in zip(weights, rows, strict=True)
                    if getattr(row, attribute) is not None
                ]
                if not pairs:
                    return 0.0
                return sum(weight * value for weight, value in pairs) / sum(
                    weight for weight, _ in pairs
                )

            return BasketScore(
                label=label,
                n=len(rows),
                weight=total,
                mean_raw=mean("raw"),
                mean_alpha=mean("alpha"),
                mean_net=mean("net_alpha"),
            )

        long_rows = [r for r in scored if r.rating in ("Buy", "Overweight", "Hold")]
        avoid_rows = [r for r in scored if r.rating in ("Underweight", "Sell")]
        baskets = {
            "equal": make("Equal-weight", scored, False),
            "target": make("Target-weight", scored, True),
            "long": make("Long-only (Buy/Overweight/Hold)", long_rows, False),
            "avoid": make("Avoid (Underweight/Sell)", avoid_rows, False),
        }
        return {key: value for key, value in baskets.items() if value is not None}


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
    trend_overlay: bool = False
    broker: dict | None = None

    def as_dict(self) -> dict:
        costs = asdict(self.costs)
        costs.pop("by_key", None)
        return {
            "as_of": self.as_of,
            "memory_log": self.memory_log,
            "trend_overlay": self.trend_overlay,
            "windows": [
                {
                    "holding_days": w.holding_days,
                    "scored": len(w.scored()),
                    "pending": w.pending,
                    "errors": w.errors,
                    "unscored": w.unscored,
                    "by_rating": {k: asdict(v) for k, v in w.by_rating().items()},
                    "baskets": {k: asdict(v) for k, v in w.basket_scores().items()},
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
            f"- Trend overlay: {'on' if self.trend_overlay else 'off'}",
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
            baskets = window.basket_scores()
            equal = baskets.get("equal")
            if baskets:
                lines.extend([
                    "| Basket | n | weight | mean alpha | net of costs | vs equal |",
                    "|---|---:|---:|---:|---:|---:|",
                ])
                for key in ("equal", "target", "long", "avoid"):
                    basket = baskets.get(key)
                    if basket is None:
                        continue
                    vs_equal = basket.mean_net - equal.mean_net if equal else 0.0
                    lines.append(
                        f"| {basket.label} | {basket.n} | {basket.weight:.2f} | "
                        f"{basket.mean_alpha:+.2%} | {basket.mean_net:+.2%} | "
                        f"{vs_equal:+.2%} |"
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
    trend_cache: dict[tuple[str, str], TrendState] | None = None,
) -> WindowScore:
    """Score every entry forward by ``holding_days`` trading days.

    When ``trend_cache`` is supplied, the logged rating is passed through the
    deterministic trend overlay and the adjusted rating is used for the basket.
    """
    window = WindowScore(holding_days)
    for entry in entries:
        original_rating = str(entry.get("rating") or RATING_REVIEW)
        if original_rating == RATING_REVIEW:
            window.unscored += 1
            continue
        ticker = str(entry["ticker"])
        date = str(entry["date"])
        trend = trend_cache.get((ticker, date)) if trend_cache is not None else None
        rating = apply_trend_overlay(original_rating, trend)
        target_weight = rating_weight(rating)
        extra = {
            "original_rating": original_rating,
            "trend": trend.state if trend is not None else None,
            "target_weight": target_weight,
        }
        if not entry.get("pending") and _holding_days(entry.get("holding")) == holding_days:
            raw = _pct(entry.get("raw"))
            alpha = _pct(entry.get("alpha"))
            if alpha is not None:
                window.rows.append(ScoreRow(
                    ticker, date, rating, raw, alpha,
                    _net_alpha(alpha, ticker, date, costs),
                    entry.get("resolved"), "log", None, **extra,
                ))
                continue
        try:
            benchmark = resolve_benchmark(ticker, config)
            raw, alpha, _days, resolution = fetch_returns(
                ticker, date, holding_days, benchmark=benchmark
            )
        except Exception as exc:  # a price outage must not hide the other rows
            window.rows.append(ScoreRow(
                ticker, date, rating, source="error", error=str(exc), **extra
            ))
            window.errors += 1
            continue
        if raw is None or alpha is None:
            window.rows.append(ScoreRow(ticker, date, rating, source="pending", **extra))
            window.pending += 1
            continue
        window.rows.append(ScoreRow(
            ticker, date, rating, raw, alpha,
            _net_alpha(alpha, ticker, date, costs),
            resolution, "prices", None, **extra,
        ))
    return window


def evaluate_gate(windows: list[WindowScore]) -> tuple[str, str]:
    """The measurement gate: 30 directional decisions at 5 and 20 days.

    A pass needs the target-weight basket to beat the equal-weight basket and
    be positive net of costs at both horizons, with a direction hit >= 50%.
    """
    by_days = {window.holding_days: window for window in windows}
    five = by_days.get(5)
    twenty = by_days.get(20)
    directional_5 = five.directional() if five else []
    directional_20 = twenty.directional() if twenty else []
    if len(directional_5) < _GATE_DECISIONS or len(directional_20) < _GATE_DECISIONS:
        return "INSUFFICIENT DATA", (
            f"5-day directional decisions {len(directional_5)}/{_GATE_DECISIONS}; "
            f"20-day {len(directional_20)}/{_GATE_DECISIONS}. Keep the measurement phase running."
        )

    def basket(window: WindowScore | None, key: str) -> BasketScore:
        scores = window.basket_scores() if window else {}
        return scores.get(key) or BasketScore("missing", 0, 0.0, 0.0, 0.0, 0.0)

    target_5, equal_5 = basket(five, "target"), basket(five, "equal")
    target_20, equal_20 = basket(twenty, "target"), basket(twenty, "equal")

    def hit_rate(rows: list[ScoreRow]) -> float:
        return sum(
            1 for row in rows if (row.alpha > 0) == (_DIRECTION[row.rating] > 0)
        ) / len(rows)

    hit_5 = hit_rate(directional_5)
    detail = (
        f"5-day target-weight net {target_5.mean_net:+.2%} vs equal-weight "
        f"{equal_5.mean_net:+.2%}; 20-day target-weight net {target_20.mean_net:+.2%} "
        f"vs equal-weight {equal_20.mean_net:+.2%}; 5-day direction hit {hit_5:.0%}. "
        "Need target > equal and > 0 at both horizons, and hit >= 50%."
    )
    if (
        target_5.mean_net <= 0
        or target_20.mean_net <= 0
        or target_5.mean_net < equal_5.mean_net
        or target_20.mean_net < equal_20.mean_net
        or hit_5 < 0.5
    ):
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
    trend_overlay: bool = False,
) -> PerformanceReport:
    """Build the live scoreboard from the memory log and the cost ledger."""
    log = TradingMemoryLog(config)
    entries = log.load_entries()
    costs = scan_costs(config, entries, position_value)
    trend_cache = None
    if trend_overlay:
        trend_cache = {}
        for entry in entries:
            key = (str(entry["ticker"]), str(entry["date"]))
            if key not in trend_cache:
                trend_cache[key] = trend_state(key[0], key[1], config)
    windows = [
        score_window(entries, days, config, costs, trend_cache=trend_cache)
        for days in holding_days
    ]
    gate, detail = evaluate_gate(windows)
    return PerformanceReport(
        as_of=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        memory_log=str(config.get("memory_log_path") or ""),
        windows=windows,
        costs=costs,
        gate=gate,
        gate_detail=detail,
        trend_overlay=trend_overlay,
        broker=scan_broker(config) if broker else None,
    )
