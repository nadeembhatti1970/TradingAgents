"""Extra news sources for the Sentiment Analyst: Marketaux and Google News RSS.

Marketaux scores each article's sentiment per entity and covers LSE symbols
(``AZN.L``); it needs ``MARKETAUX_API_KEY``. Google News RSS needs no key and
adds UK headlines. Both return a plaintext block and never raise, so a failed
source reads as an "unavailable" placeholder in the prompt.
"""

from __future__ import annotations

import json
import logging
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from tradingagents.dataflows.date_window import in_window

logger = logging.getLogger(__name__)

_MARKETAUX = "https://api.marketaux.com/v1/news/all"
_GOOGLE_NEWS = "https://news.google.com/rss/search"
_UA = "Mozilla/5.0 (compatible; tradingagents)"
_MAX_ITEMS = 15
# Marketaux match_score is 0-100; below this an article only mentions the stock in passing.
_MIN_RELEVANCE = 50.0


def _get(url: str, timeout: float) -> bytes:
    with urlopen(Request(url, headers={"User-Agent": _UA}), timeout=timeout) as resp:
        return resp.read()


def _label(score: float) -> str:
    if score >= 0.25:
        return "bullish"
    if score <= -0.25:
        return "bearish"
    return "neutral"


def fetch_marketaux_news(
    ticker: str, start_date: str, end_date: str, timeout: float = 15.0
) -> str:
    """Scored news for ``ticker`` from Marketaux, with per-article entity sentiment."""
    token = os.environ.get("MARKETAUX_API_KEY")
    if not token:
        return "<marketaux unavailable: MARKETAUX_API_KEY is not set>"
    symbol = ticker.strip().upper()
    end_exclusive = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    query = urlencode({
        "symbols": symbol,
        "filter_entities": "true",
        "language": "en",
        "limit": 50,
        "published_after": f"{start_date}T00:00",
        "published_before": f"{end_exclusive}T00:00",
        "api_token": token,
    })
    try:
        data = json.loads(_get(f"{_MARKETAUX}?{query}", timeout))
    except (OSError, URLError, ValueError) as exc:
        logger.warning("Marketaux fetch failed for %s: %s", symbol, type(exc).__name__)
        return f"<marketaux unavailable: {type(exc).__name__}>"
    if not isinstance(data, dict) or "data" not in data:
        return "<marketaux unavailable: unexpected response>"

    rows = []
    for article in data["data"]:
        entity = next(
            (e for e in article.get("entities") or [] if str(e.get("symbol", "")).upper() == symbol),
            None,
        )
        if not entity:
            continue
        score = entity.get("sentiment_score")
        match = entity.get("match_score") or 0
        rows.append((float(match), score, article))
    total = len(rows)
    rows = [r for r in rows if r[0] >= _MIN_RELEVANCE]
    if not rows:
        return (
            f"<no relevant Marketaux articles for {symbol} within {start_date}..{end_date} "
            f"({total} fetched, all below relevance {_MIN_RELEVANCE:.0f}/100)>"
        )

    # The most relevant articles first.
    rows.sort(key=lambda r: r[0], reverse=True)
    scored = [float(s) for _, s, _ in rows if s is not None]
    head = f"{len(rows)} relevant articles (of {total}; relevance >= {_MIN_RELEVANCE:.0f}/100)"
    if scored:
        avg = sum(scored) / len(scored)
        head += f"; mean entity sentiment {avg:+.2f} ({_label(avg)}, scale -1 to +1)"
    lines = [head]
    for match, score, a in rows[:_MAX_ITEMS]:
        tag = "n/a" if score is None else f"{float(score):+.2f} {_label(float(score))}"
        lines.append(
            f"- [{str(a.get('published_at', ''))[:10]}] {a.get('title', '').strip()} "
            f"({a.get('source', '?')}; sentiment {tag}; relevance {match:.0f}/100)"
        )
    return "\n".join(lines)


def fetch_google_news(
    query: str, start_date: str, end_date: str, timeout: float = 15.0
) -> str:
    """UK-edition Google News headlines for ``query`` within the window."""
    end_exclusive = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    params = urlencode({
        "q": f"{query} after:{start_date} before:{end_exclusive}",
        "hl": "en-GB",
        "gl": "GB",
        "ceid": "GB:en",
    })
    try:
        root = ET.fromstring(_get(f"{_GOOGLE_NEWS}?{params}", timeout))
    except (OSError, URLError, ET.ParseError) as exc:
        logger.warning("Google News fetch failed for %s: %s", query, type(exc).__name__)
        return f"<google news unavailable: {type(exc).__name__}>"

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    lines = []
    for item in root.iter("item"):
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "")
        except (TypeError, ValueError):
            published = None
        if not in_window(published, start_dt, end_dt):
            continue
        title = (item.findtext("title") or "").strip()
        if title:
            day = published.strftime("%Y-%m-%d") if published else "?"
            lines.append(f"- [{day}] {title}")
        if len(lines) >= _MAX_ITEMS:
            break
    if not lines:
        return f"<no Google News headlines for '{query}' within {start_date}..{end_date}>"
    return "\n".join(lines)
