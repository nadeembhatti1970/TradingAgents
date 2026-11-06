"""Jev post screening, against TypeSafe's documented request and response shapes."""

import pytest
import requests

from tradingagents.agents import post_screen as typesafe

QUESTIONS = {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}
ANSWERS = {"is_urgent": {"type": "noul", "noul": 0.95}}


class _Response:
    def __init__(self, status, payload=None, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


class _Calls(list):
    def __init__(self):
        super().__init__()
        self.queue = []
        self.sleeps = []


@pytest.fixture
def post(monkeypatch):
    """Queue responses on ``.queue``; the list records each call to requests.post."""
    calls = _Calls()
    queue = calls.queue

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(typesafe.requests, "post", fake_post)
    monkeypatch.setattr(typesafe.time, "sleep", calls.sleeps.append)
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    monkeypatch.delenv("TYPESAFE_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    return calls


def _ok():
    return _Response(200, {"model": "jev-1.13.0", "answers": ANSWERS,
                           "usage": {"input_tokens": 296, "output_tokens": 20}})


@pytest.mark.unit
def test_sends_the_documented_request_and_returns_the_answers(post):
    post.queue.append(_ok())

    assert typesafe.system_one("Help! My payouts have been failing.", QUESTIONS) == ANSWERS

    url, kwargs = post[0]
    assert url == "https://api.typesafe.ai/v1/systemone"
    assert kwargs["headers"]["Authorization"] == "Bearer ts-test"
    assert kwargs["json"] == {"state": "Help! My payouts have been failing.",
                              "model": "jev-latest", "questions": QUESTIONS}


@pytest.mark.unit
def test_the_model_follows_the_sdk_environment(post, monkeypatch):
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-1.13.0")
    post.queue.append(_ok())

    typesafe.system_one("s", QUESTIONS)

    assert post[0][1]["json"]["model"] == "jev-1.13.0"


@pytest.mark.unit
@pytest.mark.parametrize("base", ["https://openrouter.ai/api", "https://openrouter.ai/api/"])
def test_the_endpoint_follows_the_sdk_environment(post, monkeypatch, base):
    """TypeSafe's SDKs read TYPESAFE_BASE_URL and append /v1/systemone; OpenRouter
    serves Jev at https://openrouter.ai/api with an OpenRouter key (#1416)."""
    monkeypatch.setenv("TYPESAFE_BASE_URL", base)
    post.queue.append(_ok())

    typesafe.system_one("s", QUESTIONS)

    assert post[0][0] == "https://openrouter.ai/api/v1/systemone"


@pytest.mark.unit
def test_a_base_url_without_a_key_does_not_turn_screening_on(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://openrouter.ai/api")
    assert typesafe.jev_screen("AAPL") is None


@pytest.mark.unit
@pytest.mark.parametrize("transient", [
    _Response(429), _Response(529), requests.ConnectionError(), requests.Timeout(),
    requests.exceptions.ChunkedEncodingError(),
])
def test_rate_limits_overload_and_dropped_connections_are_retried(post, transient):
    post.queue.extend([transient, _ok()])

    assert typesafe.system_one("s", QUESTIONS) == ANSWERS
    assert len(post) == 2


@pytest.mark.unit
def test_a_retry_after_header_sets_the_wait(post):
    post.queue.extend([_Response(429, headers={"retry-after": "7"}), _ok()])

    typesafe.system_one("s", QUESTIONS)

    assert post.sleeps == [7.0]


@pytest.mark.unit
def test_a_long_retry_after_is_capped(post):
    post.queue.extend([_Response(529, headers={"retry-after": "600"}), _ok()])

    typesafe.system_one("s", QUESTIONS)

    assert post.sleeps == [typesafe._MAX_WAIT]


@pytest.mark.unit
def test_retries_are_bounded(post):
    post.queue.extend([_Response(529)] * 3)

    with pytest.raises(typesafe.TypeSafeError, match="HTTP 529"):
        typesafe.system_one("s", QUESTIONS)
    assert len(post) == 3


@pytest.mark.unit
@pytest.mark.parametrize("error", [requests.exceptions.InvalidHeader(), requests.exceptions.TooManyRedirects()])
def test_other_request_errors_are_screening_failures_without_retry(post, error):
    post.queue.append(error)

    with pytest.raises(typesafe.TypeSafeError, match=type(error).__name__):
        typesafe.system_one("s", QUESTIONS)
    assert len(post) == 1


@pytest.mark.unit
@pytest.mark.parametrize("status", [401, 422, 500])
def test_other_failures_raise_without_retry(post, status):
    post.queue.append(_Response(status))

    with pytest.raises(typesafe.TypeSafeError, match=f"HTTP {status}"):
        typesafe.system_one("s", QUESTIONS)
    assert len(post) == 1


@pytest.mark.unit
@pytest.mark.parametrize("payload", [
    None, {"model": "jev"}, {"answers": {"other": {}}},
    {"answers": {"is_urgent": {"type": "choice", "choice": "yes"}}},
])
def test_a_response_without_every_answer_is_an_error(post, payload):
    post.queue.append(_Response(200, payload))

    with pytest.raises(typesafe.TypeSafeError, match="malformed"):
        typesafe.system_one("s", QUESTIONS)



def _post_answers(about: float, stance: str = "bullish", confidence: float = 0.9):
    return _Response(200, {"model": "jev-1.13.0", "answers": {
        "about": {"type": "noul", "noul": about},
        "stance": {"type": "choice", "choice": stance, "confidence": confidence,
                   "probabilities": {stance: 1.0}},
    }})


@pytest.fixture
def jev(post, monkeypatch):
    """Answer each post by its text: ``post`` maps text -> response."""
    answers = {}

    def fake_post(url, **kwargs):
        post.append((url, kwargs))
        return answers[kwargs["json"]["state"]["post"]]

    monkeypatch.setattr(typesafe.requests, "post", fake_post)
    monkeypatch.setattr(typesafe, "resolve_instrument_identity",
                        lambda t: {"company_name": "NVIDIA Corporation"})
    return answers


@pytest.mark.unit
def test_no_key_no_screen(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert typesafe.jev_screen("NVDA") is None


@pytest.mark.unit
def test_each_post_is_its_own_state_under_the_fixed_questions(jev, post):
    jev["NVDA to 200"] = _post_answers(0.9)

    typesafe.jev_screen("NVDA")(["NVDA to 200"])

    body = post[0][1]["json"]
    assert body["state"] == {"instrument": "NVIDIA Corporation (NVDA)", "post": "NVDA to 200"}
    assert body["questions"] == typesafe.QUESTIONS


@pytest.mark.unit
def test_screen_drops_clear_off_topic_posts_and_counts_confident_stances(jev):
    jev.update({
        "long NVDA": _post_answers(0.95, "bullish"),
        "NVDA puts": _post_answers(0.9, "bearish"),
        "maybe NVDA": _post_answers(0.4, "neutral"),       # uncertain relevance: kept
        "NVDA?": _post_answers(0.8, "bullish", 0.3),       # uncertain stance: unclear
        "NVDA!": _post_answers(0.8, "sideways"),           # not an option: unclear
        "$AAPL $MSFT $NVDA pump": _post_answers(0.1, "bullish"),
    })

    keep, note = typesafe.jev_screen("NVDA")(list(jev))

    assert keep == [True, True, True, True, True, False]
    assert note == ("Screened by Jev: 5 of the 6 posts fetched are about NVIDIA Corporation (NVDA); "
                    "their stance on its stock: 1 bullish, 1 bearish, 1 neutral, 2 unclear.")


@pytest.mark.unit
def test_one_failed_request_leaves_every_post_unscreened(jev):
    jev.update({"a": _post_answers(0.1), "b": _Response(401)})

    keep, note = typesafe.jev_screen("NVDA")(["a", "b"])

    assert keep == [True, True]
    assert note == "<Jev screening unavailable (HTTP 401); posts are unscreened>"


@pytest.mark.unit
def test_an_answer_missing_its_fields_reads_as_malformed(jev):
    jev["a"] = _Response(200, {"answers": {"about": {"type": "noul"},
                                           "stance": {"type": "choice"}}})

    keep, note = typesafe.jev_screen("NVDA")(["a"])

    assert keep == [True]
    assert note == "<Jev screening unavailable (malformed response); posts are unscreened>"


@pytest.mark.unit
def test_the_first_failure_cancels_the_requests_not_yet_sent(jev, post, monkeypatch):
    monkeypatch.setattr(typesafe, "_WORKERS", 1)
    jev.update({"a": _Response(401), **{f"p{i}": _post_answers(0.9) for i in range(20)}})

    typesafe.jev_screen("NVDA")(list(jev))

    assert len(post) < 21


@pytest.mark.unit
def test_the_sentiment_analyst_hands_the_screen_to_both_social_fetchers(monkeypatch):
    from langchain_core.messages import AIMessage

    from tradingagents.agents.analysts import sentiment_analyst

    screen = object()
    seen = []
    monkeypatch.setattr(sentiment_analyst, "jev_screen", lambda ticker: screen)
    monkeypatch.setattr(sentiment_analyst.get_news, "func", lambda *a: "news")
    for name in ("fetch_stocktwits_messages", "fetch_reddit_posts"):
        monkeypatch.setattr(sentiment_analyst, name, lambda *a, screen=None, **k: seen.append(screen) or "")

    class _LLM:
        def with_structured_output(self, *a, **k):
            raise NotImplementedError

        def invoke(self, messages):
            return AIMessage(content="report")

    node = sentiment_analyst.create_sentiment_analyst(_LLM())
    node({"company_of_interest": "NVDA", "trade_date": "2026-01-09", "messages": []})

    assert seen == [screen, screen]


@pytest.mark.unit
def test_a_failure_does_not_wait_for_requests_still_in_flight(jev, monkeypatch):
    import threading
    import time

    release = threading.Event()

    class _Slow:
        status_code = 200
        headers = {}

        def json(self):
            release.wait(5)
            return _post_answers(0.9).json()

    jev.update({"slow": _Slow(), "bad": _Response(401)})
    started = time.monotonic()
    keep, note = typesafe.jev_screen("NVDA")(["slow", "bad"])
    elapsed = time.monotonic() - started
    release.set()

    assert keep == [True, True] and "unavailable" in note
    assert elapsed < 2


def test_social_sources_can_be_switched_off(monkeypatch):
    from tradingagents.agents.analysts import sentiment_analyst

    monkeypatch.setenv("TRADINGAGENTS_SENTIMENT_SOCIAL", "false")
    assert sentiment_analyst._social_enabled() is False
    monkeypatch.delenv("TRADINGAGENTS_SENTIMENT_SOCIAL")
    assert sentiment_analyst._social_enabled() is True


def test_extra_news_block_and_prompt(monkeypatch):
    from tradingagents.agents.analysts import sentiment_analyst as sa

    monkeypatch.setattr(sa, "resolve_instrument_identity", lambda t: {"company_name": "AstraZeneca"})
    monkeypatch.setattr(sa, "fetch_marketaux_news", lambda *a: "mx")
    monkeypatch.setattr(sa, "fetch_google_news", lambda q, *a: f"gn:{q}")
    block = sa._extra_news_block("AZN.L", "2026-10-29", "2026-11-05")
    assert "mx" in block and 'gn:"AstraZeneca" stock' in block
    msg = sa._build_system_message(
        ticker="AZN.L", start_date="a", end_date="b", news_block="n",
        stocktwits_block="s", reddit_block="r", extra_news_block=block,
    )
    assert "<start_of_extra_news>" in msg and "mx" in msg
    monkeypatch.setenv("TRADINGAGENTS_SENTIMENT_EXTRA_NEWS", "false")
    assert sa._extra_news_block("AZN.L", "a", "b") == ""


def test_extra_news_vendors_parse_and_degrade(monkeypatch):
    import json

    from tradingagents.dataflows.vendors import extra_news as en

    monkeypatch.delenv("MARKETAUX_API_KEY", raising=False)
    assert "not set" in en.fetch_marketaux_news("AZN.L", "2026-10-29", "2026-11-05")
    monkeypatch.setenv("MARKETAUX_API_KEY", "k")
    payload = {"data": [
        {"title": "Strong", "source": "x.com", "published_at": "2026-11-01T00:00:00Z",
         "entities": [{"symbol": "AZN.L", "sentiment_score": 0.5, "match_score": 90}]},
        {"title": "Other", "entities": [{"symbol": "BP.L", "sentiment_score": -1}]},
        {"title": "Passing", "entities": [{"symbol": "AZN.L", "sentiment_score": -0.9, "match_score": 12}]},
    ]}
    monkeypatch.setattr(en, "_get", lambda url, t: json.dumps(payload).encode())
    out = en.fetch_marketaux_news("AZN.L", "2026-10-29", "2026-11-05")
    assert "1 relevant articles (of 2" in out and "+0.50 bullish" in out
    assert "Other" not in out and "Passing" not in out
    rss = (b"<rss><channel><item><title>Headline A</title>"
           b"<pubDate>Wed, 04 Nov 2026 10:00:00 GMT</pubDate></item>"
           b"<item><title>Old</title><pubDate>Wed, 01 Jan 2025 10:00:00 GMT</pubDate></item>"
           b"</channel></rss>")
    monkeypatch.setattr(en, "_get", lambda url, t: rss)
    out = en.fetch_google_news("AstraZeneca", "2026-10-29", "2026-11-05")
    assert "Headline A" in out and "Old" not in out
    monkeypatch.setattr(en, "_get", lambda url, t: (_ for _ in ()).throw(OSError()))
    assert "unavailable" in en.fetch_google_news("x", "2026-10-29", "2026-11-05")


@pytest.mark.unit
def test_stats_handler_sums_openrouter_cost():
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    from cli.stats_handler import StatsCallbackHandler

    handler = StatsCallbackHandler()
    result = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="x"))]],
        llm_output={"token_usage": {"cost": 0.002}},
    )
    handler.on_llm_end(result)
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="y"))]]))
    handler.on_llm_end(result)

    assert handler.get_stats()["cost_usd"] == pytest.approx(0.004)
