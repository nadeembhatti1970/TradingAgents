"""Shared helpers for invoking an agent with structured output and a graceful fallback.

The Portfolio Manager, Trader, and Research Manager all follow the same
canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output (rare; mostly older Ollama models), the
   wrap is skipped and the agent uses free-text generation instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call itself fails for any reason
   (malformed JSON from a weak model, transient provider issue), fall
   back to a plain ``llm.invoke`` so the pipeline never blocks.

Centralising the pattern here keeps the agent factories small and ensures
all three agents log the same warnings when fallback fires.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from langchain_core.tracers.context import collect_runs
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# A structured call that fails (often a model replying in text instead of
# calling the schema tool) is retried this many times before the free-text fallback.
STRUCTURED_ATTEMPTS = 3

# The agent prompts ask for prose sections so the free-text fallback reads well.
# Models such as DeepSeek follow that literally and answer in text without calling
# the schema tool, so structured attempts end by asking for the tool call.
TOOL_CALL_REMINDER = (
    "Deliver your answer by calling the provided tool with every field filled in. "
    "Do not write the sections out as prose; a text reply will be discarded."
)
TOOL_CALL_RETRY_REMINDER = (
    "Your previous reply was plain text and was discarded. "
    "Respond only with a call to the provided tool."
)

T = TypeVar("T", bound=BaseModel)

# Schema-only structured output binds exactly one tool (the schema itself), so a
# model that reaches for a search tool emits an unknown tool call and the whole
# structured attempt is discarded for a free-text retry. Agents on this path
# state the constraint explicitly rather than relying on the binding alone
# (#1130).
NO_EXTERNAL_TOOLS = (
    "Use only the evidence provided in this prompt. Do not call external tools "
    "or search the web; if something is missing, say so explicitly."
)


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Any | None:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None


def _with_reminder(prompt: Any, reminder: str) -> Any:
    """Append ``reminder`` to a string prompt or to the last message of a dict-message list."""
    if isinstance(prompt, str):
        return f"{prompt}\n\n{reminder}"
    if isinstance(prompt, list) and prompt and isinstance(prompt[-1], dict):
        last = prompt[-1]
        if isinstance(last.get("content"), str):
            return [*prompt[:-1], {**last, "content": f"{last['content']}\n\n{reminder}"}]
    return prompt


def _describe_llm_reply(runs: list[Any]) -> str:
    """Summarise what the model actually returned, from the traced LLM child runs."""
    stack = list(runs)
    reply = ""
    while stack:
        run = stack.pop(0)
        stack.extend(getattr(run, "child_runs", None) or [])
        if getattr(run, "run_type", None) != "llm":
            continue
        try:
            message = (run.outputs or {})["generations"][0][0]["message"]
            kwargs = message.get("kwargs", message)
            meta = kwargs.get("response_metadata") or {}
            reply = (
                f"finish_reason={meta.get('finish_reason')} "
                f"model={meta.get('model_name')} "
                f"tool_calls={len(kwargs.get('tool_calls') or [])} "
                f"content={str(kwargs.get('content'))[:300]!r}"
            )
        except (KeyError, IndexError, TypeError, AttributeError):
            reply = "reply unavailable"
    return reply or "no LLM reply recorded"


def invoke_structured(
    structured_llm: Any | None,
    prompt: Any,
    agent_name: str,
    attempts: int = STRUCTURED_ATTEMPTS,
) -> T | None:
    """Run the structured call; ``None`` when there is none or every attempt fails.

    ``prompt`` is whatever the underlying LLM accepts (a string for chat
    invocations, a list of message dicts for chat models that take that
    shape), so a caller can forward the same value to its free-text fallback.
    Every attempt asks for the tool call explicitly, and the free-text fallback
    the caller runs afterwards still uses the original prompt. Each attempt that fails logs what the model actually returned, and the call
    is retried because a model may answer in plain text on one try and call
    the schema tool on the next.
    """
    if structured_llm is None:
        return None
    for attempt in range(1, max(attempts, 1) + 1):
        reminder = TOOL_CALL_REMINDER if attempt == 1 else TOOL_CALL_RETRY_REMINDER
        with collect_runs() as collector:
            try:
                result = structured_llm.invoke(_with_reminder(prompt, reminder))
                if result is None:
                    # A thinking model can answer in plain text instead of calling
                    # the tool, leaving the parser with nothing to return. Treat it
                    # as a structured miss, with a clear reason.
                    raise ValueError("structured output returned no parsed result")
                return result
            except Exception as exc:
                logger.warning(
                    "%s: structured-output attempt %d/%d failed (%s); model reply: %s",
                    agent_name, attempt, attempts, exc,
                    _describe_llm_reply(collector.traced_runs),
                )
    logger.warning("%s: structured output unavailable; falling back to free text", agent_name)
    return None


def invoke_structured_or_freetext(
    structured_llm: Any | None,
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
) -> str:
    """Run the structured call and render to markdown; fall back to free-text on any failure."""
    result = invoke_structured(structured_llm, prompt, agent_name)
    if result is not None:
        return render(result)
    return plain_llm.invoke(prompt).content
