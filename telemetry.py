"""Per-request telemetry: stage latency, pipeline path and token cost.

Split out of Chatbot.py unchanged. Nothing here touches RetrievalChatbot or
ChatbotConfig, which is what makes it safe to import from anywhere — the
dependency runs one way, from the pipeline into this module.
"""
from __future__ import annotations

import json
import os
import sys
import time
from contextvars import ContextVar
from typing import Optional


# ---------------------------------------------------------------------------
# Per-request telemetry: stage latency, pipeline path, token cost.
# ---------------------------------------------------------------------------
_ACTIVE_TELEMETRY: ContextVar[Optional[dict]] = ContextVar("active_telemetry", default=None)

# USD per 1M tokens, paid tier, from the published Gemini API pricing
# (ai.google.dev/gemini-api/docs/pricing, checked 2026-09-05). Override per
# deployment with LLM_PRICE_TABLE_JSON, e.g.
# {"gemini-3.1-flash-lite": {"input": 0.25, "output": 1.5}}
#
# These were previously a flat 0.10/0.40 for every lite model, which
# understated output on gemini-3.5-flash-lite by 6.25x and made the dashboard's
# cost figure meaningless. Thinking tokens bill at the output rate, which
# estimate_call_cost already does.
_DEFAULT_LLM_PRICES: dict[str, dict[str, float]] = {
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50, "cached": 0.03},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00, "cached": 0.15},
    # 3.1-flash promotional rates run through 2026-12-31, then double.
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50, "cached": 0.025},
    "gemini-3.1-flash": {"input": 0.75, "output": 3.75, "cached": 0.075},
    # 3.1-pro: the higher tier applies to prompts over 200k tokens.
    "gemini-3.1-pro": {"input": 2.00, "output": 12.00, "cached": 0.20},
    "gemma": {"input": 0.0, "output": 0.0},
}


def llm_price_table() -> dict[str, dict[str, float]]:
    table = {name: dict(prices) for name, prices in _DEFAULT_LLM_PRICES.items()}
    raw = os.getenv("LLM_PRICE_TABLE_JSON", "").strip()
    if raw:
        try:
            overrides = json.loads(raw)
        except json.JSONDecodeError:
            overrides = {}
        for model, prices in (overrides or {}).items():
            if isinstance(prices, dict):
                entry = {
                    "input": float(prices.get("input", 0.0) or 0.0),
                    "output": float(prices.get("output", 0.0) or 0.0),
                }
                if prices.get("cached") is not None:
                    entry["cached"] = float(prices.get("cached") or 0.0)
                table[str(model)] = entry
    return table


def price_for_model(model: str) -> Optional[dict[str, float]]:
    table = llm_price_table()
    if model in table:
        return table[model]
    for name, prices in table.items():
        if model.startswith(name) or name in model:
            return prices
    return None


def reset_request_telemetry() -> dict:
    """Start a fresh telemetry record for the current request/answer call."""
    telemetry = {"steps": [], "started_at": time.perf_counter()}
    _ACTIVE_TELEMETRY.set(telemetry)
    return telemetry


def record_pipeline_step(step: str, kind: str, latency_ms: float, **detail) -> None:
    telemetry = _ACTIVE_TELEMETRY.get()
    if telemetry is None:
        return
    record = {"step": step, "kind": kind, "latency_ms": round(float(latency_ms), 2)}
    record.update({key: value for key, value in detail.items() if value is not None})
    telemetry["steps"].append(record)


def _usage_counts(usage: object) -> dict[str, int]:
    def count(*names: str) -> int:
        for name in names:
            value = getattr(usage, name, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(name)
            if value:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return 0

    input_tokens = count("prompt_token_count", "input_tokens")
    output_tokens = count("candidates_token_count", "output_tokens")
    thinking_tokens = count("thoughts_token_count", "thinking_tokens")
    cached_tokens = count("cached_content_token_count", "cached_tokens")
    total_tokens = count("total_token_count", "total_tokens") or (
        input_tokens + output_tokens + thinking_tokens
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": total_tokens,
    }


def estimate_call_cost(model: str, counts: dict[str, int]) -> Optional[float]:
    prices = price_for_model(model)
    if prices is None:
        return None
    billed_output = counts.get("output_tokens", 0) + counts.get("thinking_tokens", 0)
    # prompt_token_count already includes cached tokens, which bill at a lower
    # rate. Default the cached rate to a quarter of the input rate unless the
    # deployment configures one explicitly.
    cached_tokens = min(counts.get("cached_tokens", 0), counts.get("input_tokens", 0))
    fresh_input = counts.get("input_tokens", 0) - cached_tokens
    cached_rate = prices.get("cached", prices.get("input", 0.0) * 0.25)
    cost = (
        fresh_input * prices.get("input", 0.0)
        + cached_tokens * cached_rate
        + billed_output * prices.get("output", 0.0)
    ) / 1_000_000
    return round(cost, 8)


def record_llm_call(*, model: str, stage: str, usage: object, latency_ms: float, streamed: bool) -> None:
    counts = _usage_counts(usage)
    record_pipeline_step(
        stage,
        "llm",
        latency_ms,
        model=model,
        streamed=streamed,
        cost_usd=estimate_call_cost(model, counts),
        priced=price_for_model(model) is not None,
        **counts,
    )


def _caller_stage(default: str = "llm_call") -> str:
    """Label an LLM call by the pipeline function that issued it."""
    frame = sys._getframe(1)
    for _ in range(6):
        frame = frame.f_back
        if frame is None:
            return default
        name = frame.f_code.co_name
        if name.startswith("_") or name in {
            "call_gemini", "call_gemini_stream", "<lambda>", "record_llm_call",
        }:
            continue
        return name
    return default


def summarize_request_telemetry(response_mode: str = "", query_route: Optional[dict] = None) -> dict:
    """Roll the recorded steps into dashboard-ready latency/path/cost fields."""
    telemetry = _ACTIVE_TELEMETRY.get() or {"steps": []}
    steps = list(telemetry.get("steps", []))

    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "call_count": 0,
    }
    by_model: dict[str, dict] = {}
    fully_priced = True
    for step in steps:
        if step.get("kind") != "llm":
            continue
        totals["call_count"] += 1
        for key in ("input_tokens", "output_tokens", "thinking_tokens", "cached_tokens", "total_tokens"):
            totals[key] += int(step.get(key, 0) or 0)
        cost = step.get("cost_usd")
        if cost is None:
            fully_priced = False
        else:
            totals["cost_usd"] += float(cost)
        model_bucket = by_model.setdefault(
            str(step.get("model", "unknown")),
            {"calls": 0, "total_tokens": 0, "cost_usd": 0.0},
        )
        model_bucket["calls"] += 1
        model_bucket["total_tokens"] += int(step.get("total_tokens", 0) or 0)
        model_bucket["cost_usd"] = round(model_bucket["cost_usd"] + float(cost or 0.0), 8)

    totals["cost_usd"] = round(totals["cost_usd"], 8)
    totals["fully_priced"] = fully_priced
    totals["by_model"] = by_model

    latency_breakdown = {"retrieval_ms": 0.0, "llm_ms": 0.0, "other_ms": 0.0}
    for step in steps:
        bucket = {
            "retrieval": "retrieval_ms",
            "llm": "llm_ms",
        }.get(str(step.get("kind")), "other_ms")
        latency_breakdown[bucket] += float(step.get("latency_ms", 0.0) or 0.0)
    latency_breakdown = {key: round(value, 2) for key, value in latency_breakdown.items()}

    route = query_route or {}
    path_steps = [
        {
            "step": step.get("step", ""),
            "kind": step.get("kind", ""),
            "latency_ms": step.get("latency_ms", 0.0),
            "total_tokens": step.get("total_tokens"),
            "model": step.get("model"),
        }
        for step in steps
    ]
    label_parts = [str(step.get("step", "")) for step in steps if step.get("step")]
    if response_mode:
        label_parts.append(str(response_mode))
    return {
        "token_usage": totals,
        "llm_calls": [step for step in steps if step.get("kind") == "llm"],
        "stage_timings": steps,
        "latency_breakdown": latency_breakdown,
        "path": path_steps,
        "path_label": " -> ".join(dict.fromkeys(label_parts)) or (response_mode or "direct"),
        "route_summary": {
            "response_mode": response_mode,
            "routing_mode": route.get("routing_mode", ""),
            "question_type": route.get("question_type", ""),
            "prefer_summary": route.get("prefer_summary"),
        },
    }
