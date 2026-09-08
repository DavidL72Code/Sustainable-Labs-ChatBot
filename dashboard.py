"""Chat-event logging and the dashboard payloads built from it.

Split out of Chatbot.py unchanged. Reads settings from config and URL safety
from sanitizers; imports nothing from the pipeline, so the dependency runs one
way — Chatbot imports this, never the reverse.

build_public_trace and build_dashboard_preview are the redaction boundary for
staff-facing views: DASHBOARD_TRACE_MODE decides whether prompts and routing
detail are included, and dashboard_exposes_private_trace is the single place
that decision is made.
"""
from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import CHAT_LOG_PATH, PROJECT_ROOT, ChatbotConfig
from sanitizers import _safe_source_url

from supabase_store import SupabaseStore

# Chatbot.py binds the same name to its own SupabaseStore instance. The class
# only reads environment variables in __init__ and holds no connection, so a
# second instance is equivalent; Chatbot.py calls load_dotenv() before any
# other import, so the environment is populated by the time this runs.
supabase_store = SupabaseStore()


def format_seconds(milliseconds: object) -> str:
    """Render a millisecond duration as seconds for the staff dashboard."""
    try:
        value = float(milliseconds or 0)
    except (TypeError, ValueError):
        value = 0.0
    seconds = value / 1000
    if 0 < seconds < 0.01:
        return "<0.01 s"
    return f"{seconds:.2f} s"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


ADMIN_AUDIT_LOG_PATH = PROJECT_ROOT / "logs" / "admin_events.jsonl"


_PINNED_RETRIEVAL_SCORE = 1_000_000


def _real_score(value: object) -> Optional[float]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return None if score >= _PINNED_RETRIEVAL_SCORE else score


def classify_answer_flags(event: dict) -> list[str]:
    """Decide whether an answer needs staff review, and why.

    Only flagged answers have their question and answer text stored, so this
    is the boundary between "a number in the metrics table" and "a transcript
    a human will read".
    """
    trace = event.get("trace", {}) or {}
    diagnostics = trace.get("retrieval_diagnostics", {}) or {}
    confidence = trace.get("confidence", {}) or {}
    status = str(event.get("status", "") or "")
    reasons: list[str] = []

    if event.get("blocked"):
        reasons.append("blocked")
    if status == "error":
        reasons.append("error")
    if status == "clarification" or event.get("needs_clarification"):
        reasons.append("clarification")
    if confidence.get("is_low_confidence"):
        reasons.append("low_confidence")

    top_score = _real_score(diagnostics.get("top_score"))
    score_gap = _real_score(diagnostics.get("score_gap"))
    min_top = ChatbotConfig.flag_min_top_score
    min_gap = ChatbotConfig.flag_min_score_gap
    if min_top > 0 and top_score is not None and top_score < min_top:
        reasons.append("weak_retrieval")
    if min_gap > 0 and score_gap is not None and score_gap < min_gap:
        reasons.append("narrow_score_gap")
    if status == "answered" and not (event.get("sources") or []):
        reasons.append("no_sources")

    return list(dict.fromkeys(reasons))


def build_chat_metrics_row(event: dict, flags: list[str]) -> dict:
    """A numbers-only row. Never include question or answer text here."""
    trace = event.get("trace", {}) or {}
    diagnostics = trace.get("retrieval_diagnostics", {}) or {}
    confidence = trace.get("confidence", {}) or {}
    telemetry = trace.get("telemetry", {}) or {}
    usage = telemetry.get("token_usage", {}) or {}
    breakdown = telemetry.get("latency_breakdown", {}) or {}
    return {
        "id": event.get("id", ""),
        "created_at": event.get("timestamp") or utc_timestamp(),
        "status": event.get("status") or "answered",
        "response_mode": event.get("response_mode") or "",
        "path_label": telemetry.get("path_label") or "",
        "blocked": bool(event.get("blocked")),
        "needs_clarification": bool(event.get("needs_clarification")),
        "latency_ms": event.get("latency_ms"),
        "retrieval_ms": breakdown.get("retrieval_ms"),
        "llm_ms": breakdown.get("llm_ms"),
        "total_tokens": usage.get("total_tokens"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cost_usd": usage.get("cost_usd"),
        "llm_call_count": usage.get("call_count"),
        "confidence_score": confidence.get("score"),
        "is_low_confidence": bool(confidence.get("is_low_confidence")),
        "top_score": _real_score(diagnostics.get("top_score")),
        "score_gap": _real_score(diagnostics.get("score_gap")),
        "source_count": len(event.get("sources") or []),
        "retrieved_count": len(trace.get("retrieved_metadata") or []),
        "flagged": bool(flags),
        "flag_reasons": flags,
    }


def append_admin_audit_event(action: str, username: str, detail: str = "") -> None:
    """Record who signed in and what they opened.

    The dashboard shows real visitor questions, so each view is attributable to
    a named employee rather than a shared account.
    """
    if not username:
        return
    try:
        ADMIN_AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": utc_timestamp(),
            "action": action,
            "username": username,
        }
        if detail:
            entry["detail"] = detail
        if supabase_store.enabled:
            supabase_store.record_audit_event(
                {
                    "created_at": entry["timestamp"],
                    "username": entry["username"],
                    "action": entry["action"],
                    "detail": entry.get("detail"),
                }
            )
            return
        with ADMIN_AUDIT_LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        # Auditing must never take the dashboard down.
        pass


def append_chat_log(event: dict) -> None:
    event.setdefault("id", uuid.uuid4().hex)
    event.setdefault("timestamp", utc_timestamp())
    flags = classify_answer_flags(event)

    if supabase_store.enabled:
        # Every answer contributes numbers; only a flagged one leaves a
        # transcript behind.
        supabase_store.record_chat_metrics(build_chat_metrics_row(event, flags))
        if flags:
            supabase_store.record_flagged_chat(
                {
                    "id": event.get("id", ""),
                    "created_at": event.get("timestamp") or utc_timestamp(),
                    "conversation_id": event.get("conversation_id", ""),
                    "question": event.get("question", ""),
                    "answer": event.get("answer", ""),
                    "flag_reasons": flags,
                    "sources": event.get("sources", []) or [],
                    "trace": event.get("trace", {}) or {},
                }
            )
        if not ChatbotConfig.chat_log_to_file:
            return

    try:
        CHAT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CHAT_LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"Could not append to the local chat log: {exc}", flush=True)


def load_chat_events(limit: Optional[int] = None) -> list[dict]:
    if not CHAT_LOG_PATH.exists():
        return []

    events: list[dict] = []
    with CHAT_LOG_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    events.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
    return events[:limit] if limit else events


def dashboard_exposes_private_trace() -> bool:
    return ChatbotConfig.dashboard_trace_mode not in {"public", "safe", "redacted"}


def extract_retrieval_scores(trace: dict) -> list[dict]:
    """Per-chunk retrieval scores, newest logs first and older logs degraded safely."""
    trace = trace or {}
    diagnostics = trace.get("retrieval_diagnostics", {}) or {}
    scored = [item for item in (diagnostics.get("selected_scores") or []) if isinstance(item, dict)]
    if not scored:
        for index, metadata in enumerate(trace.get("retrieved_metadata", []) or [], start=1):
            if not isinstance(metadata, dict) or metadata.get("retrieval_score") is None:
                continue
            scored.append(
                {
                    "rank": metadata.get("retrieval_rank", index),
                    "score": metadata.get("retrieval_score"),
                    "title": metadata.get("title", ""),
                    "section_name": metadata.get("section_name", ""),
                    "chunk_index": metadata.get("chunk_index"),
                    "source_path": metadata.get("source_path", ""),
                }
            )
    for item in scored:
        # Recovery candidates are pinned with a sentinel score; label them
        # instead of showing a meaningless six-figure number.
        item["forced"] = float(item.get("score") or 0.0) >= 1_000_000
    return sorted(scored, key=lambda item: item.get("rank") or 0)


def extract_telemetry(trace: dict) -> dict:
    telemetry = (trace or {}).get("telemetry", {}) or {}
    token_usage = telemetry.get("token_usage", {}) or {}
    return {
        "path_label": telemetry.get("path_label", ""),
        "path": telemetry.get("path", []) or [],
        "llm_calls": telemetry.get("llm_calls", []) or [],
        "stage_timings": telemetry.get("stage_timings", []) or [],
        "latency_breakdown": telemetry.get("latency_breakdown", {}) or {},
        "route_summary": telemetry.get("route_summary", {}) or {},
        "token_usage": token_usage,
        "total_tokens": token_usage.get("total_tokens"),
        "cost_usd": token_usage.get("cost_usd"),
    }


def build_public_trace(trace: dict) -> dict:
    """Expose dashboard diagnostics without replaying user prompts or LLM plans."""
    trace = trace or {}
    confidence = trace.get("confidence", {}) or {}
    diagnostics = trace.get("retrieval_diagnostics", {}) or {}
    route = trace.get("query_route", {}) or {}
    safe_metadata = []
    for metadata in trace.get("retrieved_metadata", []) or []:
        if not isinstance(metadata, dict):
            continue
        safe_metadata.append(
            {
                key: metadata.get(key)
                for key in (
                    "title", "category", "folder_label",
                    "retrieval_rank", "retrieval_score", "retrieval_hybrid_score",
                )
                if metadata.get(key) not in (None, "")
            }
        )
    public_diagnostics = {
        key: diagnostics.get(key)
        for key in ("selected_count", "distinct_source_count", "top_score", "second_score", "score_gap")
        if diagnostics.get(key) is not None
    }
    public_diagnostics["selected_scores"] = extract_retrieval_scores(trace)
    return {
        "confidence": confidence,
        # Operational telemetry carries no prompt or answer text, so it stays
        # visible even when message content is hidden.
        "telemetry": trace.get("telemetry", {}) or {},
        "retrieval_diagnostics": public_diagnostics,
        "retrieved_metadata": safe_metadata,
        "query_route": {
            key: route.get(key)
            for key in ("routing_mode", "question_type", "prefer_summary", "target_titles", "target_categories", "target_folders")
            if route.get(key) not in (None, "", [], {})
        },
    }


def build_dashboard_trace(trace: dict) -> dict:
    if dashboard_exposes_private_trace():
        return trace or {}
    return build_public_trace(trace)


def build_dashboard_preview(event: dict) -> str:
    if not dashboard_exposes_private_trace():
        return "User and assistant message content is hidden on the public dashboard."

    question = str(event.get("question", "") or "").strip()
    answer = str(event.get("answer", "") or "").strip()
    if question and answer:
        return f"Q: {question}\nA: {answer}"
    if question:
        return f"Q: {question}"
    if answer:
        return f"A: {answer}"
    return "No message content recorded for this interaction."


def summarize_chat_event(event: dict) -> dict:
    trace = event.get("trace", {}) or {}
    confidence = trace.get("confidence", {}) or {}
    retrieval_diagnostics = trace.get("retrieval_diagnostics", {}) or {}
    sources = event.get("sources", []) or []
    public_sources = [
        {
            "citation": source.get("citation"),
            "title": source.get("title", "Untitled source"),
            "url": _safe_source_url(source.get("url", "")),
        }
        for source in sources
        if isinstance(source, dict)
    ]
    retrieved_metadata = trace.get("retrieved_metadata", []) or []

    summarized = {
        "id": event.get("id", ""),
        "timestamp": event.get("timestamp", ""),
        "status": event.get("status") or "answered",
        "latency_ms": event.get("latency_ms", 0),
        "response_mode": event.get("response_mode", ""),
        "blocked": bool(event.get("blocked")),
        "needs_clarification": bool(event.get("needs_clarification")),
        "display_label": f"Interaction {(event.get('id', '') or 'unknown')[:8]}",
        "preview_text": build_dashboard_preview(event),
    }
    summarized["confidence_score"] = confidence.get("score")
    summarized["is_low_confidence"] = bool(confidence.get("is_low_confidence"))
    summarized["confidence_reasons"] = confidence.get("reasons", []) or []
    summarized["source_count"] = len(sources)
    summarized["retrieved_count"] = len(retrieved_metadata)
    summarized["top_score"] = retrieval_diagnostics.get("top_score")
    summarized["score_gap"] = retrieval_diagnostics.get("score_gap")
    summarized["retrieval_scores"] = extract_retrieval_scores(trace)
    telemetry = extract_telemetry(trace)
    summarized["path_label"] = telemetry["path_label"] or (event.get("response_mode") or "")
    summarized["path"] = telemetry["path"]
    summarized["token_usage"] = telemetry["token_usage"]
    summarized["total_tokens"] = telemetry["total_tokens"]
    summarized["cost_usd"] = telemetry["cost_usd"]
    summarized["latency_breakdown"] = telemetry["latency_breakdown"]
    summarized["sources"] = public_sources
    summarized["trace"] = build_dashboard_trace(trace)
    return summarized


def supabase_event_from_row(row: dict) -> dict:
    """Shape a flagged Supabase row like a local chat log event."""
    metrics = row.get("chat_metrics") or {}
    if isinstance(metrics, list):
        metrics = metrics[0] if metrics else {}
    return {
        "id": row.get("id", ""),
        "conversation_id": row.get("conversation_id", ""),
        "timestamp": row.get("created_at", ""),
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "sources": row.get("sources", []) or [],
        "trace": row.get("trace", {}) or {},
        "status": metrics.get("status") or "answered",
        "response_mode": metrics.get("response_mode") or "",
        "latency_ms": metrics.get("latency_ms") or 0,
        "blocked": bool(metrics.get("blocked")),
        "needs_clarification": bool(metrics.get("needs_clarification")),
        "flag_reasons": row.get("flag_reasons", []) or [],
        "reviewed_by": row.get("reviewed_by") or "",
        "reviewed_at": row.get("reviewed_at") or "",
    }


def build_supabase_dashboard_payload() -> dict:
    """Dashboard fed by Supabase.

    Transcripts exist only for flagged answers, so the history table becomes a
    review queue. Headline numbers come from the content-free metrics table so
    they still describe all traffic rather than only the failures.
    """
    flagged_rows = supabase_store.fetch_flagged_chats(limit=50)
    events = [summarize_chat_event(supabase_event_from_row(row)) for row in flagged_rows]
    for event, row in zip(events, flagged_rows):
        event["flag_reasons"] = row.get("flag_reasons", []) or []
        event["reviewed_by"] = row.get("reviewed_by") or ""

    metrics = supabase_store.fetch_chat_metrics(limit=500)
    daily = supabase_store.fetch_daily_metrics(days=30)

    def total(field: str) -> float:
        return sum(float(row.get(field) or 0) for row in metrics)

    def mean(field: str) -> float:
        values = [float(row[field]) for row in metrics if row.get(field) is not None]
        return round(sum(values) / len(values), 4) if values else 0.0

    source_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    for event in events:
        for source in event.get("sources", []) or []:
            source_counts[source.get("title") or "Unknown source"] += 1
        for metadata in (event.get("trace", {}) or {}).get("retrieved_metadata", []) or []:
            category_counts[metadata.get("category") or metadata.get("folder_label") or "Uncategorized"] += 1

    stats = {
        "total": len(metrics),
        "blocked": sum(1 for row in metrics if row.get("blocked")),
        "clarifications": sum(1 for row in metrics if row.get("needs_clarification")),
        "errors": sum(1 for row in metrics if row.get("status") == "error"),
        "low_confidence": sum(1 for row in metrics if row.get("is_low_confidence")),
        "flagged": sum(1 for row in metrics if row.get("flagged")),
        "total_tokens": int(total("total_tokens")),
        "total_cost_usd": round(total("cost_usd"), 6),
        "avg_tokens": int(mean("total_tokens")),
        "avg_cost_usd": round(mean("cost_usd"), 6),
        "avg_latency_ms": round(mean("latency_ms"), 1),
        "avg_confidence_score": mean("confidence_score"),
        "avg_top_score": mean("top_score"),
    }

    return {
        "storage": "supabase",
        "stats": stats,
        "chat_history": events,
        "recent_events": events[:25],
        "problem_events": events[:12],
        "daily": daily,
        "source_usage": source_counts.most_common(12),
        "category_usage": category_counts.most_common(8),
    }


def build_dashboard_payload() -> dict:
    if supabase_store.enabled:
        return build_supabase_dashboard_payload()
    events = [summarize_chat_event(event) for event in load_chat_events()]
    source_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    problem_events: list[dict] = []

    for event in events:
        trace = event.get("trace", {}) or {}
        if (
            event.get("blocked")
            or event.get("status") in {"error", "clarification"}
            or event.get("is_low_confidence")
        ):
            problem_events.append(event)

        for source in event.get("sources", []) or []:
            title = source.get("title") or source.get("source_path") or "Unknown source"
            source_counts[title] += 1

        for metadata in trace.get("retrieved_metadata", []) or []:
            category = metadata.get("category") or metadata.get("folder_label") or "Uncategorized"
            category_counts[category] += 1

    total_tokens = sum(int(event.get("total_tokens") or 0) for event in events)
    total_cost = round(sum(float(event.get("cost_usd") or 0.0) for event in events), 6)
    latencies = [float(event.get("latency_ms") or 0.0) for event in events if event.get("latency_ms")]

    stats = {
        "total": len(events),
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
        "avg_tokens": int(total_tokens / len(events)) if events else 0,
        "avg_cost_usd": round(total_cost / len(events), 6) if events else 0.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "blocked": sum(1 for event in events if event.get("blocked")),
        "clarifications": sum(1 for event in events if event.get("needs_clarification")),
        "errors": sum(1 for event in events if event.get("status") == "error"),
        "low_confidence": sum(
            1
            for event in events
            if (event.get("trace", {}) or {}).get("confidence", {}).get("is_low_confidence")
        ),
    }

    return {
        "storage": "local",
        "daily": [],
        "stats": stats,
        "chat_history": events[:50],
        "recent_events": events[:25],
        "problem_events": problem_events[:12],
        "source_usage": source_counts.most_common(12),
        "category_usage": category_counts.most_common(8),
    }


def find_chat_event(event_id: str) -> Optional[dict]:
    if supabase_store.enabled:
        rows = [row for row in supabase_store.fetch_flagged_chats(limit=500) if row.get("id") == event_id]
        events = [supabase_event_from_row(row) for row in rows]
    else:
        events = load_chat_events()
    for event in events:
        if event.get("id") == event_id:
            trace = event.get("trace", {}) or {}
            confidence = trace.get("confidence", {}) or {}
            retrieval_diagnostics = trace.get("retrieval_diagnostics", {}) or {}
            public_sources = []
            for source in event.get("sources", []) or []:
                public_sources.append(
                    {
                        "title": source.get("title", "Untitled source"),
                        "url": source.get("url", "URL not provided"),
                    }
                )
            return {
                "id": event.get("id", ""),
                "timestamp": event.get("timestamp", ""),
                "status": event.get("status") or "answered",
                "latency_ms": event.get("latency_ms", 0),
                "response_mode": event.get("response_mode", ""),
                "blocked": bool(event.get("blocked")),
                "needs_clarification": bool(event.get("needs_clarification")),
                "confidence_score": confidence.get("score"),
                "is_low_confidence": bool(confidence.get("is_low_confidence")),
                "confidence_reasons": confidence.get("reasons", []) or [],
                "source_count": len(public_sources),
                "retrieved_count": len(trace.get("retrieved_metadata", []) or []),
                "sources": public_sources,
                "trace": build_dashboard_trace(trace),
                "retrieval_summary": {
                    "selected_count": retrieval_diagnostics.get("selected_count"),
                    "distinct_source_count": retrieval_diagnostics.get("distinct_source_count"),
                    "top_score": retrieval_diagnostics.get("top_score"),
                    "score_gap": retrieval_diagnostics.get("score_gap"),
                },
                "display_label": f"Interaction {(event.get('id', '') or 'unknown')[:8]}",
                "preview_text": build_dashboard_preview(event),
                "retrieval_scores": extract_retrieval_scores(trace),
                **extract_telemetry(trace),
            }
    return None
