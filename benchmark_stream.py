from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path


def load_dotenv_simple(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def time_question(chatbot, question: str) -> dict:
    started = time.perf_counter()
    first_delta_at: float | None = None
    last_delta_at: float | None = None
    suggestions_at: float | None = None
    done_at: float | None = None
    delta_count = 0

    for event_text in chatbot.answer_stream(question, recent_history=[]):
        now = time.perf_counter()
        if not event_text.startswith("data: "):
            continue
        try:
            payload = json.loads(event_text[len("data: "):].strip())
        except json.JSONDecodeError:
            continue

        etype = payload.get("type")
        if etype == "delta":
            if first_delta_at is None:
                first_delta_at = now
            last_delta_at = now
            delta_count += 1
        elif etype == "suggestions":
            suggestions_at = now
        elif etype == "done":
            done_at = now
            break
        elif payload.get("done") is True:
            done_at = now
            break

    total = (done_at or time.perf_counter()) - started
    ttft = (first_delta_at - started) if first_delta_at else None
    stream_dur = (last_delta_at - first_delta_at) if (first_delta_at and last_delta_at) else None
    suggestion_gap = (suggestions_at - last_delta_at) if (suggestions_at and last_delta_at) else None
    done_gap = (done_at - last_delta_at) if (done_at and last_delta_at) else None

    return {
        "total": total,
        "ttft": ttft,
        "stream_dur": stream_dur,
        "suggestion_gap": suggestion_gap,
        "done_gap": done_gap,
        "delta_count": delta_count,
        "got_suggestions": suggestions_at is not None,
    }


def fmt(value: float | None) -> str:
    return f"{value:5.2f}s" if value is not None else "  n/a"


def main() -> None:
    load_dotenv_simple(Path(__file__).resolve().parent / ".env")
    import Chatbot
    Chatbot.ChatbotConfig.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    cfg = Chatbot.ChatbotConfig()
    chatbot = Chatbot.create_chatbot(cfg)

    test_questions = [
        "What is the Sustainable Solutions Lab and what is its mission?",
        "What is SSL's annual operating budget for fiscal year 2025?",
        "Tell me about the partnership between SSL and the World Bank.",
    ]

    print(f"{'question':<55} {'ttft':>7} {'stream':>7} {'sugg_gap':>9} {'done_gap':>9} {'total':>7} sugg?")
    print("-" * 105)
    for question in test_questions:
        try:
            timing = time_question(chatbot, question)
        except Exception as exc:
            print(f"{question[:54]:<55}  error: {exc}")
            continue
        label = question[:54]
        print(
            f"{label:<55} {fmt(timing['ttft']):>7} {fmt(timing['stream_dur']):>7} "
            f"{fmt(timing['suggestion_gap']):>9} {fmt(timing['done_gap']):>9} "
            f"{fmt(timing['total']):>7} {'yes' if timing['got_suggestions'] else 'no'}"
        )


if __name__ == "__main__":
    main()
