from __future__ import annotations

import os
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


def main() -> None:
    load_dotenv_simple(Path(__file__).resolve().parent / ".env")
    import Chatbot
    Chatbot.ChatbotConfig.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    cfg = Chatbot.ChatbotConfig()
    chatbot = Chatbot.create_chatbot(cfg)

    # A pronoun-heavy follow-up query that should force the planner to do work.
    test_query = "Tell me more about it and how it relates to climate work"

    # Warm-up first to remove cold-start bias.
    _ = chatbot.plan_query_with_llm(test_query, recent_history=[])

    print(f"{'run':<12} {'plan_query_with_llm':>22} {'rewrite':<60}")
    print("-" * 100)
    for run_index in range(3):
        started = time.perf_counter()
        plan = chatbot.plan_query_with_llm(test_query, recent_history=[])
        elapsed = time.perf_counter() - started
        rewrite = (plan.get("rewritten_query") or "")[:55]
        print(f"run {run_index + 1:<8} {elapsed:18.2f}s    {rewrite}")


if __name__ == "__main__":
    main()
