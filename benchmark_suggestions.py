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


def main() -> None:
    load_dotenv_simple(Path(__file__).resolve().parent / ".env")
    import Chatbot
    Chatbot.ChatbotConfig.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

    user_question = "What is the Sustainable Solutions Lab and what is its mission?"
    fake_answer = (
        "The Sustainable Solutions Lab (SSL) at UMass Boston is an applied research "
        "and action institute working at the intersection of climate and equity. It "
        "partners with communities to develop equitable and sustainable solutions."
    )

    # Old style: with answer, thinking_budget=1024
    old_prompt = (
        "A user asked a chatbot about the Sustainable Solutions Lab (SSL) this question:\n\n"
        f"Question: {user_question}\n\n"
        f"The chatbot answered:\n{fake_answer}\n\n"
        "Based on the question and answer, suggest exactly 3 short follow-up questions "
        "a new user might want to explore next.\n"
        "Focus on SSL's research, staff, projects, publications, or initiatives.\n"
        "Return ONLY a valid JSON array of 3 strings. No preamble, no markdown fences.\n"
        'Example: ["What projects is SSL currently working on?", "Who leads SSL?", "How is SSL funded?"]'
    )

    new_prompt = (
        "A user is chatting with a chatbot about the Sustainable Solutions Lab (SSL).\n\n"
        f"Question: {user_question}\n\n"
        "Suggest exactly 3 short follow-up questions a new user might want to explore "
        "after asking the question above.\n"
        "Focus on SSL's research, staff, projects, publications, or initiatives.\n"
        "Return ONLY a valid JSON array of 3 strings. No preamble, no markdown fences.\n"
        'Example: ["What projects is SSL currently working on?", "Who leads SSL?", "How is SSL funded?"]'
    )

    def time_call(label: str, prompt: str, thinking_budget: int) -> None:
        started = time.perf_counter()
        try:
            raw = Chatbot.call_gemini(prompt, temperature=0.4, thinking_budget=thinking_budget)
            elapsed = time.perf_counter() - started
            parsed = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
            suggestions = json.loads(parsed) if parsed.startswith("[") else "(parse fail)"
            print(f"{label:<50} {elapsed:6.2f}s  suggestions={suggestions if isinstance(suggestions, list) else 'parse_err'}")
        except Exception as exc:
            print(f"{label:<50} ERROR {str(exc)[:120]}")

    # Warm any client init
    time_call("warm-up (new style, budget=0)", new_prompt, 0)
    print("-" * 100)

    time_call("OLD: with-answer, thinking_budget=1024", old_prompt, 1024)
    time_call("NEW: no-answer, thinking_budget=0    ", new_prompt, 0)
    time_call("OLD again (avoid first-call bias)    ", old_prompt, 1024)
    time_call("NEW again                              ", new_prompt, 0)


if __name__ == "__main__":
    main()
