from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
QUESTIONS_PATH = PROJECT_ROOT / "questions.json"
OUTPUT_PATH = PROJECT_ROOT / "question_eval_results.json"
OVERWRITE_RESULTS = os.getenv("EVAL_OVERWRITE", "").lower() in {"1", "true", "yes"}

ChatbotConfig = None
ConversationTurn = None
call_gemini = None
create_chatbot = None
LAST_GEMINI_CALL_AT = 0.0
MIN_GEMINI_INTERVAL_SECONDS = 4.5


def load_dotenv_simple(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def extract_json_block(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(1))

    return json.loads(text)


def list_folder_inventory(folder_path: Path) -> list[str]:
    return sorted(
        str(path.relative_to(PROJECT_ROOT).as_posix())
        for path in folder_path.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".pdf"}
    )


def build_corpus_reference(target_sources: list[str]) -> str:
    blocks: list[str] = []

    for source in target_sources:
        source_path = PROJECT_ROOT / source
        if source_path.is_dir():
            inventory = list_folder_inventory(source_path)
            blocks.append(
                "\n".join(
                    [
                        f"Folder: {source}",
                        "Known source document inventory:",
                        *inventory,
                    ]
                )
            )
            continue

        if not source_path.exists():
            blocks.append(f"Missing source reference: {source}")
            continue

        if source_path.suffix.lower() == ".txt":
            text = source_path.read_text(encoding="utf-8")
            blocks.append(f"Source: {source}\n{text[:14000]}")
        else:
            blocks.append(f"Source: {source}\nBinary/PDF source document. Use file title/path as inventory evidence.")

    return "\n\n" + ("\n\n".join(blocks) if blocks else "No target source references provided.")


def judge_response(
    *,
    prompt_kind: str,
    question_text: str,
    answer_text: str,
    sources: list[dict[str, Any]],
    target_sources: list[str],
    corpus_reference: str,
    conversation: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    judge_prompt = f"""
You are evaluating a RAG chatbot answer against a known corpus.

Return valid JSON only with this exact schema:
{{
  "clarity": 1,
  "professional_tone": 1,
  "correctness_vs_corpus": 1,
  "citations": 1,
  "answered_question": "yes",
  "hallucinated": "no",
  "right_citations": "yes",
  "notes": "short explanation"
}}

Scoring rules:
- clarity: 1-5
- professional_tone: 1-5
- correctness_vs_corpus: 1-5, based only on the provided corpus reference
- citations: 1-5, based on whether the provided sources are useful/relevant support
- answered_question: yes if it directly answers the question asked
- hallucinated: yes if it states unsupported or clearly incorrect facts
- right_citations: yes if the returned sources match the relevant corpus sources well enough

Prompt kind: {prompt_kind}
Question:
{question_text}

Conversation context:
{json.dumps(conversation or [], indent=2)}

Assistant answer:
{answer_text}

Returned sources:
{json.dumps(sources, indent=2)}

Expected target sources:
{json.dumps(target_sources, indent=2)}

Corpus reference:
{corpus_reference}
""".strip()

    raw_judgment = gemini_call_with_retry(judge_prompt, temperature=0.0)
    parsed = extract_json_block(raw_judgment)

    return {
        "clarity": int(parsed["clarity"]),
        "professional_tone": int(parsed["professional_tone"]),
        "correctness_vs_corpus": int(parsed["correctness_vs_corpus"]),
        "citations": int(parsed["citations"]),
        "answered_question": str(parsed["answered_question"]).strip().lower(),
        "hallucinated": str(parsed["hallucinated"]).strip().lower(),
        "right_citations": str(parsed["right_citations"]).strip().lower(),
        "notes": str(parsed.get("notes", "")).strip(),
        "judge_raw": raw_judgment,
    }


def run_single_turn(chatbot: Any, item: dict[str, Any]) -> dict[str, Any]:
    question = item["question"]
    target_sources = item.get("target_sources", [])
    result = chatbot.answer(question, recent_history=[])
    corpus_reference = build_corpus_reference(target_sources)
    judgment = judge_response(
        prompt_kind=item.get("type", "single_turn"),
        question_text=question,
        answer_text=result["reply"],
        sources=result.get("sources", []),
        target_sources=target_sources,
        corpus_reference=corpus_reference,
    )
    return {
        "id": item["id"],
        "kind": "single_turn",
        "type": item.get("type", "single_turn"),
        "question": question,
        "target_sources": target_sources,
        "output": result["reply"],
        "sources": result.get("sources", []),
        "scores": {
            "clarity": judgment["clarity"],
            "professional_tone": judgment["professional_tone"],
            "correctness_vs_corpus": judgment["correctness_vs_corpus"],
            "citations": judgment["citations"],
        },
        "classification": {
            "answered_question": judgment["answered_question"],
            "hallucinated": judgment["hallucinated"],
            "right_citations": judgment["right_citations"],
        },
        "notes": judgment["notes"],
    }


def run_multi_turn(chatbot: Any, item: dict[str, Any]) -> dict[str, Any]:
    turns = item["turns"]
    target_sources = item.get("target_sources", [])
    recent_history: list[ConversationTurn] = []
    transcript: list[dict[str, str]] = []
    last_result: dict[str, Any] | None = None

    for turn in turns:
        last_result = chatbot.answer(turn, recent_history=recent_history)
        transcript.append({"user": turn, "assistant": last_result["reply"]})
        recent_history.append(ConversationTurn(user=turn, assistant=last_result["reply"]))

    assert last_result is not None
    final_question = turns[-1]
    corpus_reference = build_corpus_reference(target_sources)
    judgment = judge_response(
        prompt_kind=item.get("type", "multi_turn"),
        question_text=final_question,
        answer_text=last_result["reply"],
        sources=last_result.get("sources", []),
        target_sources=target_sources,
        corpus_reference=corpus_reference,
        conversation=transcript,
    )

    return {
        "id": item["id"],
        "kind": "multi_turn",
        "type": item.get("type", "multi_turn"),
        "question": final_question,
        "conversation": transcript,
        "target_sources": target_sources,
        "output": last_result["reply"],
        "sources": last_result.get("sources", []),
        "scores": {
            "clarity": judgment["clarity"],
            "professional_tone": judgment["professional_tone"],
            "correctness_vs_corpus": judgment["correctness_vs_corpus"],
            "citations": judgment["citations"],
        },
        "classification": {
            "answered_question": judgment["answered_question"],
            "hallucinated": judgment["hallucinated"],
            "right_citations": judgment["right_citations"],
        },
        "notes": judgment["notes"],
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}

    def average(key: str) -> float:
        values = [result["scores"][key] for result in results]
        return round(sum(values) / len(values), 2)

    def count_flag(key: str, value: str) -> int:
        return sum(1 for result in results if result["classification"][key] == value)

    return {
        "total_cases": len(results),
        "average_scores": {
            "clarity": average("clarity"),
            "professional_tone": average("professional_tone"),
            "correctness_vs_corpus": average("correctness_vs_corpus"),
            "citations": average("citations"),
        },
        "classification_counts": {
            "answered_yes": count_flag("answered_question", "yes"),
            "answered_no": count_flag("answered_question", "no"),
            "hallucinated_yes": count_flag("hallucinated", "yes"),
            "hallucinated_no": count_flag("hallucinated", "no"),
            "right_citations_yes": count_flag("right_citations", "yes"),
            "right_citations_no": count_flag("right_citations", "no"),
        },
    }


def save_results(results: list[dict[str, Any]]) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": ChatbotConfig.gemini_model,
        "question_file": str(QUESTIONS_PATH.relative_to(PROJECT_ROOT)),
        "summary": summarize_results(results),
        "results": results,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_existing_results() -> list[dict[str, Any]]:
    if OVERWRITE_RESULTS:
        return []

    if not OUTPUT_PATH.exists():
        return []

    try:
        with OUTPUT_PATH.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError:
        return []

    return payload.get("results", []) if isinstance(payload, dict) else []


def extract_retry_delay_seconds(error_text: str) -> float:
    match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", error_text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 20.0


def gemini_call_with_retry(prompt: str, *, model: str | None = None, temperature: float | None = None) -> str:
    global LAST_GEMINI_CALL_AT

    max_attempts = 8
    for attempt in range(1, max_attempts + 1):
        elapsed = time.monotonic() - LAST_GEMINI_CALL_AT
        if LAST_GEMINI_CALL_AT and elapsed < MIN_GEMINI_INTERVAL_SECONDS:
            time.sleep(MIN_GEMINI_INTERVAL_SECONDS - elapsed)

        try:
            response = call_gemini(prompt, model=model, temperature=temperature)
            LAST_GEMINI_CALL_AT = time.monotonic()
            return response
        except Exception as exc:
            error_text = str(exc)
            if "429" not in error_text and "quota" not in error_text.lower():
                raise

            if attempt == max_attempts:
                raise

            delay_seconds = extract_retry_delay_seconds(error_text) + 2.0
            print(f"Rate limit hit. Sleeping {delay_seconds:.1f}s before retry {attempt + 1}/{max_attempts}...")
            time.sleep(delay_seconds)

    raise RuntimeError("Gemini call retry loop exited unexpectedly.")


def load_chatbot_symbols() -> None:
    global ChatbotConfig, ConversationTurn, call_gemini, create_chatbot

    import Chatbot as chatbot_module

    chatbot_module.ChatbotConfig.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    chatbot_module.ChatbotConfig.gemini_model = os.getenv("GEMINI_MODEL", chatbot_module.ChatbotConfig.gemini_model)
    chatbot_module.ChatbotConfig.gemini_temperature = float(
        os.getenv("GEMINI_TEMPERATURE", str(chatbot_module.ChatbotConfig.gemini_temperature))
    )
    chatbot_module.ChatbotConfig.seed_documents_directory = os.getenv(
        "SEED_DOCUMENTS_DIRECTORY",
        chatbot_module.ChatbotConfig.seed_documents_directory,
    )
    chatbot_module.ChatbotConfig.force_reindex = os.getenv("FORCE_REINDEX", "").lower() in {"1", "true", "yes"}

    ChatbotConfig = chatbot_module.ChatbotConfig
    ConversationTurn = chatbot_module.ConversationTurn
    call_gemini = chatbot_module.call_gemini
    create_chatbot = chatbot_module.create_chatbot


def main() -> None:
    load_dotenv_simple(PROJECT_ROOT / ".env")
    load_chatbot_symbols()
    chatbot = create_chatbot(ChatbotConfig())
    chatbot.llm_callable = gemini_call_with_retry

    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    results: list[dict[str, Any]] = load_existing_results()
    completed_ids = {result["id"] for result in results}

    for item in questions.get("single_turn", []):
        if item["id"] in completed_ids:
            print(f"Skipping {item['id']} (already completed)...")
            continue
        print(f"Running {item['id']}...")
        results.append(run_single_turn(chatbot, item))
        save_results(results)

    for item in questions.get("multi_turn", []):
        if item["id"] in completed_ids:
            print(f"Skipping {item['id']} (already completed)...")
            continue
        print(f"Running {item['id']}...")
        results.append(run_multi_turn(chatbot, item))
        save_results(results)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": ChatbotConfig.gemini_model,
        "question_file": str(QUESTIONS_PATH.relative_to(PROJECT_ROOT)),
        "summary": summarize_results(results),
        "results": results,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved evaluation results to {OUTPUT_PATH.name}")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
