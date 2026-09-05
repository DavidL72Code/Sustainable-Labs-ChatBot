from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
QUESTIONS_PATH = PROJECT_ROOT / os.getenv("EVAL_QUESTIONS_FILE", "questions.json")
OUTPUT_PATH = PROJECT_ROOT / os.getenv("EVAL_OUTPUT_FILE", "question_eval_results.json")
OVERWRITE_RESULTS = os.getenv("EVAL_OVERWRITE", "").lower() in {"1", "true", "yes"}
STOP_ON_FAILURE = os.getenv("EVAL_STOP_ON_FAILURE", "").lower() in {"1", "true", "yes"}
ONLY_IDS = {
    item.strip()
    for item in os.getenv("EVAL_ONLY_IDS", "").split(",")
    if item.strip()
}

ChatbotConfig = None
ConversationTurn = None
call_gemini = None
create_chatbot = None
LAST_GEMINI_CALL_AT = 0.0
MIN_GEMINI_INTERVAL_SECONDS = float(os.getenv("EVAL_MIN_GEMINI_INTERVAL_SECONDS", "4.5"))
# The judge must stay fixed while GEMINI_MODEL varies, or a model comparison
# also swaps the grader and the two effects cannot be separated.
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "").strip() or None


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


_PDF_TEXT_CACHE: dict[str, str] = {}


def _pdf_text(path: Path) -> str:
    """Extract and cache PDF text so the judge can verify answers against it."""
    key = str(path)
    if key not in _PDF_TEXT_CACHE:
        try:
            from Chatbot import extract_pdf_text
            _PDF_TEXT_CACHE[key] = extract_pdf_text(path) or ""
        except Exception as exc:
            print(f"Could not read {path.name}: {exc}", flush=True)
            _PDF_TEXT_CACHE[key] = ""
    return _PDF_TEXT_CACHE[key]


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
            # Use the full document. A low cap (previously 14k chars) truncated long
            # sources like the annual reports (~50k chars), so facts stated later in the
            # document were invisible to the judge and correct answers were wrongly
            # flagged as hallucinations.
            blocks.append(f"Source: {source}\n{text[:CORPUS_REFERENCE_CHARS]}")
        else:
            # Extract PDF text rather than telling the judge to grade on the
            # filename alone. Without this, every answer drawn from a PDF looks
            # unverifiable and correct facts get marked as hallucinated.
            text = _pdf_text(source_path)
            if text:
                blocks.append(f"Source: {source}\n{text[:CORPUS_REFERENCE_CHARS]}")
            else:
                blocks.append(f"Source: {source}\nPDF text could not be extracted. Use file title/path as inventory evidence.")

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
- answered_question: yes if it directly answers the question asked. If the
  conversation context shows the assistant asked a clarifying question because
  the question was ambiguous, and the user answered it, judge the FINAL answer
  only. Asking a reasonable clarifying question is correct behaviour and must
  not on its own count as failing to answer.
- hallucinated: yes if it states unsupported or clearly incorrect facts
- If the corpus reference genuinely does not contain what the question asks for,
  a clear refusal ("the available documents do not state this") is the CORRECT
  answer: score answered_question yes, hallucinated no, right_citations yes, and
  correctness_vs_corpus 5. Do not penalise an accurate refusal for returning no
  sources. Only score it as failing if the answer IS present in the corpus
  reference and the assistant missed it.
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

    raw_judgment = ""
    parsed: dict[str, Any] | None = None
    for attempt in range(1, 4):
        prompt = judge_prompt
        if attempt > 1:
            prompt = (
                judge_prompt
                + "\n\nYour previous response could not be parsed as JSON. "
                + "Return only one valid JSON object with the exact schema and no markdown."
            )
        raw_judgment = gemini_call_with_retry(prompt, temperature=0.0, model=JUDGE_MODEL)
        try:
            parsed = extract_json_block(raw_judgment)
            break
        except json.JSONDecodeError:
            if attempt == 3:
                # Never lose a whole run to one bad judge response. Generation is
                # greedy with a fixed seed, so all three attempts return the same
                # truncated JSON and the run died at question 8 of 11 with the
                # other 10 verdicts already computed. Record the failure as its
                # own outcome instead — a scoring gap is visible in the results
                # and costs one question, not the entire run.
                print(
                    f"Judge returned unparseable JSON after 3 attempts; "
                    f"recording as judge_error: {raw_judgment[:120]!r}",
                    flush=True,
                )
                return {
                    "clarity": 0,
                    "professional_tone": 0,
                    "correctness_vs_corpus": 0,
                    "citations": 0,
                    "answered_question": "judge_error",
                    "hallucinated": "judge_error",
                    "right_citations": "judge_error",
                    "notes": f"Judge response could not be parsed as JSON: {raw_judgment[:400]}",
                }

    assert parsed is not None

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


def run_stream_answer(chatbot: Any, question: str, conversation: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Capture the same streaming path used by the HTTP website endpoint."""
    answer_parts: list[str] = []
    sources: list[dict[str, Any]] = []
    trace: dict[str, Any] = {}
    status = "answered"
    response_mode = ""
    needs_clarification = False
    clarification_options: list[Any] = []
    conversation_state: dict[str, Any] | None = None
    stream_error = ""

    # answer_stream generates through call_gemini_stream, which does not go via
    # gemini_call_with_retry, so a transient 503 or 429 from Google would abort
    # the whole eval. Retry the question instead of losing the run.
    frames: list[str] = []
    for attempt in range(1, 7):
        try:
            frames = list(chatbot.answer_stream(question, recent_history=conversation or []))
            break
        except Exception as exc:
            message = str(exc).lower()
            transient = any(
                marker in message
                for marker in ("429", "quota", "503", "unavailable", "500", "internal error", "deadline", "timeout")
            )
            if not transient or attempt == 6:
                raise
            delay = min(60, 5 * (2 ** (attempt - 1)))
            print(f"  transient error, retrying in {delay}s: {str(exc)[:70]}", flush=True)
            time.sleep(delay)

    for frame in frames:
        for line in str(frame).splitlines():
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if event.get("type") == "meta":
                sources = event.get("sources", []) or []
                trace = event.get("trace", {}) or {}
                status = event.get("status", status)
                response_mode = event.get("response_mode", response_mode)
                needs_clarification = bool(event.get("needs_clarification", False))
                clarification_options = event.get("clarification_options", []) or clarification_options
            elif event.get("type") == "delta":
                answer_parts.append(str(event.get("delta", "")))
            elif event.get("type") == "error":
                stream_error = str(event.get("error", ""))
                status = "error"
            elif "reply" in event:
                answer_parts = [str(event.get("reply", ""))]
                sources = event.get("sources", []) or []
                trace = event.get("trace", {}) or {}
                status = event.get("status", status)
                response_mode = event.get("response_mode", response_mode)
                needs_clarification = bool(event.get("needs_clarification", False))
                clarification_options = event.get("clarification_options", []) or clarification_options
                if isinstance(event.get("conversation_state"), dict):
                    conversation_state = event["conversation_state"]

    return {
        "reply": "".join(answer_parts).strip(),
        "sources": sources,
        "trace": trace,
        "status": status,
        "response_mode": response_mode,
        "needs_clarification": needs_clarification,
        "clarification_options": clarification_options,
        "conversation_state": conversation_state,
        "stream_error": stream_error,
    }


MAX_CLARIFICATION_ROUNDS = int(os.getenv("EVAL_MAX_CLARIFICATION_ROUNDS", "2"))
# The judge grades against this text, so truncating it invents failures. The
# Feasibility report is 695k chars; at the old 200k cap the judge saw 29% of it
# and marked correct answers "not in the corpus" — n142/n173/n174 sit at offsets
# 378k/250k/262k, past the cut. Large enough to hold the biggest document whole.
CORPUS_REFERENCE_CHARS = int(os.getenv("EVAL_CORPUS_REFERENCE_CHARS", "750000"))
# The harness paces judge calls (MIN_GEMINI_INTERVAL_SECONDS) but the chatbot's
# planner, selector and generation calls fire back-to-back, which bursts past the
# free tier's per-minute cap and kills a run two questions in with "Evidence
# selector RPM limit". Pace whole questions instead.
QUESTION_DELAY_SECONDS = float(os.getenv("EVAL_QUESTION_DELAY_SECONDS", "0"))


def document_label(source_path: str) -> str:
    """Readable report name for a corpus path, for the simulated user reply."""
    stem = Path(str(source_path or "")).stem
    stem = re.sub(r"(?i)^executive\s+summary[_\s-]*", "", stem)
    stem = stem.split("_")[0].strip() or stem
    return re.sub(r"\s+", " ", stem).strip()


def simulated_clarification_reply(item: dict[str, Any], result: dict[str, Any]) -> str:
    """What the user who asked this question would say when asked to narrow it.

    The eval knows which document the question was written from, which is the
    scope a real user carries in their head — not the answer. Supplying it lets
    the run continue instead of scoring a reasonable clarifying question as a
    failure to answer.
    """
    targets = [document_label(path) for path in item.get("target_sources", []) if str(path).strip()]
    options = [
        str(option.get("label") or option.get("value") or option)
        if isinstance(option, dict) else str(option)
        for option in (result.get("clarification_options") or [])
        if option
    ]
    for option in options:
        if any(target and target.lower() in option.lower() for target in targets):
            return option
    if targets:
        return f"I mean the {targets[0]} report."
    return "Please answer using whichever source is most relevant."


def run_single_turn(chatbot: Any, item: dict[str, Any]) -> dict[str, Any]:
    question = item["question"]
    target_sources = item.get("target_sources", [])
    conversation: list[dict[str, str]] = []
    transcript: list[dict[str, str]] = []
    result = run_stream_answer(chatbot, question, conversation=[])
    asked = question
    # A clarifying question is a correct response to an ambiguous question, not a
    # refusal to answer. Resolve it the way the user would and grade what comes
    # back, with the whole exchange handed to the judge.
    rounds = 0
    while result.get("needs_clarification") and rounds < MAX_CLARIFICATION_ROUNDS:
        rounds += 1
        transcript.append({"user": asked, "assistant": result["reply"]})
        turn = ConversationTurn(user=asked, assistant=result["reply"])
        state = result.get("conversation_state")
        if isinstance(state, dict):
            turn["state"] = state
        conversation.append(turn)
        asked = simulated_clarification_reply(item, result)
        print(f"  clarification round {rounds}: {asked[:70]}", flush=True)
        result = run_stream_answer(chatbot, asked, conversation=conversation)
    transcript.append({"user": asked, "assistant": result["reply"]})
    corpus_reference = build_corpus_reference(target_sources)
    judgment = judge_response(
        prompt_kind=item.get("type", "single_turn"),
        question_text=question,
        answer_text=result["reply"],
        sources=result.get("sources", []),
        target_sources=target_sources,
        corpus_reference=corpus_reference,
        conversation=transcript if rounds else None,
    )
    return {
        "id": item["id"],
        "kind": "single_turn",
        "type": item.get("type", "single_turn"),
        "question": question,
        "target_sources": target_sources,
        "clarification_rounds": rounds,
        "transcript": transcript if rounds else [],
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
        "stream_error": result.get("stream_error", ""),
    }


def run_multi_turn(chatbot: Any, item: dict[str, Any]) -> dict[str, Any]:
    turns = item["turns"]
    target_sources = item.get("target_sources", [])
    recent_history: list[ConversationTurn] = []
    transcript: list[dict[str, str]] = []
    last_result: dict[str, Any] | None = None

    for turn in turns:
        last_result = run_stream_answer(chatbot, turn, conversation=recent_history)
        transcript.append({"user": turn, "assistant": last_result["reply"]})
        turn_payload = ConversationTurn(user=turn, assistant=last_result["reply"])
        conversation_state = last_result.get("conversation_state")
        if not isinstance(conversation_state, dict):
            conversation_state = chatbot.build_next_conversation_state(
                recent_history,
                turn,
                last_result,
            )
        if isinstance(conversation_state, dict):
            turn_payload["state"] = conversation_state
        recent_history.append(turn_payload)

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


def is_failure(result: dict[str, Any]) -> bool:
    classification = result["classification"]
    scores = result["scores"]
    return (
        classification["answered_question"] != "yes"
        or classification["hallucinated"] != "no"
        or classification["right_citations"] != "yes"
        or scores["correctness_vs_corpus"] < 4
    )


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
            lowered_error = error_text.lower()
            transient_markers = (
                "429", "quota", "503", "unavailable", "500", "internal error", "deadline", "timeout",
                "connection reset", "readerror", "read error", "connection refused", "temporarily",
            )
            if not any(marker in lowered_error for marker in transient_markers):
                raise

            if attempt == max_attempts:
                raise

            delay_seconds = extract_retry_delay_seconds(error_text) + 2.0
            print(f"Transient API error ({error_text[:80]}). Sleeping {delay_seconds:.1f}s before retry {attempt + 1}/{max_attempts}...", flush=True)
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

    stopped_early = False
    for item in questions.get("single_turn", []):
        if ONLY_IDS and item["id"] not in ONLY_IDS:
            continue
        if item["id"] in completed_ids:
            print(f"Skipping {item['id']} (already completed)...")
            continue
        print(f"Running {item['id']}...")
        if QUESTION_DELAY_SECONDS:
            time.sleep(QUESTION_DELAY_SECONDS)
        result = run_single_turn(chatbot, item)
        results.append(result)
        save_results(results)
        if STOP_ON_FAILURE and is_failure(result):
            print(f"Stopping on failure: {item['id']}", flush=True)
            stopped_early = True
            break

    for item in questions.get("multi_turn", []):
        if stopped_early:
            break
        if ONLY_IDS and item["id"] not in ONLY_IDS:
            continue
        if item["id"] in completed_ids:
            print(f"Skipping {item['id']} (already completed)...")
            continue
        print(f"Running {item['id']}...")
        result = run_multi_turn(chatbot, item)
        results.append(result)
        save_results(results)
        if STOP_ON_FAILURE and is_failure(result):
            print(f"Stopping on failure: {item['id']}", flush=True)
            break

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
