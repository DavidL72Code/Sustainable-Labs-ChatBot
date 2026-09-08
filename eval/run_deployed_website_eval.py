"""Exercise the deployed website's real streaming API and save what users see.

Default usage is capture-only and does not call a judge model:
    python3 run_deployed_website_eval.py \
      --questions question_eval_set/website/website_quality_smoke.json

Set WEBSITE_API_BASE to the deployed backend, or pass --base-url. The output is
JSON so it can later be graded without rerunning the deployment.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_BASE = "https://davidl72code-umb-sustainable-chatbot.hf.space"


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    if isinstance(payload, list):
        return payload
    for item in payload.get("single_turn", []):
        cases.append({**item, "kind": "single_turn", "turns": [item["question"]]})
    for item in payload.get("multi_turn", []):
        cases.append({**item, "kind": "multi_turn"})
    return cases


def parse_sse(response) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data: "):
            continue
        try:
            events.append(json.loads(line[6:]))
        except json.JSONDecodeError:
            events.append({"type": "parse_error", "raw": line[6:]})
    return events


def ask(base_url: str, message: str, conversation_id: str | None, timeout: float) -> dict[str, Any]:
    payload: dict[str, Any] = {"message": message}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            events = parse_sse(response)
            elapsed = round(time.perf_counter() - started, 3)
            reply = ""
            sources: list[dict[str, Any]] = []
            suggestions: list[str] = []
            status = ""
            returned_id = response.headers.get("X-Conversation-Id") or conversation_id
            for event in events:
                if event.get("type") == "delta":
                    reply += str(event.get("delta", ""))
                elif event.get("reply"):
                    reply = str(event["reply"])
                if event.get("type") == "meta":
                    sources = event.get("sources", []) or []
                    status = str(event.get("status", ""))
                elif event.get("sources"):
                    sources = event.get("sources", []) or []
                if event.get("type") == "suggestions":
                    suggestions = [str(value) for value in event.get("suggestions", [])]
                if event.get("status"):
                    status = str(event["status"])
                if event.get("conversation_id"):
                    returned_id = str(event["conversation_id"])
            return {
                "question": message,
                "answer": reply,
                "sources": sources,
                "suggestions": suggestions,
                "status": status or "answered",
                "conversation_id": returned_id,
                "elapsed_seconds": elapsed,
                "event_types": [str(event.get("type", "done")) for event in events],
                "stream_error": next((event.get("error") for event in events if event.get("type") == "error"), None),
            }
    except Exception as exc:
        return {
            "question": message,
            "answer": "",
            "sources": [],
            "suggestions": [],
            "status": "request_error",
            "conversation_id": conversation_id,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "event_types": [],
            "stream_error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("WEBSITE_API_BASE", DEFAULT_BASE))
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    cases = load_cases(args.questions)
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        conversation_id: str | None = None
        turns: list[dict[str, Any]] = []
        for turn in case.get("turns", []):
            result = ask(args.base_url, str(turn), conversation_id, args.timeout)
            conversation_id = result.get("conversation_id") or conversation_id
            turns.append(result)
        final = turns[-1] if turns else {}
        results.append({
            "id": case.get("id", f"website_{index:03d}"),
            "kind": case.get("kind", "single_turn"),
            "type": case.get("type", ""),
            "target_sources": case.get("target_sources", []),
            "turns": turns,
            "final_answer": final.get("answer", ""),
            "final_sources": final.get("sources", []),
            "final_suggestions": final.get("suggestions", []),
            "final_status": final.get("status", ""),
            "failed_transport": any(turn.get("stream_error") for turn in turns),
        })
        print(f"{index}/{len(cases)} {case.get('id', '')} {final.get('status', '')} "
              f"sources={len(final.get('sources', []))} suggestions={len(final.get('suggestions', []))}", flush=True)

    artifact = {
        "base_url": args.base_url.rstrip("/"),
        "captured_at_epoch": time.time(),
        "capture_only": True,
        "case_count": len(results),
        "results": results,
    }
    output = args.output or Path("Eval_ordered/website/deployed_website_smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
