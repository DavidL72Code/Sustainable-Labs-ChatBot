"""The Gemini client: safety settings, generation config, and the two call paths.

Split out of Chatbot.py unchanged. Depends only on config (for settings) and
telemetry (to record each call), both of which are leaves, so this module can be
imported from anywhere in the pipeline without a cycle.

_caller_stage walks the interpreter call stack to label a call by the pipeline
function that issued it. Moving the definition into this module does not add a
frame, so the labels it produces are the same.
"""
from __future__ import annotations

import os
import time
from typing import Optional

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - dependency availability depends on the runtime
    genai = None
    genai_types = None

from config import ChatbotConfig
from telemetry import _caller_stage, record_llm_call


_gemini_client: Optional[object] = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        if genai is None:
            raise ImportError("Install google-genai to use Gemini.")
        cfg = ChatbotConfig()
        if not cfg.gemini_api_key:
            raise ValueError("Set GEMINI_API_KEY before using Gemini.")
        _gemini_client = genai.Client(api_key=cfg.gemini_api_key)
    return _gemini_client


_DEFAULT_SAFETY_SETTINGS = [
    genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
    genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_MEDIUM_AND_ABOVE"),
    genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
    genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
] if genai_types is not None else None



# Reproducibility: temperature alone does not make decoding greedy. With
# top_p=0.95/top_k=40 the model still samples from a 40-token distribution, so
# the same question at temperature 0 returned four different answers in four
# runs. At temperature 0 pin top_k=1 and top_p=1.0 (greedy) and send a fixed
# seed, so a run can be reproduced and a change can be told apart from noise.
GEMINI_SEED = int(os.getenv("GEMINI_SEED", "7"))


# Models that reject a thinking_config outright (Gemma, for one). Populated the
# first time a model 400s on it, so the probe costs one failed call per process
# rather than one per question.
_MODELS_WITHOUT_THINKING: set[str] = set()


def _rejects_thinking(exc: Exception) -> bool:
    message = str(exc)
    return "INVALID_ARGUMENT" in message and "hinking" in message


_WARNED_ONCE: set[str] = set()


def _warn_once(message: str) -> None:
    """Log a recurring failure once per process so it cannot hide, without
    printing the same line on every request."""
    if message in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(message)
    print(f"[warn] {message}", file=sys.stderr, flush=True)


def _gemini_gen_config(
    temperature: float,
    thinking_budget: int = 1024,
    include_thinking: bool = True,
) -> "genai_types.GenerateContentConfig":
    deterministic = float(temperature or 0.0) <= 0.0
    return genai_types.GenerateContentConfig(
        temperature=temperature,
        top_p=1.0 if deterministic else 0.95,
        top_k=1 if deterministic else 40,
        seed=GEMINI_SEED,
        # Thinking tokens count against this budget, so a 1024-token thinking
        # pass left ~1024 for the answer. The eval judge writes a JSON object
        # with a prose "notes" field and ran out mid-key ('"right_c'), which
        # cost n146 a verdict on an answer it had already scored 5/5. Raising
        # the ceiling cannot change a response that already fit.
        max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096")),
        thinking_config=(
            genai_types.ThinkingConfig(thinking_budget=thinking_budget) if include_thinking else None
        ),
        safety_settings=_DEFAULT_SAFETY_SETTINGS,
    )


def call_gemini(prompt: str, model: Optional[str] = None, temperature: Optional[float] = None, thinking_budget: int = 1024) -> str:
    cfg = ChatbotConfig()
    client = _get_gemini_client()
    model_name = model or cfg.gemini_model
    temp = temperature if temperature is not None else cfg.gemini_temperature
    stage = _caller_stage()
    started_at = time.perf_counter()
    supports_thinking = model_name not in _MODELS_WITHOUT_THINKING
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=_gemini_gen_config(
                temp, thinking_budget=thinking_budget, include_thinking=supports_thinking
            ),
        )
    except Exception as exc:
        # Some models reject thinking_budget=0 and some reject a thinking_config
        # at all — Gemma raises "Thinking budget is not supported for this
        # model." Retrying with the *default* budget only helps the first case;
        # the second needs the field dropped entirely. Getting this wrong meant
        # every planner call failed silently and no question was ever split into
        # facets, so multi-part questions only ever answered their first half.
        if not (supports_thinking and _rejects_thinking(exc)):
            raise
        _MODELS_WITHOUT_THINKING.add(model_name)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=_gemini_gen_config(temp, include_thinking=False),
        )
    record_llm_call(
        model=model_name,
        stage=stage,
        usage=getattr(response, "usage_metadata", None),
        latency_ms=(time.perf_counter() - started_at) * 1000,
        streamed=False,
    )
    return response.text.strip()


def call_gemini_stream(prompt: str, model: Optional[str] = None, temperature: Optional[float] = None, stage: Optional[str] = None):
    """Yields text chunks as they stream from the Gemini API."""
    cfg = ChatbotConfig()
    client = _get_gemini_client()
    model_name = model or cfg.gemini_model
    temp = temperature if temperature is not None else cfg.gemini_temperature
    stage = stage or _caller_stage(default="generation")
    started_at = time.perf_counter()
    usage = None
    for chunk in client.models.generate_content_stream(
        model=model_name,
        contents=prompt,
        config=_gemini_gen_config(temp),
    ):
        chunk_usage = getattr(chunk, "usage_metadata", None)
        if chunk_usage is not None:
            usage = chunk_usage
        if chunk.text:
            yield chunk.text
    record_llm_call(
        model=model_name,
        stage=stage,
        usage=usage,
        latency_ms=(time.perf_counter() - started_at) * 1000,
        streamed=True,
    )
