"""Configuration and the small shared types.

Split out of Chatbot.py unchanged. Every other module reads settings from here,
so it must not import any of them back — keep this file free of pipeline
imports or the dependency graph turns into a cycle.

PROJECT_ROOT stays correct after the move because this module sits beside
Chatbot.py at the repository root; both resolve to the same directory. Moving
either into a subpackage would break the 27 call sites that hang off it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


LLMCallable = Callable[[str], str]



class ChatbotConfig:
    collection_name: str = "docs"
    persist_directory: str = "./chroma_db"
    seed_documents_directory: str = os.getenv("SEED_DOCUMENTS_DIRECTORY", "./SEED_DOCUMENTS")
    force_reindex: bool = os.getenv("FORCE_REINDEX", "").lower() in {"1", "true", "yes"}
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    # bge is asymmetric: passages are embedded bare, queries need this prefix.
    # Omitting it measurably degrades retrieval quality.
    query_embedding_prefix: str = os.getenv(
        "QUERY_EMBEDDING_PREFIX",
        "Represent this sentence for searching relevant passages: ",
    )
    chunk_size: int = 512
    chunk_overlap: int = 50
    summary_chunk_size: int = 1400
    summary_chunk_overlap: int = 140
    # 5 meant only the top 5 reranked chunks seeded the context. Measured on the
    # 2026-08-29 set, 12 put the answer-bearing document in front of the model
    # far more often, on both previously-failing and previously-passing questions.
    top_k: int = 10
    retrieval_candidate_pool: int = 12
    document_neighbor_count: int = int(os.getenv("DOCUMENT_NEIGHBOR_COUNT", "2"))
    document_neighbor_limit: int = int(os.getenv("DOCUMENT_NEIGHBOR_LIMIT", "8"))
    recent_history_turns: int = int(os.getenv("RECENT_HISTORY_TURNS", "6"))
    # Off by default, deliberately. This read "1" for a long time, but every
    # planner call was failing on a 400 the caller swallowed, so the effective
    # behaviour was off and both 208-question benchmarks were measured that way.
    # Repairing the call (see _MODELS_WITHOUT_THINKING) made the planner
    # authoritative again and changed 5 of 20 spot-check answers, two of them
    # regressions: entity questions rerouted from the entity registry to the
    # document registry and started listing filenames instead of people.
    # Leave this off until a full 208 run with it on beats the current baseline.
    always_llm_query_planning: bool = os.getenv("ALWAYS_LLM_QUERY_PLANNING", "0").lower() in {"1", "true", "yes"}
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    rewrite_model: str = os.getenv("REWRITE_MODEL", "gemma-4-26b-a4b-it")
    # Default to greedy decoding. At 0.7 the same question returned a different
    # answer on every run, which made pipeline behaviour unobservable: a fix and
    # a coin flip looked identical. With this at 0.0, _gemini_gen_config pins
    # top_k=1/top_p=1.0 and a fixed seed, so a given question and evidence
    # produce one answer. Set GEMINI_TEMPERATURE=0.7 to restore varied phrasing.
    gemini_temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))
    web_host: str = os.getenv("CHATBOT_HOST", "0.0.0.0")
    web_port: int = int(os.getenv("PORT", os.getenv("CHATBOT_PORT", "7860")))
    cors_origins: str = os.getenv("CORS_ORIGINS", "")
    trust_proxy_headers: bool = os.getenv("TRUST_PROXY_HEADERS", "0").lower() in {"1", "true", "yes"}
    dashboard_trace_mode: str = os.getenv("DASHBOARD_TRACE_MODE", "staff").strip().lower()
    admin_username: str = os.getenv("ADMIN_USERNAME", "").strip()
    admin_password_hash: str = os.getenv("ADMIN_PASSWORD_HASH", "").strip()
    dashboard_session_secret: str = os.getenv("DASHBOARD_SESSION_SECRET", "").strip()
    admin_users_json: str = os.getenv("ADMIN_USERS_JSON", "").strip()
    evidence_selection: bool = os.getenv("EVIDENCE_SELECTION", "1").lower() in {"1", "true", "yes"}
    # Cross-encoder reranking. Off by default: measured on this corpus it costs
    # 60s per question with BAAI/bge-reranker-base (44s at max_length=256) and
    # 7.6s with ms-marco-MiniLM-L-6-v2, against a pool of ~66 chunks on CPU.
    # MPS was slower than CPU at this batch size. Enable only with a GPU or a
    # smaller candidate pool.
    cross_encoder_rerank: bool = os.getenv("CROSS_ENCODER_RERANK", "0").lower() in {"1", "true", "yes"}
    cross_encoder_model: str = os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-base").strip()
    cross_encoder_max_length: int = int(os.getenv("CROSS_ENCODER_MAX_LENGTH", "512"))
    cross_encoder_top_n: int = int(os.getenv("CROSS_ENCODER_TOP_N", "40"))
    evidence_selection_model: str = os.getenv("EVIDENCE_SELECTION_MODEL", "").strip()
    # Answers scoring below this retrieval strength are kept for staff review.
    # Set FLAG_MIN_SCORE_GAP above 0 to also flag near-tied top chunks.
    flag_min_top_score: float = float(os.getenv("FLAG_MIN_TOP_SCORE", "0.90"))
    flag_min_score_gap: float = float(os.getenv("FLAG_MIN_SCORE_GAP", "0"))
    # With Supabase configured the local JSONL is redundant, and on an
    # ephemeral Space it is lost on restart anyway. Keep it for local dev.
    chat_log_to_file: bool = os.getenv("CHAT_LOG_TO_FILE", "").lower() in {"1", "true", "yes"}
    debug_mode: bool = os.getenv("FLASK_DEBUG", "0") == "1"
    chat_rate_limit_count: int = int(os.getenv("CHAT_RATE_LIMIT_COUNT", "10"))
    chat_rate_limit_window_seconds: int = int(os.getenv("CHAT_RATE_LIMIT_WINDOW_SECONDS", "60"))
    suggestions_rate_limit_count: int = int(os.getenv("SUGGESTIONS_RATE_LIMIT_COUNT", "30"))
    suggestions_rate_limit_window_seconds: int = int(os.getenv("SUGGESTIONS_RATE_LIMIT_WINDOW_SECONDS", "60"))
    suggestions_verify_retrieval: bool = os.getenv("SUGGESTIONS_VERIFY_RETRIEVAL", "0").lower() in {"1", "true", "yes"}
    conversation_ttl_seconds: int = int(os.getenv("CONVERSATION_TTL_SECONDS", "3600"))


class SourceDocument(dict):
    pass


class ConversationTurn(dict):
    pass


PROJECT_ROOT = Path(__file__).resolve().parent
CHAT_LOG_PATH = PROJECT_ROOT / "logs" / "chat_events.jsonl"
