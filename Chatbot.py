from __future__ import annotations

import math
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import chromadb
from chromadb.api.models.Collection import Collection
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

try:
    from flask import Flask, jsonify, render_template, request
except ImportError:  # pragma: no cover - dependency availability depends on the runtime
    Flask = None
    jsonify = None
    render_template = None
    request = None

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - dependency availability depends on the runtime
    genai = None

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - dependency availability depends on the runtime
    PdfReader = None


LLMCallable = Callable[[str], str]


class ChatbotConfig:
    collection_name: str = "docs"
    persist_directory: str = "./chroma_db"
    seed_documents_directory: str = os.getenv("SEED_DOCUMENTS_DIRECTORY", "./SEED_DOCUMENTS")
    force_reindex: bool = os.getenv("FORCE_REINDEX", "").lower() in {"1", "true", "yes"}
    embedding_model_name: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    summary_chunk_size: int = 1400
    summary_chunk_overlap: int = 140
    top_k: int = 5
    retrieval_candidate_pool: int = 24
    recent_history_turns: int = int(os.getenv("RECENT_HISTORY_TURNS", "4"))
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
    gemini_temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
    web_host: str = os.getenv("CHATBOT_HOST", "127.0.0.1")
    web_port: int = int(os.getenv("CHATBOT_PORT", "8000"))


class SourceDocument(dict):
    pass


class ConversationTurn(dict):
    pass


class RetrievalChatbot:
    def __init__(self, llm_callable: LLMCallable, config: Optional[ChatbotConfig] = None) -> None:
        self.config = config or ChatbotConfig()
        self.llm_callable = llm_callable
        self.embedder = SentenceTransformer(self.config.embedding_model_name)
        self.client = chromadb.PersistentClient(path=self.config.persist_directory)
        self.collection = self._get_or_create_collection()
        self.search_records: list[dict] = []
        self.bm25_idf: dict[str, float] = {}
        self.avg_document_length: float = 0.0
        self.refresh_search_index()

    def _get_or_create_collection(self) -> Collection:
        return self.client.get_or_create_collection(name=self.config.collection_name)

    def reset_collection(self) -> None:
        self.client.delete_collection(name=self.config.collection_name)
        self.collection = self._get_or_create_collection()
        self.refresh_search_index()

    def chunk_documents(self, documents: list[str]) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ".", " "],
        )
        return splitter.split_text("\n\n".join(documents))

    def split_document_into_chunks(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "],
        )
        return splitter.split_text(text)

    def index_documents(self, documents: list[SourceDocument]) -> None:
        existing_ids = set(self.collection.get(include=[])["ids"])
        new_ids: list[str] = []
        new_chunks: list[str] = []
        new_embeddings: list[list[float]] = []
        new_metadatas: list[dict] = []

        for document in documents:
            text = document["text"]
            if not text.strip():
                continue

            chunk_plans = [
                ("detail", self.config.chunk_size, self.config.chunk_overlap),
                ("summary", self.config.summary_chunk_size, self.config.summary_chunk_overlap),
            ]

            for chunk_level, chunk_size, chunk_overlap in chunk_plans:
                chunks = self.split_document_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if not chunks:
                    continue

                for index, chunk_text in enumerate(chunks):
                    chunk_id = f"{document['source_path']}::{chunk_level}-chunk-{index}"
                    if chunk_id in existing_ids:
                        continue

                    chunk_text_for_embedding = self.build_chunk_text_for_embedding(
                        document=document,
                        chunk_text=chunk_text,
                        chunk_level=chunk_level,
                    )
                    new_ids.append(chunk_id)
                    new_chunks.append(chunk_text_for_embedding)
                    new_embeddings.append(self.embedder.encode([chunk_text_for_embedding], convert_to_numpy=True)[0].tolist())
                    new_metadatas.append(
                        {
                            "source_path": document["source_path"],
                            "source_url": document["source_url"],
                            "title": document["title"],
                            "category": document["category"],
                            "folder_label": self.get_folder_label(document["source_path"]),
                            "document_type": document["document_type"],
                            "chunk_index": index,
                            "chunk_level": chunk_level,
                        }
                    )

        if new_ids:
            self.collection.add(
                ids=new_ids,
                documents=new_chunks,
                embeddings=new_embeddings,
                metadatas=new_metadatas,
            )
            self.refresh_search_index()

    def build_chunk_text_for_embedding(self, document: SourceDocument, chunk_text: str, chunk_level: str) -> str:
        labels = [document.get("title", "").strip(), document.get("category", "").strip()]
        folder_label = self.get_folder_label(document.get("source_path", ""))
        if folder_label:
            labels.append(folder_label)
        labels.append("Summary" if chunk_level == "summary" else "Detail")

        cleaned_labels: list[str] = []
        for label in labels:
            if not label or label == "SEED_DOCUMENTS":
                continue
            if label not in cleaned_labels:
                cleaned_labels.append(label)

        label_header = " | ".join(cleaned_labels)
        if not label_header:
            return chunk_text

        return f"Document Labels: {label_header}\n\n{chunk_text}"

    def get_folder_label(self, source_path: str) -> str:
        path = Path(source_path)
        if len(path.parts) <= 2:
            return ""
        return path.parts[-2]

    def refresh_search_index(self) -> None:
        self.search_records = []
        self.bm25_idf = {}
        self.avg_document_length = 0.0

        if self.collection.count() == 0:
            return

        stored = self.collection.get(include=["documents", "metadatas"])
        ids = stored.get("ids", [])
        documents = stored.get("documents", [])
        metadatas = stored.get("metadatas", [])
        if not ids or not documents:
            return

        document_frequency: Counter[str] = Counter()
        total_length = 0

        for chunk_id, document, metadata in zip(ids, documents, metadatas):
            metadata = metadata or {}
            tokens = self.tokenize_for_bm25(document)
            term_counts = Counter(tokens)
            document_length = len(tokens)

            self.search_records.append(
                {
                    "id": chunk_id,
                    "document": document,
                    "metadata": metadata,
                    "term_counts": term_counts,
                    "length": document_length,
                }
            )

            total_length += document_length
            for token in term_counts:
                document_frequency[token] += 1

        if not self.search_records:
            return

        self.avg_document_length = total_length / len(self.search_records) if total_length else 1.0
        document_count = len(self.search_records)
        self.bm25_idf = {
            token: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def tokenize_for_bm25(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def retrieve_dense_candidates(self, query: str, limit: int) -> list[dict]:
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(limit, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunk_ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        candidates: list[dict] = []
        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(chunk_ids, documents, metadatas, distances),
            start=1,
        ):
            candidates.append(
                {
                    "id": chunk_id,
                    "document": document,
                    "metadata": metadata or {},
                    "dense_rank": rank,
                    "dense_distance": float(distance),
                }
            )

        return candidates

    def retrieve_bm25_candidates(self, query: str, limit: int) -> list[dict]:
        if not self.search_records:
            return []

        query_terms = self.tokenize_for_bm25(query)
        if not query_terms:
            return []

        scored_candidates: list[dict] = []
        k1 = 1.5
        b = 0.75
        unique_terms = list(dict.fromkeys(query_terms))

        for record in self.search_records:
            document_length = record["length"] or 1
            term_counts: Counter[str] = record["term_counts"]
            score = 0.0

            for term in unique_terms:
                term_frequency = term_counts.get(term, 0)
                if term_frequency == 0:
                    continue

                idf = self.bm25_idf.get(term, 0.0)
                numerator = term_frequency * (k1 + 1)
                denominator = term_frequency + k1 * (
                    1 - b + b * (document_length / max(self.avg_document_length, 1.0))
                )
                score += idf * (numerator / denominator)

            if score > 0:
                scored_candidates.append(
                    {
                        "id": record["id"],
                        "document": record["document"],
                        "metadata": record["metadata"],
                        "bm25_score": score,
                    }
                )

        scored_candidates.sort(key=lambda candidate: candidate["bm25_score"], reverse=True)

        for rank, candidate in enumerate(scored_candidates, start=1):
            candidate["bm25_rank"] = rank

        return scored_candidates[:limit]

    def fuse_candidates(self, query_profile: dict, dense_candidates: list[dict], bm25_candidates: list[dict]) -> list[dict]:
        fused: dict[str, dict] = {}
        dense_weight, bm25_weight = self.get_hybrid_weights(query_profile)
        rrf_k = 60

        for candidate in dense_candidates:
            fused[candidate["id"]] = {
                "id": candidate["id"],
                "document": candidate["document"],
                "metadata": candidate["metadata"],
                "dense_rank": candidate.get("dense_rank"),
                "dense_distance": candidate.get("dense_distance"),
                "bm25_rank": None,
                "bm25_score": 0.0,
            }

        for candidate in bm25_candidates:
            record = fused.setdefault(
                candidate["id"],
                {
                    "id": candidate["id"],
                    "document": candidate["document"],
                    "metadata": candidate["metadata"],
                    "dense_rank": None,
                    "dense_distance": None,
                    "bm25_rank": None,
                    "bm25_score": 0.0,
                },
            )
            record["bm25_rank"] = candidate.get("bm25_rank")
            record["bm25_score"] = candidate.get("bm25_score", 0.0)

        fused_candidates: list[dict] = []
        for candidate in fused.values():
            hybrid_score = 0.0
            if candidate["dense_rank"] is not None:
                hybrid_score += dense_weight / (rrf_k + candidate["dense_rank"])
            if candidate["bm25_rank"] is not None:
                hybrid_score += bm25_weight / (rrf_k + candidate["bm25_rank"])
            candidate["hybrid_score"] = hybrid_score
            fused_candidates.append(candidate)

        fused_candidates.sort(key=lambda candidate: candidate["hybrid_score"], reverse=True)
        return fused_candidates

    def get_hybrid_weights(self, query_profile: dict) -> tuple[float, float]:
        if query_profile["topic_bias"]["keywords"] and not query_profile["is_broad"]:
            return 1.0, 1.15
        if query_profile["is_broad"]:
            return 1.15, 1.0
        return 1.0, 1.0

    def retrieve_context(self, query: str, top_k: Optional[int] = None) -> tuple[list[str], list[dict]]:
        if self.collection.count() == 0:
            return [], []

        query_profile = self.classify_query(query)
        requested_top_k = top_k or self.config.top_k
        candidate_pool = max(requested_top_k * 4, self.config.retrieval_candidate_pool)
        dense_candidates = self.retrieve_dense_candidates(query, limit=candidate_pool)
        bm25_candidates = self.retrieve_bm25_candidates(query, limit=candidate_pool)
        fused_candidates = self.fuse_candidates(query_profile=query_profile, dense_candidates=dense_candidates, bm25_candidates=bm25_candidates)
        reranked_candidates = self.rerank_candidates(query=query, candidates=fused_candidates, query_profile=query_profile)
        context_blocks: list[str] = []
        metadata_blocks: list[dict] = []

        for candidate in reranked_candidates[:requested_top_k]:
            chunk_text = candidate["document"]
            metadata = candidate["metadata"]
            metadata = metadata or {}
            source_url = metadata.get("source_url", "URL not provided")
            title = metadata.get("title", "Untitled source")
            source_path = metadata.get("source_path", "Unknown source")
            chunk_index = metadata.get("chunk_index", "?")
            chunk_level = metadata.get("chunk_level", "detail")
            context_blocks.append(
                f"Title: {title}\n"
                f"Source URL: {source_url}\n"
                f"Source Path: {source_path}\n"
                f"Chunk Level: {chunk_level}\n"
                f"Chunk Index: {chunk_index}\n\n"
                f"{chunk_text}"
            )
            metadata_blocks.append(metadata)

        return context_blocks, metadata_blocks

    def rerank_candidates(self, query: str, candidates: list[dict], query_profile: dict) -> list[dict]:
        topic_bias = query_profile["topic_bias"]
        reranked: list[dict] = []

        for candidate in candidates:
            document = candidate["document"]
            metadata = candidate["metadata"] or {}
            title = metadata.get("title", "")
            category = metadata.get("category", "")
            folder_label = metadata.get("folder_label") or self.get_folder_label(metadata.get("source_path", ""))
            chunk_level = metadata.get("chunk_level", "detail")
            score = float(candidate.get("hybrid_score", 0.0))

            if title in topic_bias["preferred_titles"]:
                score += 1.35
            if category in topic_bias["preferred_categories"]:
                score += 0.45
            if folder_label in topic_bias["preferred_folders"]:
                score += 0.65
            if chunk_level == "summary" and topic_bias["prefer_summary"]:
                score += 0.75
            if chunk_level == "detail" and not topic_bias["prefer_summary"]:
                score += 0.2

            lowered_document = document.lower()
            keyword_hits = sum(1 for keyword in topic_bias["keywords"] if keyword in lowered_document)
            score += keyword_hits * 0.12

            reranked.append(
                {
                    "document": document,
                    "metadata": metadata,
                    "distance": candidate.get("dense_distance"),
                    "score": score,
                }
            )

        reranked.sort(key=lambda candidate: candidate["score"], reverse=True)
        return reranked

    def classify_query(self, query: str) -> dict:
        lowered_query = query.lower()
        broad_markers = (
            "overview",
            "tell me about",
            "what is",
            "what are",
            "summarize",
            "general",
            "overall",
            "projects",
            "initiatives",
            "leadership",
            "staff",
            "people",
        )
        is_broad = len(query.split()) <= 8 or any(marker in lowered_query for marker in broad_markers)
        topic_bias = self.detect_query_bias(query)
        topic_bias["prefer_summary"] = bool(is_broad or topic_bias["prefer_summary"])
        return {
            "is_broad": is_broad,
            "topic_bias": topic_bias,
        }

    def detect_query_bias(self, query: str) -> dict[str, list[str]]:
        lowered_query = query.lower()
        bias = {
            "preferred_titles": [],
            "preferred_categories": [],
            "preferred_folders": [],
            "keywords": [],
            "prefer_summary": False,
        }

        if any(term in lowered_query for term in ("project", "projects", "initiative", "initiatives", "program", "programs")):
            bias["preferred_titles"].extend(["Projects"])
            bias["preferred_folders"].extend(["Annual Reports"])
            bias["keywords"].extend(["project", "projects", "initiative", "initiatives", "program", "programs"])
            bias["prefer_summary"] = True

        if any(term in lowered_query for term in ("staff", "people", "team", "employees")):
            bias["preferred_titles"].extend(["Staff", "SSLAbout"])
            bias["keywords"].extend(["staff", "people", "team", "director"])
            bias["prefer_summary"] = True

        if any(term in lowered_query for term in ("student", "students", "intern", "interns")):
            bias["preferred_titles"].extend(["StudentsInterns"])
            bias["keywords"].extend(["student", "students", "intern", "interns"])
            bias["prefer_summary"] = True

        if any(term in lowered_query for term in ("board", "director", "directors", "leadership", "leaders")):
            bias["preferred_titles"].extend(["BoardOfDirectors", "SSLAbout"])
            bias["keywords"].extend(["board", "director", "directors", "leadership", "leaders"])
            bias["prefer_summary"] = True

        if any(term in lowered_query for term in ("publication", "publications", "paper", "papers", "report", "reports", "research")):
            bias["preferred_folders"].extend(["Publications", "Annual Reports"])
            bias["preferred_categories"].extend(["Publications", "Annual Reports"])
            bias["keywords"].extend(["publication", "publications", "report", "reports", "research", "paper", "papers"])
            bias["prefer_summary"] = True

        return {
            "preferred_titles": list(dict.fromkeys(bias["preferred_titles"])),
            "preferred_categories": list(dict.fromkeys(bias["preferred_categories"])),
            "preferred_folders": list(dict.fromkeys(bias["preferred_folders"])),
            "keywords": list(dict.fromkeys(bias["keywords"])),
            "prefer_summary": bias["prefer_summary"],
        }

    def build_prompt(
        self,
        user_message: str,
        retrieved_context: list[str],
        recent_history: Optional[list[ConversationTurn]] = None,
        rewritten_query: Optional[str] = None,
    ) -> str:
        retrieved_text = "\n\n".join(retrieved_context) if retrieved_context else "No relevant context found."
        history_text = format_recent_history(recent_history or [])
        history_section = f"\nRecent conversation:\n{history_text}\n" if history_text else ""
        rewritten_query_section = (
            f"\nResolved retrieval query:\n{rewritten_query}\n"
            if rewritten_query and rewritten_query.strip().lower() != user_message.strip().lower()
            else ""
        )
        return f"""
You are the Sustainable Labs retrieval assistant. Answer only from the provided retrieved context.
If the answer is not supported by the context, say you do not have enough information.
If the user asks a follow-up that remains unclear, ask a brief clarifying question instead of guessing.
Use the recent conversation only when it helps resolve ambiguous follow-up references.
{history_section}{rewritten_query_section}

Retrieved context:
{retrieved_text}

Question:
{user_message}
""".strip()

    def answer(self, user_message: str, recent_history: Optional[list[ConversationTurn]] = None) -> dict:
        recent_history = recent_history or []
        is_follow_up_ambiguous = self.is_ambiguous_query(user_message)
        rewritten_query = self.rewrite_query_if_needed(user_message=user_message, recent_history=recent_history)
        retrieval_k = self.choose_top_k(rewritten_query)
        retrieved_context, retrieved_metadata = self.retrieve_context(rewritten_query, top_k=retrieval_k)

        if self.should_ask_clarifying_question(
            original_query=user_message,
            rewritten_query=rewritten_query,
            retrieved_context=retrieved_context,
            retrieved_metadata=retrieved_metadata,
        ):
            return {
                "reply": self.build_clarifying_question(user_message=user_message, recent_history=recent_history),
                "sources": self.extract_sources(retrieved_metadata),
            }

        prompt = self.build_prompt(
            user_message=user_message,
            retrieved_context=retrieved_context,
            recent_history=recent_history if is_follow_up_ambiguous else None,
            rewritten_query=rewritten_query,
        )
        return {
            "reply": self.llm_callable(prompt).strip(),
            "sources": self.extract_sources(retrieved_metadata),
        }

    def choose_top_k(self, query: str) -> int:
        return 8 if self.classify_query(query)["is_broad"] else self.config.top_k

    def rewrite_query_if_needed(self, user_message: str, recent_history: list[ConversationTurn]) -> str:
        if not recent_history or not self.is_ambiguous_query(user_message):
            return user_message

        history_text = format_recent_history(recent_history)
        rewrite_prompt = f"""
You rewrite ambiguous follow-up questions into standalone retrieval queries.
Use the recent conversation only to resolve references like "they", "them", "it", or "that".
Return only the rewritten query. If the user's question is still too unclear, return the original question unchanged.

Recent conversation:
{history_text}

User question:
{user_message}
""".strip()
        return self.llm_callable(rewrite_prompt).strip() or user_message

    def is_ambiguous_query(self, user_message: str) -> bool:
        lowered_query = user_message.lower().strip()
        pronoun_markers = {"it", "they", "them", "that", "those", "these", "he", "she", "this"}
        follow_up_markers = ("more", "explain", "elaborate", "expand", "tell me more", "go deeper")
        words = re.findall(r"\b\w+\b", lowered_query)
        return (
            any(word in pronoun_markers for word in words)
            or any(marker in lowered_query for marker in follow_up_markers)
            or len(words) <= 4
        )

    def should_ask_clarifying_question(
        self,
        original_query: str,
        rewritten_query: str,
        retrieved_context: list[str],
        retrieved_metadata: list[dict],
    ) -> bool:
        if not retrieved_context:
            return True

        ambiguous = self.is_ambiguous_query(original_query)
        distinct_sources = {
            (metadata.get("source_path"), metadata.get("title"))
            for metadata in retrieved_metadata
            if metadata
        }
        weak_context = len(retrieved_context) < 2 or len(distinct_sources) == 1
        rewrite_failed = ambiguous and rewritten_query.strip().lower() == original_query.strip().lower()
        return weak_context and (ambiguous or rewrite_failed)

    def build_clarifying_question(self, user_message: str, recent_history: list[ConversationTurn]) -> str:
        history_text = format_recent_history(recent_history) or "No recent conversation."
        clarification_prompt = f"""
You are helping a user clarify a question for retrieval.
Ask one short follow-up question that resolves the current ambiguity.
Do not answer the original question.
Do not mention embeddings, vector search, or retrieval internals.

Recent conversation:
{history_text}

        Current user question:
{user_message}
""".strip()
        return self.llm_callable(clarification_prompt).strip()

    def extract_sources(self, retrieved_metadata: list[dict]) -> list[dict]:
        sources: list[dict] = []
        seen: set[tuple[str, str]] = set()

        for metadata in retrieved_metadata:
            metadata = metadata or {}
            title = metadata.get("title", "Untitled source").strip() or "Untitled source"
            source_url = metadata.get("source_url", "").strip()
            source_path = metadata.get("source_path", "").strip() or "Unknown source"
            key = (title, source_url or source_path)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "title": title,
                    "url": source_url or "URL not provided",
                    "source_path": source_path,
                }
            )

        return sources


def call_gemini(prompt: str, model: Optional[str] = None, temperature: Optional[float] = None) -> str:
    if genai is None:
        raise ImportError("Install google-generativeai to use Gemini.")

    config = ChatbotConfig()
    if not config.gemini_api_key:
        raise ValueError("Set GEMINI_API_KEY before using Gemini.")

    genai.configure(api_key=config.gemini_api_key)
    model_name = model or config.gemini_model
    response_temperature = temperature if temperature is not None else config.gemini_temperature

    model_obj = genai.GenerativeModel(model_name)
    generation_config = genai.types.GenerationConfig(
        temperature=response_temperature,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
    )
    response = model_obj.generate_content(prompt, generation_config=generation_config)
    return response.text.strip()


def format_recent_history(recent_history: list[ConversationTurn]) -> str:
    return "\n".join(
        f"Turn {index} User: {turn['user']}\nTurn {index} Assistant: {turn['assistant']}"
        for index, turn in enumerate(recent_history, start=1)
        if turn.get("user") and turn.get("assistant")
    )


def load_seed_documents() -> list[SourceDocument]:
    seed_directory = Path(ChatbotConfig.seed_documents_directory)
    metadata_by_path = load_metadata_registry(seed_directory)

    if seed_directory.exists():
        documents: list[SourceDocument] = []
        supported_files = sorted(
            path for path in seed_directory.rglob("*") if path.is_file() and path.suffix.lower() in {".txt", ".pdf"}
        )

        for path in supported_files:
            if path.suffix.lower() == ".txt":
                text = path.read_text(encoding="utf-8")
            else:
                text = extract_pdf_text(path)

            cleaned_text = text.strip()
            if not cleaned_text:
                continue

            documents.append(
                build_document_record(
                    path=path,
                    seed_directory=seed_directory,
                    text=cleaned_text,
                    metadata_by_path=metadata_by_path,
                )
            )

        if documents:
            return documents

    return [
        SourceDocument(
            source_path="fallback://sustainable-labs-overview",
            source_url="URL not provided",
            title="Sustainable Labs Overview",
            category="Fallback",
            document_type="txt",
            text="Sustainable Labs helps teams explore sustainable AI workflows, practical research tooling, and responsible deployment patterns.",
        ),
        SourceDocument(
            source_path="fallback://rag-demo",
            source_url="URL not provided",
            title="RAG Demo Overview",
            category="Fallback",
            document_type="txt",
            text="This demo chatbot answers questions by retrieving relevant chunks from indexed source documents.",
        ),
    ]


def extract_pdf_text(path: Path) -> str:
    if PdfReader is None:
        raise ImportError("Install pypdf to ingest PDF seed documents.")

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(page.strip() for page in pages if page.strip())


def load_metadata_registry(seed_directory: Path) -> dict[str, dict]:
    metadata_path = seed_directory / "metadata_template.json"
    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    documents = payload.get("documents", [])
    return {
        str(Path(entry["source_path"]).as_posix()): entry
        for entry in documents
        if isinstance(entry, dict) and entry.get("source_path")
    }


def build_document_record(path: Path, seed_directory: Path, text: str, metadata_by_path: dict[str, dict]) -> SourceDocument:
    project_relative_path = Path("SEED_DOCUMENTS") / path.relative_to(seed_directory)
    metadata = metadata_by_path.get(str(project_relative_path.as_posix()), {})

    url = metadata.get("url", "").strip()
    notes = metadata.get("notes", "").strip()
    fallback_url = notes if notes.startswith("http") else ""
    effective_url = url or fallback_url or "URL not provided"

    title = metadata.get("title", path.stem).strip() or path.stem
    category = metadata.get("category", "Uncategorized").strip() or "Uncategorized"
    document_type = metadata.get("document_type", path.suffix.lstrip(".")).strip() or path.suffix.lstrip(".")

    return SourceDocument(
        source_path=project_relative_path.as_posix(),
        source_url=effective_url,
        title=title,
        category=category,
        document_type=document_type,
        text=text,
    )


def create_chatbot(config: Optional[ChatbotConfig] = None) -> RetrievalChatbot:
    resolved_config = config or ChatbotConfig()
    chatbot = RetrievalChatbot(llm_callable=call_gemini, config=resolved_config)
    if resolved_config.force_reindex:
        chatbot.reset_collection()
        chatbot.index_documents(load_seed_documents())
    elif chatbot.collection.count() == 0:
        chatbot.index_documents(load_seed_documents())
    return chatbot


def create_app() -> Flask:
    if Flask is None:
        raise ImportError("Install Flask to run the local web demo.")

    config = ChatbotConfig()
    chatbot = create_chatbot(config)
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True) or {}
        user_message = str(payload.get("message", "")).strip()
        recent_history = payload.get("recent_history", [])
        if not user_message:
            return jsonify({"error": "Message is required."}), 400

        try:
            history_window = max(config.recent_history_turns, 1)
            safe_recent_history = [
                ConversationTurn(user=str(turn.get("user", "")).strip(), assistant=str(turn.get("assistant", "")).strip())
                for turn in recent_history[-history_window:]
                if isinstance(turn, dict)
            ]
            result = chatbot.answer(user_message, recent_history=safe_recent_history)
        except Exception as exc:  # pragma: no cover - user-facing error path
            return jsonify({"error": str(exc)}), 500

        return jsonify(result)

    return app


def main() -> None:
    app = create_app()
    config = ChatbotConfig()
    app.run(debug=True, host=config.web_host, port=config.web_port)


if __name__ == "__main__":
    main()
