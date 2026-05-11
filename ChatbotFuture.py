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
    collection_name: str = os.getenv("CHATBOT_FUTURE_COLLECTION", "docs_future")
    persist_directory: str = os.getenv("CHATBOT_FUTURE_CHROMA_DB", "./chroma_db_future")
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
    web_host: str = os.getenv("CHATBOT_FUTURE_HOST", os.getenv("CHATBOT_HOST", "127.0.0.1"))
    web_port: int = int(os.getenv("CHATBOT_FUTURE_PORT", "8001"))
    route_debug_enabled: bool = os.getenv("CHATBOT_FUTURE_ROUTE_DEBUG", os.getenv("CHATBOT_ROUTE_DEBUG", "")).lower() in {
        "1", "true", "yes"
    }


class SourceDocument(dict):
    pass


class ConversationTurn(dict):
    pass


class RetrievalChatbot:
    MAX_CHROMA_BATCH_SIZE = 5000

    def __init__(self, llm_callable: LLMCallable, config: Optional[ChatbotConfig] = None) -> None:
        self.config = config or ChatbotConfig()
        self.llm_callable = llm_callable
        self.embedder = SentenceTransformer(self.config.embedding_model_name)
        self.client = chromadb.PersistentClient(path=self.config.persist_directory)
        self.collection = self._get_or_create_collection()
        self.search_records: list[dict] = []
        self.document_registry: list[dict] = []
        self.entity_registry: list[dict] = []
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

    def slugify(self, value: str) -> str:
        cleaned = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE).strip().lower()
        slug = re.sub(r"[-\s]+", "-", cleaned)
        return slug or "section"

    def normalize_paragraphs(self, text: str) -> list[str]:
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", text) if paragraph.strip()]
        return [paragraph for paragraph in paragraphs if paragraph]

    def is_probable_person_name(self, value: str) -> bool:
        candidate = re.sub(r"\([^)]*\)", "", value).strip(" ,.;:-")
        if not candidate or ":" in candidate or len(candidate) > 90:
            return False

        lowered = candidate.lower()
        blocked_phrases = {
            "our staff",
            "students",
            "graduate students and interns",
            "ssl alumni",
            "external affiliates",
            "visiting scholars",
            "contact us",
            "external advisory board",
            "university affiliates",
            "the sustainable solutions lab",
        }
        if lowered in blocked_phrases:
            return False

        tokens = [token.strip(" ,.;:()[]{}\"'“”’") for token in candidate.split()]
        tokens = [token for token in tokens if token]
        if len(tokens) < 2 or len(tokens) > 6:
            return False

        blocked_tokens = {
            "director",
            "associate",
            "executive",
            "president",
            "consultant",
            "leader",
            "practice",
            "professor",
            "dean",
            "officer",
            "founder",
            "principal",
            "architect",
            "foundation",
            "university",
            "school",
            "institute",
            "hospital",
            "health",
            "care",
            "harm",
            "city",
            "resilience",
            "climate",
            "commercial",
            "solar",
            "program",
            "office",
            "faculty",
            "commission",
            "group",
            "energy",
            "lab",
            "solutions",
            "boston",
            "chair",
        }
        lowered_tokens = {token.lower() for token in tokens}
        if lowered_tokens & blocked_tokens:
            return False

        uppercase_tokens = 0
        for token in tokens:
            if not any(character.isalpha() for character in token):
                return False
            if token[0].isupper():
                uppercase_tokens += 1

        return uppercase_tokens >= max(2, len(tokens) - 1)

    def extract_heading_name(self, paragraph: str) -> str:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            return ""

        first_line = lines[0]
        candidate = re.sub(r"\([^)]*\)", "", first_line).strip(" ,.;:-")
        if "," in candidate:
            leading_segment = candidate.split(",", 1)[0].strip()
            if self.is_probable_person_name(leading_segment):
                return leading_segment

        if self.is_probable_person_name(candidate):
            return candidate

        return ""

    def looks_like_visual_caption(self, value: str) -> bool:
        lowered = value.lower().strip()
        if not lowered:
            return False

        if lowered == "photo avatar":
            return True

        caption_markers = (
            "woman ",
            "man ",
            "photo ",
            "wearing ",
            "standing in front of",
            "smiling at camera",
            "arms crossed",
            "in front of",
            "with long hair",
            "with short hair",
            "with glasses",
            "yellow shirt",
            "black jacket",
        )
        return len(value) <= 140 and any(marker in lowered for marker in caption_markers)

    def names_refer_to_same_person(self, current_name: str, candidate_name: str) -> bool:
        current_tokens = [token for token in re.findall(r"\w+", current_name.lower()) if token]
        candidate_tokens = [token for token in re.findall(r"\w+", candidate_name.lower()) if token]
        if not current_tokens or not candidate_tokens:
            return False
        if current_tokens == candidate_tokens:
            return True
        if current_tokens[:2] == candidate_tokens[:2]:
            return True
        if current_tokens[-1] == candidate_tokens[-1]:
            current_initials = [token[0] for token in current_tokens[:-1] if token]
            candidate_initials = [token[0] for token in candidate_tokens[:-1] if token]
            if current_initials and candidate_initials and current_initials == candidate_initials[: len(current_initials)]:
                return True
        return current_tokens[0] == candidate_tokens[0] and current_tokens[-1] == candidate_tokens[-1]

    def extract_person_name_from_line(self, line: str) -> str:
        normalized_line = re.sub(r"([a-z])([A-Z])", r"\1 \2", line).strip()
        sentence_match = re.match(r"^([A-Z][^\n]{1,100}?)\s+is\b", normalized_line)
        if sentence_match:
            sentence_name = sentence_match.group(1).strip(" ,.;:-")
            if self.is_probable_person_name(sentence_name):
                return sentence_name

        trailing_match = re.search(r"([A-Z][\w'’“”.\-]+(?:\s+[A-Z][\w'’“”.\-]+){1,5})$", normalized_line)
        if trailing_match:
            trailing_name = trailing_match.group(1).strip(" ,.;:-")
            if self.is_probable_person_name(trailing_name):
                return trailing_name

        return self.extract_heading_name(normalized_line)

    def extract_email_hint(self, lines: list[str]) -> str:
        for line in lines:
            email_match = re.search(r"mailto:([A-Za-z0-9._%+-]+)@", line, re.IGNORECASE)
            if email_match:
                return email_match.group(1).lower()

            plain_email_match = re.search(r"\b([A-Za-z0-9._%+-]+)@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", line)
            if plain_email_match:
                return plain_email_match.group(1).lower()

        return ""

    def score_name_against_hint(self, name: str, hint: str) -> int:
        if not name or not hint:
            return 0

        name_tokens = [token for token in re.findall(r"\w+", name.lower()) if token]
        hint_tokens = [token for token in re.split(r"[._-]+", hint.lower()) if token]
        return sum(1 for token in name_tokens if token in hint_tokens)

    def collect_person_name_candidates(self, lines: list[str]) -> list[dict]:
        candidates: list[dict] = []
        seen: set[str] = set()

        for line in lines:
            if self.looks_like_visual_caption(line):
                continue

            candidate_name = self.extract_person_name_from_line(line)
            if not candidate_name:
                continue

            normalized_name = self.slugify(candidate_name)
            if normalized_name in seen:
                continue
            seen.add(normalized_name)
            candidates.append(
                {
                    "name": candidate_name,
                    "line": line,
                    "exact": line.strip().strip(" ,.;:-") == candidate_name,
                }
            )

        return candidates

    def choose_best_person_name(self, lines: list[str], preferred_hint: str = "") -> str:
        candidates = self.collect_person_name_candidates(lines)
        if not candidates:
            return ""

        if preferred_hint:
            scored_candidates = []
            for index, candidate in enumerate(candidates):
                scored_candidates.append(
                    (
                        self.score_name_against_hint(candidate["name"], preferred_hint),
                        1 if candidate["exact"] else 0,
                        len(candidate["name"]),
                        -index,
                        candidate["name"],
                    )
                )
            best_match = max(scored_candidates)
            if best_match[0] > 0:
                return best_match[-1]

        ranked_candidates = []
        for index, candidate in enumerate(candidates):
            ranked_candidates.append(
                (
                    1 if candidate["exact"] else 0,
                    -index,
                    len(candidate["name"].split()),
                    len(candidate["name"]),
                    candidate["name"],
                )
            )

        return max(ranked_candidates)[-1]

    def build_structured_unit(
        self,
        document: SourceDocument,
        *,
        section_name: str,
        section_text: str,
        entity_type: str,
        section_index: int,
    ) -> SourceDocument:
        section_slug = self.slugify(section_name)
        # Include the section index so repeated headings in the same source
        # still produce stable, unique unit IDs.
        base_unit_id = f"{document['source_path']}#{entity_type}-{section_index}-{section_slug}"
        normalized_text = section_text.strip()
        if normalized_text and not normalized_text.startswith(section_name):
            normalized_text = f"{section_name}\n\n{normalized_text}"

        return SourceDocument(
            source_path=document["source_path"],
            source_url=document["source_url"],
            title=document["title"],
            category=document["category"],
            document_type=document["document_type"],
            text=normalized_text,
            unit_id=base_unit_id,
            section_name=section_name,
            entity_type=entity_type,
            section_index=section_index,
        )

    def split_project_sections(self, document: SourceDocument) -> list[SourceDocument]:
        lines = document["text"].splitlines()
        units: list[SourceDocument] = []
        current_lines: list[str] = []
        section_index = 0

        def flush_current() -> None:
            nonlocal current_lines, section_index
            section_text = "\n".join(current_lines).strip()
            current_lines = []
            if not section_text:
                return

            section_lines = [line.strip() for line in section_text.splitlines() if line.strip()]
            if not section_lines:
                return

            section_name = section_lines[0]
            units.append(
                self.build_structured_unit(
                    document,
                    section_name=section_name,
                    section_text=section_text,
                    entity_type="project",
                    section_index=section_index,
                )
            )
            section_index += 1

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("##"):
                lowered = stripped.lstrip("#").strip().lower()
                if lowered.startswith("end"):
                    flush_current()
                    continue

                if current_lines:
                    flush_current()
                continue

            if current_lines or stripped:
                current_lines.append(line)

        if current_lines:
            flush_current()

        return units

    def split_ssl_about_sections(self, document: SourceDocument) -> list[SourceDocument]:
        lines = [line.rstrip() for line in document["text"].splitlines()]
        if not lines:
            return []

        headings = {"Pursuing Climate Justice", "Our Vision", "What We Do", "Contact Us"}
        units: list[SourceDocument] = []
        current_heading = ""
        current_lines: list[str] = []
        section_index = 0

        def flush_current() -> None:
            nonlocal current_heading, current_lines, section_index
            section_text = "\n".join(line for line in current_lines if line.strip()).strip()
            if current_heading and section_text:
                units.append(
                    self.build_structured_unit(
                        document,
                        section_name=current_heading,
                        section_text=section_text,
                        entity_type="section",
                        section_index=section_index,
                    )
                )
                section_index += 1
            current_heading = ""
            current_lines = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            if line in headings:
                flush_current()
                current_heading = line
                current_lines = [line]
                continue
            if current_heading:
                current_lines.append(line)

        flush_current()
        return units

    def split_slide_sections(self, document: SourceDocument) -> list[SourceDocument]:
        lines = [line.rstrip() for line in document["text"].splitlines()]
        if not lines:
            return []

        units: list[SourceDocument] = []
        current_heading = ""
        current_lines: list[str] = []
        section_index = 0

        def flush_current() -> None:
            nonlocal current_heading, current_lines, section_index
            section_text = "\n".join(line for line in current_lines if line.strip()).strip()
            if current_heading and section_text:
                units.append(
                    self.build_structured_unit(
                        document,
                        section_name=current_heading,
                        section_text=section_text,
                        entity_type="section",
                        section_index=section_index,
                    )
                )
                section_index += 1
            current_heading = ""
            current_lines = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("SLIDE "):
                flush_current()
                heading = line.split("—", 1)[1].strip() if "—" in line else line
                current_heading = heading
                current_lines = [line]
                continue
            if current_heading:
                current_lines.append(line)

        flush_current()
        return units

    def split_people_sections(self, document: SourceDocument, entity_type: str = "person") -> list[SourceDocument]:
        lines = [line.strip() for line in document["text"].splitlines() if line.strip()]
        if not lines:
            return []

        section_breaks = {
            "our staff",
            "students",
            "graduate students and interns",
            "ssl alumni",
            "external affiliates",
            "visiting scholars",
            "contact us",
            "external advisory board",
            "university affiliates",
        }
        units: list[SourceDocument] = []
        current_name = ""
        current_lines: list[str] = []
        section_index = 0

        def flush_current() -> None:
            nonlocal current_name, current_lines, section_index
            if current_name and current_lines:
                section_text = "\n".join(line for line in current_lines if line.strip()).strip()
                if section_text:
                    units.append(
                        self.build_structured_unit(
                            document,
                            section_name=current_name,
                            section_text=section_text,
                            entity_type=entity_type,
                            section_index=section_index,
                        )
                    )
                    section_index += 1
            current_name = ""
            current_lines = []

        for line in lines:
            lowered = line.lower().strip()
            if lowered in section_breaks:
                flush_current()
                continue

            if self.looks_like_visual_caption(line):
                continue

            heading_name = self.extract_person_name_from_line(line)
            if heading_name:
                if current_name and self.names_refer_to_same_person(current_name, heading_name):
                    if len(heading_name) > len(current_name):
                        current_name = heading_name
                    if line not in current_lines:
                        current_lines.append(line)
                    continue

                flush_current()
                current_name = heading_name
                current_lines = [line]
                continue

            if current_name:
                current_lines.append(line)

        flush_current()
        return units

    def split_staff_sections(self, document: SourceDocument) -> list[SourceDocument]:
        lines = [line.strip() for line in document["text"].splitlines() if line.strip()]
        if not lines:
            return []

        units: list[SourceDocument] = []
        current_lines: list[str] = []
        current_entity_type = "staff_member"
        section_index = 0
        section_headers = {
            "our staff": "staff_member",
            "external affiliates": "affiliate",
            "visiting scholars": "visiting_scholar",
            "contact us": "contact",
        }

        def flush_current() -> None:
            nonlocal current_lines, section_index
            if not current_lines:
                return

            filtered_lines = [line for line in current_lines if not self.looks_like_visual_caption(line)]
            email_hint = self.extract_email_hint(filtered_lines)
            section_name = self.choose_best_person_name(filtered_lines, preferred_hint=email_hint)
            current_lines = []
            if not section_name or current_entity_type == "contact":
                return

            section_text = "\n".join(filtered_lines).strip()
            if not section_text:
                return

            units.append(
                self.build_structured_unit(
                    document,
                    section_name=section_name,
                    section_text=section_text,
                    entity_type=current_entity_type,
                    section_index=section_index,
                )
            )
            section_index += 1

        for line in lines:
            lowered = line.lower()
            if lowered in section_headers:
                flush_current()
                current_entity_type = section_headers[lowered]
                continue

            if lowered.startswith("contact us"):
                flush_current()
                current_entity_type = "contact"
                continue

            current_lines.append(line)
            if "mailto:" in lowered or "linkedin " in lowered or "linkedin.com/" in lowered:
                flush_current()

        flush_current()
        return units

    def split_board_sections(self, document: SourceDocument) -> list[SourceDocument]:
        paragraphs = self.normalize_paragraphs(document["text"])
        if not paragraphs:
            return []

        units: list[SourceDocument] = []
        current_name = ""
        current_paragraphs: list[str] = []
        section_index = 0
        section_breaks = {"external advisory board"}

        def flush_current() -> None:
            nonlocal current_name, current_paragraphs, section_index
            if current_name and current_paragraphs:
                section_text = "\n\n".join(current_paragraphs).strip()
                if section_text:
                    units.append(
                        self.build_structured_unit(
                            document,
                            section_name=current_name,
                            section_text=section_text,
                            entity_type="board_member",
                            section_index=section_index,
                        )
                    )
                    section_index += 1
            current_name = ""
            current_paragraphs = []

        for paragraph in paragraphs:
            lowered_paragraph = paragraph.lower().strip()
            if lowered_paragraph in section_breaks or lowered_paragraph.startswith("ssl’s direction and work is guided"):
                continue

            filtered_lines: list[str] = []
            for raw_line in paragraph.splitlines():
                line = raw_line.strip()
                if not line:
                    continue

                if self.looks_like_visual_caption(line):
                    candidate_name = self.extract_person_name_from_line(line)
                    if candidate_name:
                        filtered_lines.append(candidate_name)
                    continue

                filtered_lines.append(line)
            if not filtered_lines:
                continue

            candidate_name = self.choose_best_person_name(filtered_lines)
            line_is_name = False
            if candidate_name:
                for line in filtered_lines:
                    normalized_line = re.sub(r"([a-z])([A-Z])", r"\1 \2", line).strip(" ,.;:-")
                    leading_segment = normalized_line.split(",", 1)[0].strip()
                    if (
                        normalized_line == candidate_name
                        or leading_segment == candidate_name
                        or normalized_line.endswith(candidate_name)
                    ):
                        line_is_name = True
                        break

            is_header_paragraph = candidate_name and line_is_name and len(filtered_lines) <= 4 and len(paragraph) <= 260

            if is_header_paragraph:
                flush_current()
                current_name = candidate_name
                current_paragraphs = ["\n".join(filtered_lines)]
                continue

            if current_name:
                current_paragraphs.append("\n".join(filtered_lines))

        flush_current()
        return units

    def split_affiliate_sections(self, document: SourceDocument) -> list[SourceDocument]:
        paragraphs = self.normalize_paragraphs(document["text"])
        if not paragraphs:
            return []

        units: list[SourceDocument] = []
        section_index = 0
        intro_prefixes = {
            "university affiliates",
            "ssl university affiliates are faculty and staff",
        }

        for paragraph in paragraphs:
            lowered_paragraph = paragraph.lower().strip()
            if any(lowered_paragraph.startswith(prefix) for prefix in intro_prefixes):
                continue

            lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
            if not lines:
                continue

            filtered_lines = [line for line in lines if not self.looks_like_visual_caption(line) and line.lower() != "photo avatar"]
            if not filtered_lines:
                continue

            email_hint = self.extract_email_hint(filtered_lines)
            section_name = self.choose_best_person_name(filtered_lines, preferred_hint=email_hint)
            if not section_name:
                continue

            section_text = "\n".join(filtered_lines).strip()
            units.append(
                self.build_structured_unit(
                    document,
                    section_name=section_name,
                    section_text=section_text,
                    entity_type="affiliate",
                    section_index=section_index,
                )
            )
            section_index += 1

        return units

    def expand_structured_document(self, document: SourceDocument) -> list[SourceDocument]:
        source_path = document.get("source_path", "")
        source_name = Path(source_path).name

        if source_name == "Projects.txt":
            project_units = self.split_project_sections(document)
            return project_units or [document]

        if source_name == "SSLAbout.txt":
            about_units = self.split_ssl_about_sections(document)
            return about_units or [document]

        if self.get_folder_label(source_path) == "Annual Reports" and source_name.endswith(".txt"):
            slide_units = self.split_slide_sections(document)
            return slide_units or [document]

        if source_name == "StudentsInterns.txt":
            people_units = self.split_people_sections(document)
            return people_units or [document]

        if source_name == "Staff.txt":
            staff_units = self.split_staff_sections(document)
            return staff_units or [document]

        if source_name == "BoardOfDirectors.txt":
            board_units = self.split_board_sections(document)
            return board_units or [document]

        if source_name == "UniversityAffiliates.txt":
            affiliate_units = self.split_affiliate_sections(document)
            return affiliate_units or [document]

        return [document]

    def expand_structured_documents(self, documents: list[SourceDocument]) -> list[SourceDocument]:
        expanded_documents: list[SourceDocument] = []
        for document in documents:
            expanded_documents.extend(self.expand_structured_document(document))
        return expanded_documents

    def index_documents(self, documents: list[SourceDocument]) -> None:
        existing_ids = set(self.collection.get(include=[])["ids"])
        pending_ids: set[str] = set()
        new_ids: list[str] = []
        new_chunks: list[str] = []
        new_embeddings: list[list[float]] = []
        new_metadatas: list[dict] = []

        for document in self.expand_structured_documents(documents):
            text = document["text"]
            if not text.strip():
                continue

            document_key = document.get("unit_id", document["source_path"])

            chunk_plans = [
                ("detail", self.config.chunk_size, self.config.chunk_overlap),
                ("summary", self.config.summary_chunk_size, self.config.summary_chunk_overlap),
            ]

            for chunk_level, chunk_size, chunk_overlap in chunk_plans:
                chunks = self.split_document_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if not chunks:
                    continue

                for index, chunk_text in enumerate(chunks):
                    chunk_id = f"{document_key}::{chunk_level}-chunk-{index}"
                    if chunk_id in existing_ids or chunk_id in pending_ids:
                        continue

                    chunk_text_for_embedding = self.build_chunk_text_for_embedding(
                        document=document,
                        chunk_text=chunk_text,
                        chunk_level=chunk_level,
                    )
                    pending_ids.add(chunk_id)
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
                            "unit_id": document.get("unit_id", document["source_path"]),
                            "section_name": document.get("section_name", ""),
                            "entity_type": document.get("entity_type", ""),
                            "section_index": document.get("section_index", -1),
                            "chunk_index": index,
                            "chunk_level": chunk_level,
                        }
                    )

        if new_ids:
            for start_index in range(0, len(new_ids), self.MAX_CHROMA_BATCH_SIZE):
                end_index = start_index + self.MAX_CHROMA_BATCH_SIZE
                self.collection.add(
                    ids=new_ids[start_index:end_index],
                    documents=new_chunks[start_index:end_index],
                    embeddings=new_embeddings[start_index:end_index],
                    metadatas=new_metadatas[start_index:end_index],
                )
            self.refresh_search_index()

    def build_chunk_text_for_embedding(self, document: SourceDocument, chunk_text: str, chunk_level: str) -> str:
        labels = [document.get("title", "").strip(), document.get("category", "").strip()]
        folder_label = self.get_folder_label(document.get("source_path", ""))
        if folder_label:
            labels.append(folder_label)
        section_name = document.get("section_name", "").strip()
        if section_name:
            labels.append(section_name)
        entity_type = document.get("entity_type", "").strip()
        if entity_type:
            labels.append(entity_type.title())
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
        self.document_registry = []
        self.entity_registry = []
        self.bm25_idf = {}
        self.avg_document_length = 0.0

        if self.collection.count() == 0:
            return

        stored = self.collection.get(include=["documents", "metadatas", "embeddings"])
        ids = stored.get("ids", [])
        documents = stored.get("documents", [])
        metadatas = stored.get("metadatas", [])
        embeddings = stored.get("embeddings", [])
        if not ids or not documents:
            return

        document_frequency: Counter[str] = Counter()
        document_registry_map: dict[str, dict] = {}
        entity_registry_map: dict[str, dict] = {}
        total_length = 0

        for chunk_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            metadata = metadata or {}
            tokens = self.tokenize_for_bm25(document)
            term_counts = Counter(tokens)
            document_length = len(tokens)
            embedding_vector = embedding.tolist() if hasattr(embedding, "tolist") else (embedding if embedding is not None else [])
            embedding_norm = math.sqrt(sum(value * value for value in embedding_vector)) or 1.0

            self.search_records.append(
                {
                    "id": chunk_id,
                    "document": document,
                    "metadata": metadata,
                    "embedding": embedding_vector,
                    "embedding_norm": embedding_norm,
                    "term_counts": term_counts,
                    "length": document_length,
                }
            )

            source_path = metadata.get("source_path", "").strip()
            if source_path:
                registry_record = document_registry_map.setdefault(
                    source_path,
                    {
                        "source_path": source_path,
                        "source_url": metadata.get("source_url", "URL not provided"),
                        "title": metadata.get("title", "Untitled source"),
                        "category": metadata.get("category", "Uncategorized"),
                        "folder_label": metadata.get("folder_label") or self.get_folder_label(source_path),
                        "document_type": metadata.get("document_type", ""),
                        "chunk_count": 0,
                    },
                )
                registry_record["chunk_count"] += 1

            unit_id = metadata.get("unit_id", "").strip()
            section_name = metadata.get("section_name", "").strip()
            if unit_id and section_name:
                entity_record = entity_registry_map.setdefault(
                    unit_id,
                    {
                        "unit_id": unit_id,
                        "section_name": section_name,
                        "entity_type": metadata.get("entity_type", "").strip() or "entity",
                        "source_path": source_path or "Unknown source",
                        "source_url": metadata.get("source_url", "URL not provided"),
                        "title": metadata.get("title", "Untitled source"),
                        "category": metadata.get("category", "Uncategorized"),
                        "folder_label": metadata.get("folder_label") or self.get_folder_label(source_path),
                        "document_type": metadata.get("document_type", ""),
                        "section_index": metadata.get("section_index", -1),
                        "chunk_count": 0,
                        "summary_text": "",
                        "detail_text": "",
                    },
                )
                entity_record["chunk_count"] += 1
                if metadata.get("chunk_level") == "summary" and not entity_record["summary_text"]:
                    entity_record["summary_text"] = document
                if metadata.get("chunk_level") == "detail" and not entity_record["detail_text"]:
                    entity_record["detail_text"] = document

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
        self.document_registry = sorted(document_registry_map.values(), key=lambda record: record["source_path"])
        self.entity_registry = sorted(
            entity_registry_map.values(),
            key=lambda record: (
                record.get("source_path", ""),
                int(record.get("section_index", -1)),
                record.get("section_name", ""),
            ),
        )

    def tokenize_for_bm25(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def retrieve_dense_candidates(self, query: str, limit: int, query_route: Optional[dict] = None) -> list[dict]:
        records = self.filter_records_by_route(query_route)
        if not records:
            return []

        query_embedding = self.embedder.encode([query], convert_to_numpy=True)[0].tolist()
        query_norm = math.sqrt(sum(value * value for value in query_embedding)) or 1.0
        scored_candidates: list[dict] = []

        for record in records:
            embedding = record.get("embedding") or []
            if not embedding:
                continue

            numerator = sum(query_value * record_value for query_value, record_value in zip(query_embedding, embedding))
            cosine_similarity = numerator / (query_norm * max(record.get("embedding_norm", 1.0), 1e-8))
            scored_candidates.append(
                {
                    "id": record["id"],
                    "document": record["document"],
                    "metadata": record["metadata"] or {},
                    "dense_score": cosine_similarity,
                }
            )

        scored_candidates.sort(key=lambda candidate: candidate["dense_score"], reverse=True)

        candidates: list[dict] = []
        for rank, candidate in enumerate(scored_candidates[:limit], start=1):
            candidates.append(
                {
                    "id": candidate["id"],
                    "document": candidate["document"],
                    "metadata": candidate["metadata"],
                    "dense_rank": rank,
                    "dense_distance": float(1.0 - candidate["dense_score"]),
                    "dense_score": float(candidate["dense_score"]),
                }
            )

        return candidates

    def retrieve_bm25_candidates(self, query: str, limit: int, query_route: Optional[dict] = None) -> list[dict]:
        if not self.search_records:
            return []

        query_terms = self.tokenize_for_bm25(query)
        if not query_terms:
            return []

        scored_candidates: list[dict] = []
        k1 = 1.5
        b = 0.75
        unique_terms = list(dict.fromkeys(query_terms))
        records = self.filter_records_by_route(query_route)

        for record in records:
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
        if query_profile.get("routing_mode") == "hard":
            return 1.05, 1.05
        if not query_profile.get("prefer_summary", False):
            return 1.0, 1.15
        if query_profile.get("prefer_summary", False):
            return 1.15, 1.0
        return 1.0, 1.0

    def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        query_route: Optional[dict] = None,
    ) -> tuple[list[str], list[dict], dict]:
        if self.collection.count() == 0:
            return [], [], {
                "candidate_count": 0,
                "selected_count": 0,
                "distinct_source_count": 0,
                "top_score": 0.0,
                "second_score": 0.0,
                "score_gap": 0.0,
            }

        query_profile = query_route or self.default_query_route(query)
        requested_top_k = top_k or self.config.top_k
        candidate_pool = max(requested_top_k * 4, self.config.retrieval_candidate_pool)
        dense_candidates = self.retrieve_dense_candidates(query, limit=candidate_pool, query_route=query_profile)
        bm25_candidates = self.retrieve_bm25_candidates(query, limit=candidate_pool, query_route=query_profile)
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
            section_name = metadata.get("section_name", "")
            entity_type = metadata.get("entity_type", "")
            chunk_index = metadata.get("chunk_index", "?")
            chunk_level = metadata.get("chunk_level", "detail")
            context_blocks.append(
                f"Title: {title}\n"
                f"Source URL: {source_url}\n"
                f"Source Path: {source_path}\n"
                f"Section Name: {section_name or 'N/A'}\n"
                f"Entity Type: {entity_type or 'N/A'}\n"
                f"Chunk Level: {chunk_level}\n"
                f"Chunk Index: {chunk_index}\n\n"
                f"{chunk_text}"
            )
            metadata_blocks.append(metadata)

        distinct_source_count = len(
            {
                (metadata.get("source_path", ""), metadata.get("title", ""))
                for metadata in metadata_blocks
                if metadata
            }
        )
        top_score = float(reranked_candidates[0]["score"]) if reranked_candidates else 0.0
        second_score = float(reranked_candidates[1]["score"]) if len(reranked_candidates) > 1 else 0.0
        diagnostics = {
            "candidate_count": len(reranked_candidates),
            "selected_count": len(metadata_blocks),
            "distinct_source_count": distinct_source_count,
            "top_score": top_score,
            "second_score": second_score,
            "score_gap": top_score - second_score,
        }

        return context_blocks, metadata_blocks, diagnostics

    def rerank_candidates(self, query: str, candidates: list[dict], query_profile: dict) -> list[dict]:
        target_titles = set(query_profile.get("target_titles", []))
        target_categories = set(query_profile.get("target_categories", []))
        target_folders = set(query_profile.get("target_folders", []))
        target_source_paths = set(query_profile.get("target_source_paths", []))
        prefer_summary = bool(query_profile.get("prefer_summary", False))
        query_terms = [term for term in self.tokenize_for_bm25(query) if len(term) > 2]
        reranked: list[dict] = []

        for candidate in candidates:
            document = candidate["document"]
            metadata = candidate["metadata"] or {}
            title = metadata.get("title", "")
            category = metadata.get("category", "")
            folder_label = metadata.get("folder_label") or self.get_folder_label(metadata.get("source_path", ""))
            source_path = metadata.get("source_path", "")
            section_name = metadata.get("section_name", "")
            chunk_level = metadata.get("chunk_level", "detail")
            score = float(candidate.get("hybrid_score", 0.0))

            if source_path in target_source_paths:
                score += 1.55
            if title in target_titles:
                score += 1.2
            if category in target_categories:
                score += 0.45
            if folder_label in target_folders:
                score += 0.65
            if chunk_level == "summary" and prefer_summary:
                score += 0.75
            if chunk_level == "detail" and not prefer_summary:
                score += 0.2

            lowered_document = document.lower()
            exact_term_hits = sum(1 for term in query_terms if term in lowered_document)
            score += min(exact_term_hits, 8) * 0.05
            if section_name:
                lowered_section_name = section_name.lower()
                exact_section_hits = sum(1 for term in query_terms if term in lowered_section_name)
                score += min(exact_section_hits, 4) * 0.08

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

    def default_query_route(self, query: str) -> dict:
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
        return {
            "question_type": "broad_overview" if any(marker in lowered_query for marker in broad_markers) else "specific_fact",
            "routing_mode": "global",
            "prefer_summary": len(query.split()) <= 8 or any(marker in lowered_query for marker in broad_markers),
            "target_titles": [],
            "target_categories": [],
            "target_folders": [],
            "target_source_paths": [],
            "reason": "Fallback global retrieval route.",
        }

    def merge_query_routes(self, base_route: Optional[dict], override_route: Optional[dict], *, replace_targets: bool = False) -> dict:
        merged = dict(base_route or self.default_query_route(""))
        if not override_route:
            return merged

        target_fields = ("target_titles", "target_categories", "target_folders", "target_source_paths")
        override_has_targets = any(override_route.get(field) for field in target_fields)

        if replace_targets and override_has_targets:
            for field in target_fields:
                merged[field] = list(dict.fromkeys(override_route.get(field, [])))
            merged["routing_mode"] = override_route.get("routing_mode", merged.get("routing_mode", "global"))
        else:
            for field in target_fields:
                combined = list(dict.fromkeys((merged.get(field, []) or []) + (override_route.get(field, []) or [])))
                merged[field] = combined
            if override_has_targets and merged.get("routing_mode") == "global":
                merged["routing_mode"] = override_route.get("routing_mode", "soft")

        if merged.get("question_type") == "specific_fact" and override_route.get("question_type") in {"people_lookup", "follow_up", "publication_inventory", "list_inventory"}:
            merged["question_type"] = override_route["question_type"]
        merged["prefer_summary"] = bool(merged.get("prefer_summary", False) or override_route.get("prefer_summary", False))
        if override_route.get("reason"):
            merged["reason"] = override_route["reason"]

        return merged

    def get_route_catalog(self) -> dict[str, list[str]]:
        titles = sorted(
            {
                record["metadata"].get("title", "").strip()
                for record in self.search_records
                if record.get("metadata", {}).get("title")
            }
        )
        categories = sorted(
            {
                record["metadata"].get("category", "").strip()
                for record in self.search_records
                if record.get("metadata", {}).get("category")
            }
        )
        folders = sorted(
            {
                (record["metadata"].get("folder_label") or self.get_folder_label(record["metadata"].get("source_path", ""))).strip()
                for record in self.search_records
                if (record.get("metadata", {}).get("folder_label") or self.get_folder_label(record.get("metadata", {}).get("source_path", ""))).strip()
            }
        )
        source_paths = sorted(
            {
                record["metadata"].get("source_path", "").strip()
                for record in self.search_records
                if record.get("metadata", {}).get("source_path")
            }
        )
        entity_names = sorted(
            {
                entity.get("section_name", "").strip()
                for entity in self.entity_registry
                if entity.get("section_name")
            }
        )
        return {
            "titles": titles,
            "categories": categories,
            "folders": folders,
            "source_paths": source_paths,
            "entity_names": entity_names,
        }

    def parse_json_object(self, text: str) -> dict:
        text = text.strip()
        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fenced_match:
            return json.loads(fenced_match.group(1))

        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            return json.loads(brace_match.group(1))

        return json.loads(text)
    def normalize_query_route(self, route: dict, route_catalog: dict[str, list[str]], retrieval_query: str) -> dict:
        default_route = self.default_query_route(retrieval_query)
        normalized_route = {
            "question_type": str(route.get("question_type", default_route["question_type"])).strip() or default_route["question_type"],
            "routing_mode": str(route.get("routing_mode", default_route["routing_mode"])).strip().lower() or default_route["routing_mode"],
            "prefer_summary": bool(route.get("prefer_summary", default_route["prefer_summary"])),
            "target_titles": [
                title for title in route.get("target_titles", []) if isinstance(title, str) and title in route_catalog["titles"]
            ],
            "target_categories": [
                category
                for category in route.get("target_categories", [])
                if isinstance(category, str) and category in route_catalog["categories"]
            ],
            "target_folders": [
                folder for folder in route.get("target_folders", []) if isinstance(folder, str) and folder in route_catalog["folders"]
            ],
            "target_source_paths": [
                source_path
                for source_path in route.get("target_source_paths", [])
                if isinstance(source_path, str) and source_path in route_catalog["source_paths"]
            ],
            "reason": str(route.get("reason", default_route["reason"])).strip() or default_route["reason"],
        }

        if normalized_route["routing_mode"] not in {"hard", "soft", "global"}:
            normalized_route["routing_mode"] = default_route["routing_mode"]

        if not any(
            [
                normalized_route["target_titles"],
                normalized_route["target_categories"],
                normalized_route["target_folders"],
                normalized_route["target_source_paths"],
            ]
        ):
            normalized_route["routing_mode"] = "global"

        return normalized_route

    def normalize_query_plan(self, plan: dict, route_catalog: dict[str, list[str]], user_message: str) -> dict:
        normalized_route = self.normalize_query_route(plan, route_catalog, user_message)
        rewritten_query = str(plan.get("rewritten_query", user_message)).strip() or user_message
        clarification_options = [
            option.strip()
            for option in plan.get("clarification_options", [])
            if isinstance(option, str) and option.strip()
        ]
        unique_options = list(dict.fromkeys(clarification_options))[:4]
        needs_clarification = bool(plan.get("needs_clarification", False))
        clarifying_question = str(plan.get("clarifying_question", "")).strip()

        if needs_clarification and not clarifying_question:
            clarifying_question = "Can you clarify what you mean?"

        if not needs_clarification:
            clarifying_question = ""
            unique_options = []

        normalized_route.update(
            {
                "rewritten_query": rewritten_query,
                "needs_clarification": needs_clarification,
                "clarifying_question": clarifying_question,
                "clarification_options": unique_options,
            }
        )
        return normalized_route

    def filter_records_by_route(self, query_route: Optional[dict]) -> list[dict]:
        if not self.search_records:
            return []
        if not query_route or query_route.get("routing_mode") == "global":
            return self.search_records

        target_titles = set(query_route.get("target_titles", []))
        target_categories = set(query_route.get("target_categories", []))
        target_folders = set(query_route.get("target_folders", []))
        target_source_paths = set(query_route.get("target_source_paths", []))

        matched_records = []
        for record in self.search_records:
            metadata = record.get("metadata", {})
            source_path = metadata.get("source_path", "")
            folder_label = metadata.get("folder_label") or self.get_folder_label(source_path)
            if (
                source_path in target_source_paths
                or metadata.get("title", "") in target_titles
                or metadata.get("category", "") in target_categories
                or folder_label in target_folders
            ):
                matched_records.append(record)

        if not matched_records:
            return self.search_records

        if query_route.get("routing_mode") == "soft" and len(matched_records) < max(self.config.top_k * 2, 6):
            return self.search_records

        return matched_records

    def filter_documents_by_route(self, query_route: Optional[dict]) -> list[dict]:
        if not self.document_registry:
            return []
        if not query_route or query_route.get("routing_mode") == "global":
            return self.document_registry

        target_titles = set(query_route.get("target_titles", []))
        target_categories = set(query_route.get("target_categories", []))
        target_folders = set(query_route.get("target_folders", []))
        target_source_paths = set(query_route.get("target_source_paths", []))

        matched_documents = [
            document
            for document in self.document_registry
            if (
                document.get("source_path", "") in target_source_paths
                or document.get("title", "") in target_titles
                or document.get("category", "") in target_categories
                or document.get("folder_label", "") in target_folders
            )
        ]

        if not matched_documents:
            return self.document_registry

        if query_route.get("routing_mode") == "soft" and len(matched_documents) < 2:
            return self.document_registry

        return matched_documents

    def filter_entities_by_route(self, query_route: Optional[dict]) -> list[dict]:
        if not self.entity_registry:
            return []
        if not query_route or query_route.get("routing_mode") == "global":
            return self.entity_registry

        target_titles = set(query_route.get("target_titles", []))
        target_categories = set(query_route.get("target_categories", []))
        target_folders = set(query_route.get("target_folders", []))
        target_source_paths = set(query_route.get("target_source_paths", []))

        matched_entities = [
            entity
            for entity in self.entity_registry
            if (
                entity.get("source_path", "") in target_source_paths
                or entity.get("title", "") in target_titles
                or entity.get("category", "") in target_categories
                or entity.get("folder_label", "") in target_folders
            )
        ]

        if not matched_entities:
            return self.entity_registry

        if query_route.get("routing_mode") == "soft" and len(matched_entities) < 2:
            return self.entity_registry

        return matched_entities

    def strip_embedding_labels(self, text: str) -> str:
        if text.startswith("Document Labels:"):
            _, _, remainder = text.partition("\n\n")
            return remainder.strip() or text
        return text.strip()

    def extract_query_named_phrases(self, query: str) -> list[str]:
        phrases = re.findall(r"\b(?:[A-Z][\w'’.-]+(?:\s+[A-Z][\w'’.-]+)+)\b", query)
        cleaned_phrases: list[str] = []
        seen: set[str] = set()
        for phrase in phrases:
            cleaned = phrase.strip(" ,.;:-")
            if len(cleaned) < 5:
                continue
            normalized = cleaned.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            cleaned_phrases.append(cleaned)
        return cleaned_phrases

    def normalize_entity_name(self, value: str) -> str:
        stripped = re.sub(r"\([^)]*\)", "", value)
        normalized = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", stripped)).strip().lower()
        return normalized

    def normalize_query_aliases(self, query: str) -> str:
        normalized = query
        alias_patterns = (
            (r"\bexternal advisory board\b", "board of directors"),
            (r"\badvisory board\b", "board of directors"),
            (r"\buniversity affiliate(s)?\b", "affiliates"),
            (r"\bstudents and interns\b", "students interns"),
        )
        for pattern, replacement in alias_patterns:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        return normalized

    def role_group_in_query(self, query: str, group: str) -> bool:
        lowered = query.lower()
        if group == "board":
            return any(token in lowered for token in ("board", "advisory"))
        if group == "staff":
            return "staff" in lowered or "team" in lowered
        if group == "affiliate":
            return "affiliate" in lowered
        if group == "student":
            return any(token in lowered for token in ("student", "intern"))
        return False

    def find_matching_entities(self, user_message: str, entities: Optional[list[dict]] = None) -> list[dict]:
        rewritten_user_message = self.normalize_query_aliases(user_message)
        lowered_query = rewritten_user_message.lower()
        normalized_query = self.normalize_entity_name(rewritten_user_message)
        candidate_entities = entities or self.entity_registry
        matched_entities: list[dict] = []
        person_matched_entities: list[dict] = []
        surname_matches: list[dict] = []
        person_query_markers = (
            "dr ",
            "dr.",
            "professor",
            "director",
            "background",
            "research background",
            "bio",
            "biography",
            "who is",
            "role",
            "title",
        )
        person_query = any(marker in lowered_query for marker in person_query_markers)

        for entity in candidate_entities:
            section_name = entity.get("section_name", "").strip()
            if not section_name:
                continue

            entity_type = entity.get("entity_type", "").strip()
            full_name = section_name.lower()
            normalized_name = self.normalize_entity_name(section_name)
            name_tokens = [token for token in normalized_name.split() if token]
            is_person_entity = self.is_person_entity_type(entity_type)

            # Ignore generic single-token non-person section labels like "RESEARCH"
            # when the user is asking about a person.
            if person_query and not is_person_entity and len(name_tokens) < 2:
                continue

            if full_name in lowered_query or (normalized_name and normalized_name in normalized_query):
                if is_person_entity:
                    person_matched_entities.append(entity)
                matched_entities.append(entity)
                continue

            if len(name_tokens) >= 2:
                surname = name_tokens[-1]
                if len(surname) >= 4 and re.search(rf"\b{re.escape(surname)}\b", normalized_query):
                    surname_matches.append(entity)

        if person_matched_entities:
            return list(
                {
                    entity.get("unit_id", "").strip() or entity.get("section_name", "").strip(): entity
                    for entity in person_matched_entities
                }.values()
            )

        if matched_entities:
            return matched_entities

        unique_surname_matches = list(
            {
                entity.get("unit_id", "").strip() or entity.get("section_name", "").strip(): entity
                for entity in surname_matches
            }.values()
        )
        return unique_surname_matches if len(unique_surname_matches) == 1 else []

    def is_person_entity_type(self, entity_type: str) -> bool:
        return entity_type in {"person", "staff_member", "board_member", "affiliate", "visiting_scholar"}

    def is_project_entity_type(self, entity_type: str) -> bool:
        return entity_type == "project"

    def infer_subject_type_from_document(self, document: dict) -> str:
        title = (document.get("title", "") or "").lower()
        category = (document.get("category", "") or "").lower()
        folder_label = (document.get("folder_label", "") or "").lower()
        combined = f"{title} {category} {folder_label}"
        if any(marker in combined for marker in ("project", "initiative", "program")):
            return "project"
        if any(marker in combined for marker in ("publication", "report", "paper", "brief")):
            return "publication"
        if any(marker in combined for marker in ("staff", "board", "affiliate", "student", "intern")):
            return "people"
        return "topic"

    def find_matching_document_subjects(self, text: str) -> list[dict]:
        if not text or not self.document_registry:
            return []
        normalized_text = self.normalize_entity_name(text)
        if not normalized_text:
            return []

        matched_subjects: list[dict] = []
        seen: set[str] = set()
        for document in self.document_registry:
            title = (document.get("title", "") or "").strip()
            if not title:
                continue
            normalized_title = self.normalize_entity_name(title)
            title_tokens = [token for token in normalized_title.split() if token]
            if len(title_tokens) < 2 and len(normalized_title) < 7:
                continue
            if normalized_title not in normalized_text:
                continue

            source_path = (document.get("source_path", "") or "").strip()
            subject_id = f"doc::{source_path or normalized_title}"
            if subject_id in seen:
                continue
            seen.add(subject_id)
            matched_subjects.append(
                {
                    "subject_id": subject_id,
                    "label": title,
                    "subject_type": self.infer_subject_type_from_document(document),
                    "source_path": source_path or "Unknown source",
                    "source_url": document.get("source_url", "URL not provided"),
                    "title": title,
                    "category": document.get("category", "Uncategorized"),
                    "folder_label": document.get("folder_label", ""),
                    "entity_type": "document_subject",
                }
            )

        return matched_subjects

    def build_recent_entity_memory(self, recent_history: Optional[list[ConversationTurn]]) -> list[dict]:
        if not recent_history:
            return []

        memories: list[dict] = []
        for turn_index, turn in enumerate(recent_history, start=1):
            turn_entities_by_id: dict[str, dict] = {}
            turn_subjects_by_id: dict[str, dict] = {}
            for speaker in ("user", "assistant"):
                text = (turn.get(speaker) or "").strip()
                if not text:
                    continue
                for entity in self.find_matching_entities(text):
                    unit_id = entity.get("unit_id", "").strip()
                    if not unit_id:
                        continue
                    turn_entities_by_id.setdefault(unit_id, entity)
                    entity_type = entity.get("entity_type", "").strip()
                    if self.is_person_entity_type(entity_type):
                        subject_type = "person"
                    elif self.is_project_entity_type(entity_type):
                        subject_type = "project"
                    else:
                        subject_type = "topic"
                    turn_subjects_by_id.setdefault(
                        f"entity::{unit_id}",
                        {
                            "subject_id": f"entity::{unit_id}",
                            "label": entity.get("section_name", "").strip(),
                            "subject_type": subject_type,
                            "source_path": entity.get("source_path", "Unknown source"),
                            "source_url": entity.get("source_url", "URL not provided"),
                            "title": entity.get("title", "Untitled source"),
                            "category": entity.get("category", "Uncategorized"),
                            "folder_label": entity.get("folder_label", ""),
                            "entity_type": entity_type or "entity",
                            "entity_ref": entity,
                        },
                    )

                for subject in self.find_matching_document_subjects(text):
                    turn_subjects_by_id.setdefault(subject["subject_id"], subject)

            if not turn_entities_by_id and not turn_subjects_by_id:
                continue

            all_entities = list(turn_entities_by_id.values())
            all_subjects = [
                subject
                for subject in turn_subjects_by_id.values()
                if subject.get("label", "").strip()
            ]
            person_entities = [
                entity
                for entity in all_entities
                if self.is_person_entity_type(entity.get("entity_type", ""))
            ]
            project_entities = [
                entity
                for entity in all_entities
                if self.is_project_entity_type(entity.get("entity_type", ""))
            ]
            person_subjects = [subject for subject in all_subjects if subject.get("subject_type") == "person"]
            project_subjects = [subject for subject in all_subjects if subject.get("subject_type") == "project"]
            memories.append(
                {
                    "turn_index": turn_index,
                    "all_entities": all_entities,
                    "person_entities": person_entities,
                    "project_entities": project_entities,
                    "all_subjects": all_subjects,
                    "person_subjects": person_subjects,
                    "project_subjects": project_subjects,
                }
            )

        return memories

    def format_recent_entity_memory(self, recent_history: Optional[list[ConversationTurn]]) -> str:
        memories = self.build_recent_entity_memory(recent_history)
        if not memories:
            return "No recent entity memory."

        lines: list[str] = []
        for memory in memories[-4:]:
            people = ", ".join(entity.get("section_name", "") for entity in memory["person_entities"] if entity.get("section_name"))
            projects = ", ".join(entity.get("section_name", "") for entity in memory.get("project_entities", []) if entity.get("section_name"))
            entities = ", ".join(entity.get("section_name", "") for entity in memory["all_entities"] if entity.get("section_name"))
            subjects = ", ".join(subject.get("label", "") for subject in memory.get("all_subjects", []) if subject.get("label"))
            if people:
                lines.append(f"Turn {memory['turn_index']} people: {people}")
            if projects:
                lines.append(f"Turn {memory['turn_index']} projects: {projects}")
            if subjects:
                lines.append(f"Turn {memory['turn_index']} subjects: {subjects}")
            elif entities:
                lines.append(f"Turn {memory['turn_index']} entities: {entities}")

        return "\n".join(lines) if lines else "No recent entity memory."

    def build_entity_follow_up_rewrite(self, user_message: str, entity_name: str, entity_type: str = "") -> str:
        stripped_message = user_message.strip()
        lowered_message = stripped_message.lower()
        if not stripped_message:
            return entity_name

        rewritten = stripped_message
        substitution_patterns = [
            (r"\bthat project\b", entity_name),
            (r"\bthis project\b", entity_name),
            (r"\bthat initiative\b", entity_name),
            (r"\bthis initiative\b", entity_name),
            (r"\bthat program\b", entity_name),
            (r"\bthis program\b", entity_name),
            (r"\bthat person\b", entity_name),
            (r"\bthis person\b", entity_name),
            (r"\bthat one\b", entity_name),
            (r"\bthis one\b", entity_name),
            (r"\bthose people\b", entity_name),
            (r"\bthese people\b", entity_name),
            (r"\btheir\b", f"{entity_name}'s"),
            (r"\bthem\b", entity_name),
            (r"\bthey\b", entity_name),
            (r"\bit\b", entity_name),
        ]
        for pattern, replacement in substitution_patterns:
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)

        if rewritten != stripped_message:
            return rewritten

        person_follow_up_markers = (
            "research background",
            "background",
            "research",
            "focus",
            "practice",
            "bio",
            "biography",
            "role",
            "work",
            "tell me more",
            "more about",
            "who is",
        )
        if any(marker in lowered_message for marker in person_follow_up_markers):
            return f"What is {entity_name}'s {stripped_message.rstrip('?.!').lower()}?"

        if entity_type == "project" and any(marker in lowered_message for marker in ("tell me more", "more about", "overview", "what is", "what does")):
            return f"Tell me more about {entity_name}."

        return f"{entity_name}: {stripped_message}"

    def is_group_selection_follow_up(self, user_message: str) -> bool:
        lowered_query = user_message.lower()
        selection_markers = (
            "which one",
            "which of them",
            "which of those",
            "which of the",
            "who among",
            "who else",
            "which student",
            "which intern",
            "which person",
            "which board member",
            "which affiliate",
            "which staff",
        )
        singular_detail_markers = (
            "that person",
            "this person",
            "their research background",
            "their background",
            "their bio",
            "their role",
            "tell me more about one",
            "tell me more about them",
        )
        return any(marker in lowered_query for marker in selection_markers) and not any(
            marker in lowered_query for marker in singular_detail_markers
        )

    def is_ambiguous_query(self, user_message: str) -> bool:
        lowered_query = user_message.lower().strip()
        pronoun_markers = {"it", "they", "them", "that", "those", "these", "he", "she", "this"}
        follow_up_markers = ("more", "explain", "elaborate", "expand", "tell me more", "go deeper")
        words = re.findall(r"\b\w+\b", lowered_query)
        clear_topic_markers = {"ssl", "mission", "vision", "projects", "staff", "board", "publications", "contact"}
        return (
            any(word in pronoun_markers for word in words)
            or any(marker in lowered_query for marker in follow_up_markers)
            or (len(words) <= 4 and not any(word in clear_topic_markers for word in words))
        )

    def is_specific_entity_detail_query(self, user_message: str) -> bool:
        lowered_query = user_message.lower()
        detail_markers = (
            "what does",
            "what is",
            "tell me about",
            "focus",
            "research",
            "background",
            "bio",
            "biography",
            "practice",
            "role",
            "involved in",
            "working on",
            "works on",
        )
        list_markers = ("who are", "list", "name all", "how many", "count", "overview")
        return any(marker in lowered_query for marker in detail_markers) and not any(
            marker in lowered_query for marker in list_markers
        )

    def is_person_background_query(self, user_message: str) -> bool:
        lowered_query = user_message.lower()
        return any(marker in lowered_query for marker in ("background", "research background", "bio", "biography", "research"))

    def is_named_person_query(self, user_message: str) -> bool:
        matched_people = [
            entity
            for entity in self.find_matching_entities(user_message)
            if self.is_person_entity_type(entity.get("entity_type", ""))
        ]
        return len(matched_people) == 1

    def is_role_lookup_query(self, user_message: str) -> bool:
        lowered_query = user_message.lower()
        role_markers = ("director", "executive", "manager", "dean", "chair", "lead", "coordinator", "officer", "president")
        who_markers = ("who is", "who's", "who serves as", "which person is")
        return any(marker in lowered_query for marker in who_markers) and any(marker in lowered_query for marker in role_markers)

    def normalize_lookup_tokens(self, value: str) -> list[str]:
        stopwords = {
            "who", "what", "is", "the", "of", "and", "at", "for", "in", "a", "an",
            "ssl", "lab", "labs", "sustainable", "solutions", "currently", "serves", "as",
        }
        return [token for token in self.normalize_entity_name(value).split() if token and token not in stopwords]

    def find_related_source_paths_for_entity(self, entity: dict) -> list[str]:
        section_name = entity.get("section_name", "").strip()
        normalized_name = self.normalize_entity_name(section_name)
        if not normalized_name:
            return []

        name_tokens = [token for token in normalized_name.split() if token]
        search_terms = {normalized_name}
        if len(name_tokens) >= 2 and len(name_tokens[-1]) >= 4:
            search_terms.add(name_tokens[-1])

        matched_source_paths = sorted(
            {
                record.get("metadata", {}).get("source_path", "")
                for record in self.search_records
                if record.get("metadata", {}).get("source_path")
                and any(term in self.normalize_entity_name(record.get("document", "")) for term in search_terms)
            }
        )
        return [path for path in matched_source_paths if path]

    def find_role_lookup_titles(self) -> list[str]:
        preferred_titles = {"Staff", "Board", "Affiliates"}
        available_titles = {
            entity.get("title", "").strip()
            for entity in self.entity_registry
            if self.is_person_entity_type(entity.get("entity_type", "")) and entity.get("title", "").strip()
        }
        ranked_titles = [title for title in ("Staff", "Board", "Affiliates") if title in available_titles]
        if ranked_titles:
            return ranked_titles
        return sorted(available_titles)

    def apply_generic_plan_adjustments(self, query_plan: dict, rewritten_query: str) -> dict:
        adjusted_plan = dict(query_plan)
        matched_people = [
            entity
            for entity in self.find_matching_entities(rewritten_query)
            if self.is_person_entity_type(entity.get("entity_type", ""))
        ]

        if len(matched_people) == 1:
            related_source_paths = self.find_related_source_paths_for_entity(matched_people[0])
            adjusted_plan["use_entity_registry"] = False
            adjusted_plan["use_document_registry"] = False
            adjusted_plan["question_type"] = "people_lookup"
            adjusted_plan["prefer_summary"] = False
            adjusted_plan["top_k"] = min(int(adjusted_plan.get("top_k", self.config.top_k)), 5)
            if related_source_paths:
                adjusted_plan["routing_mode"] = "soft"
                adjusted_plan["target_source_paths"] = list(
                    dict.fromkeys((adjusted_plan.get("target_source_paths", []) or []) + related_source_paths)
                )

        if self.is_role_lookup_query(rewritten_query) and not matched_people:
            adjusted_plan["use_entity_registry"] = False
            adjusted_plan["use_document_registry"] = False
            adjusted_plan["question_type"] = "specific_fact"
            adjusted_plan["prefer_summary"] = False
            adjusted_plan["top_k"] = 3
            role_titles = self.find_role_lookup_titles()
            if role_titles:
                adjusted_plan["routing_mode"] = "soft"
                adjusted_plan["target_titles"] = list(
                    dict.fromkeys((adjusted_plan.get("target_titles", []) or []) + role_titles)
                )

        return adjusted_plan

    def answer_role_lookup_from_entity_registry(self, rewritten_query: str, query_plan: dict) -> Optional[dict]:
        if not self.is_role_lookup_query(rewritten_query):
            return None

        matched_people = [
            entity
            for entity in self.find_matching_entities(rewritten_query)
            if self.is_person_entity_type(entity.get("entity_type", ""))
        ]
        if matched_people:
            return None

        candidate_entities = [
            entity
            for entity in self.filter_entities_by_route(query_plan)
            if self.is_person_entity_type(entity.get("entity_type", ""))
        ]
        if not candidate_entities:
            candidate_entities = [
                entity for entity in self.entity_registry if self.is_person_entity_type(entity.get("entity_type", ""))
            ]

        query_tokens = self.normalize_lookup_tokens(rewritten_query)
        if not query_tokens:
            return None

        ranked_matches: list[tuple[float, dict, str]] = []
        for entity in candidate_entities:
            role = self.extract_entity_role(entity)
            if not role:
                continue

            role_tokens = self.normalize_lookup_tokens(role)
            if not role_tokens:
                continue

            overlap = len(set(query_tokens) & set(role_tokens))
            if overlap == 0:
                continue

            score = float(overlap)
            normalized_role = self.normalize_entity_name(role)
            normalized_query = self.normalize_entity_name(rewritten_query)
            if normalized_role and normalized_role in normalized_query:
                score += 3.0
            if normalized_query and normalized_query in normalized_role:
                score += 1.5
            if entity.get("title") == "Staff":
                score += 0.75

            ranked_matches.append((score, entity, role))

        if not ranked_matches:
            return None

        ranked_matches.sort(key=lambda item: item[0], reverse=True)
        top_score, top_entity, top_role = ranked_matches[0]
        second_score = ranked_matches[1][0] if len(ranked_matches) > 1 else float("-inf")
        if top_score < 2.0 or top_score == second_score:
            return None

        return {
            "answer": f"{top_entity['section_name']} is the {top_role}.",
            "sources": [
                {
                    "title": top_entity.get("title", "Untitled source"),
                    "url": top_entity.get("source_url", "URL not provided"),
                    "source_path": top_entity.get("source_path", "Unknown source"),
                }
            ],
            "needs_clarification": False,
            "clarification_options": [],
        }

    def find_recent_people_from_last_turn(self, recent_history: Optional[list[ConversationTurn]]) -> list[dict]:
        if not recent_history:
            return []
        last_turn = recent_history[-1]
        last_text = f"{last_turn.get('user', '')}\n{last_turn.get('assistant', '')}".strip()
        if not last_text:
            return []
        return [
            entity
            for entity in self.find_matching_entities(last_text)
            if self.is_person_entity_type(entity.get("entity_type", ""))
        ]

    def resolve_recent_entity_follow_up(self, user_message: str, recent_history: Optional[list[ConversationTurn]]) -> Optional[dict]:
        if not recent_history or not self.is_ambiguous_query(user_message):
            return None

        normalized_message = self.normalize_query_aliases(user_message)
        memories = self.build_recent_entity_memory(recent_history)
        if not memories:
            return None

        lowered_query = normalized_message.lower()
        project_follow_up_markers = ("project", "initiative", "program", "this", "that", "it")
        if any(marker in lowered_query for marker in project_follow_up_markers):
            recent_project_memories = [memory for memory in reversed(memories) if memory.get("project_entities")]
            if recent_project_memories:
                candidate_memory = recent_project_memories[0]
                unique_projects = list(
                    {
                        entity.get("section_name", "").strip(): entity
                        for entity in candidate_memory.get("project_entities", [])
                        if entity.get("section_name", "").strip()
                    }.values()
                )
                if len(unique_projects) == 1:
                    project = unique_projects[0]
                    rewritten_query = self.build_entity_follow_up_rewrite(normalized_message, project["section_name"], entity_type="project")
                    return {
                        "resolved": True,
                        "rewritten_query": rewritten_query,
                    }
                if len(unique_projects) > 1:
                    options = [entity["section_name"] for entity in unique_projects[:4]]
                    return {
                        "resolved": False,
                        "needs_clarification": True,
                        "clarifying_question": "Which project do you mean?",
                        "clarification_options": options,
                    }
            elif any(marker in lowered_query for marker in ("project", "initiative", "program")):
                recent_project_subject_memories = [
                    memory for memory in reversed(memories) if memory.get("project_subjects")
                ]
                if recent_project_subject_memories:
                    candidate_subjects = recent_project_subject_memories[0]["project_subjects"]
                    unique_subjects = list(
                        {
                            subject.get("label", "").strip().lower(): subject
                            for subject in candidate_subjects
                            if subject.get("label", "").strip()
                        }.values()
                    )
                    if len(unique_subjects) == 1:
                        subject = unique_subjects[0]
                        rewritten_query = self.build_entity_follow_up_rewrite(
                            normalized_message,
                            subject["label"],
                            entity_type="project",
                        )
                        return {"resolved": True, "rewritten_query": rewritten_query}
                    if len(unique_subjects) > 1:
                        options = [subject["label"] for subject in unique_subjects[:4]]
                        return {
                            "resolved": False,
                            "needs_clarification": True,
                            "clarifying_question": "Which project do you mean?",
                            "clarification_options": options,
                        }
                return None

        recent_person_memories = [memory for memory in reversed(memories) if memory["person_entities"]]
        if not recent_person_memories:
            return None

        candidate_memory = recent_person_memories[0]
        person_entities = candidate_memory["person_entities"]
        # Constrain ambiguous person follow-ups to people explicitly present in the last turn when possible.
        last_turn_people = self.find_recent_people_from_last_turn(recent_history)
        if last_turn_people:
            constrained_ids = {entity.get("unit_id", "").strip() for entity in last_turn_people if entity.get("unit_id", "").strip()}
            constrained_people = [
                entity
                for entity in person_entities
                if entity.get("unit_id", "").strip() in constrained_ids
            ]
            if constrained_people:
                person_entities = constrained_people
        unique_people = list(
            {
                entity.get("section_name", "").strip(): entity
                for entity in person_entities
                if entity.get("section_name", "").strip()
            }.values()
        )

        if len(unique_people) == 1:
            entity = unique_people[0]
            rewritten_query = self.build_entity_follow_up_rewrite(normalized_message, entity["section_name"])
            return {
                "resolved": True,
                "rewritten_query": rewritten_query,
            }

        selection_markers = (
            "which one",
            "which of them",
            "which of those",
            "who among",
            "who else",
            "which student",
            "which intern",
            "which person",
            "which board member",
            "which affiliate",
            "which staff",
        )
        singular_detail_markers = (
            "that person",
            "this person",
            "their research background",
            "their background",
            "their bio",
            "their role",
            "tell me more about one",
            "tell me more about them",
        )

        if any(marker in lowered_query for marker in selection_markers) and not any(
            marker in lowered_query for marker in singular_detail_markers
        ):
            target_titles = list(
                dict.fromkeys(
                    entity.get("title", "").strip()
                    for entity in unique_people
                    if entity.get("title", "").strip()
                )
            )
            target_categories = list(
                dict.fromkeys(
                    entity.get("category", "").strip()
                    for entity in unique_people
                    if entity.get("category", "").strip()
                )
            )
            target_folders = list(
                dict.fromkeys(
                    entity.get("folder_label", "").strip()
                    for entity in unique_people
                    if entity.get("folder_label", "").strip()
                )
            )
            target_source_paths = list(
                dict.fromkeys(
                    entity.get("source_path", "").strip()
                    for entity in unique_people
                    if entity.get("source_path", "").strip()
                )
            )
            return {
                "resolved": True,
                "rewritten_query": user_message,
                "query_route": {
                    "question_type": "people_lookup",
                    "routing_mode": "hard",
                    "prefer_summary": False,
                    "target_titles": target_titles,
                    "target_categories": target_categories,
                    "target_folders": target_folders,
                    "target_source_paths": target_source_paths,
                    "reason": "recent people selection follow-up",
                },
            }

        options = [entity["section_name"] for entity in unique_people[:4]]
        if not options:
            subject_candidates: list[dict] = []
            seen_subject_labels: set[str] = set()
            for memory in reversed(memories):
                for subject in memory.get("all_subjects", []):
                    label = subject.get("label", "").strip()
                    if not label:
                        continue
                    normalized_label = label.lower()
                    if normalized_label in seen_subject_labels:
                        continue
                    seen_subject_labels.add(normalized_label)
                    subject_candidates.append(subject)

            if len(subject_candidates) == 1:
                subject = subject_candidates[0]
                rewritten_query = self.build_entity_follow_up_rewrite(
                    normalized_message,
                    subject["label"],
                    entity_type=subject.get("subject_type", ""),
                )
                return {"resolved": True, "rewritten_query": rewritten_query}
            return None

        return {
            "resolved": False,
            "needs_clarification": True,
            "clarifying_question": "Which person do you mean?",
            "clarification_options": options,
        }

    def extract_entity_role(self, entity: dict) -> str:
        source_text = self.strip_embedding_labels(entity.get("detail_text", "") or entity.get("summary_text", ""))
        lines = [line.strip() for line in source_text.splitlines() if line.strip()]
        section_name = entity.get("section_name", "").strip()

        for line in lines:
            if self.names_refer_to_same_person(section_name, line) or line == section_name:
                continue
            lowered = line.lower()
            if lowered.startswith("title:"):
                return line.split(":", 1)[1].strip()
            if any(
                lowered.startswith(prefix)
                for prefix in ("phone:", "send email", "email:", "linkedin", "focus:", "bio:", "expertise:")
            ):
                continue
            if "mailto:" in lowered:
                continue
            if len(line) <= 140 and not line.endswith("."):
                return line

        return ""


    # ------------------------------------------------------------------ #
    #  LLM-based query planner (replaces all regex classification logic)  #
    # ------------------------------------------------------------------ #

    def plan_query_with_llm(
        self,
        user_message: str,
        recent_history: Optional[list[ConversationTurn]] = None,
    ) -> dict:
        """Single LLM call that handles intent classification, entity resolution,
        follow-up rewriting, and source routing — replacing all regex heuristics."""
        if not self.search_records:
            route = self.default_query_route(user_message)
            route.update({
                "rewritten_query": user_message,
                "use_entity_registry": False,
                "use_document_registry": False,
                "entity_filter_type": "",
                "top_k": self.config.top_k,
                "needs_clarification": False,
                "clarifying_question": "",
                "clarification_options": [],
            })
            return route

        normalized_user_message = self.normalize_query_aliases(user_message)
        route_catalog = self.get_route_catalog()
        history_text = format_recent_history(recent_history or []) or "No recent conversation."
        entity_memory_text = self.format_recent_entity_memory(recent_history)

        planning_prompt = f"""
You are the query planner for a RAG assistant about the UMass Boston Sustainable Solutions Lab (SSL).

Your job in ONE pass:
1. Understand what the user is really asking, including resolving pronouns and follow-up references from the conversation history.
2. Rewrite the question into a clear standalone retrieval query (resolve "he", "she", "they", "it", "that person", "this one", etc. to specific named entities from the history when possible).
3. Classify the question type and choose the best retrieval strategy.
4. Decide whether to answer from a registry (structured list) or from document retrieval.
5. Only ask for clarification if there is genuine ambiguity you cannot resolve from context.

Return ONLY valid JSON matching this exact schema — no preamble, no markdown fences:
{{
  "rewritten_query": "standalone question with all references resolved",
  "question_type": "one of: specific_fact | broad_overview | list_inventory | people_lookup | publication_inventory | contact | comparison | follow_up",
  "routing_mode": "one of: hard | soft | global",
  "prefer_summary": false,
  "target_titles": [],
  "target_categories": [],
  "target_folders": [],
  "target_source_paths": [],
  "use_entity_registry": false,
  "use_document_registry": false,
  "entity_filter_type": "",
  "top_k": 5,
  "needs_clarification": false,
  "clarifying_question": "",
  "clarification_options": [],
  "reason": "one line explanation"
}}

Field rules:
- rewritten_query: Always fill this in. If no rewrite needed, copy the original question.
- question_type: Use "people_lookup" when asking about a specific named person. Use "list_inventory" when listing people or projects. Use "publication_inventory" when listing documents/reports. Use "broad_overview" for general SSL questions. Use "specific_fact" for targeted factual queries like role/title lookup without a named person.
- routing_mode: "hard" = only search the specified targets; "soft" = prefer targets but fall back to all; "global" = search everything. Use "global" for open-ended or multi-document questions unless the user clearly narrowed scope. Use "soft" or "hard" when the user names a document, person, or project tied to specific source_paths/titles.
- Named entities in the question: if the user names a person or project, set target_source_paths or target_titles from the catalog to improve retrieval when possible.
- Questions about a named person should use document retrieval, not the entity registry, because exact-name retrieval usually lands on the right source first.
- Role/title questions without a named person, such as "who is the director", should be routed toward current profile sources with a soft route and should not use the entity registry listing path.
- use_entity_registry: Set true ONLY for enumeration/listing questions (who are the staff, list the projects, how many board members). Do NOT set true for questions about a specific named person.
- use_document_registry: Set true ONLY for listing/counting publication or document titles.
- entity_filter_type: When use_entity_registry is true, set to one of: staff_member | board_member | affiliate | person | project | "" (for all types).
- top_k: Set 8 for broad/overview questions, 5 for specific facts, 3 for very targeted lookups.
- For current role/title questions, prefer current profile sources such as Staff when available.
- For any question about a named person, use document retrieval rather than a registry answer, and do not narrow to only one current-role source unless the user specifically asks only about their current title/role.
- If the user asks to filter a previously discussed group by a condition, such as "which of them work on X", do NOT use a registry enumeration answer. Use retrieval over the previously discussed group scope.
- target_titles/categories/folders/source_paths: Only use values from the available lists below.

Available titles: {json.dumps(route_catalog["titles"])}
Available categories: {json.dumps(route_catalog["categories"])}
Available folders: {json.dumps(route_catalog["folders"])}
Available source_paths: {json.dumps(route_catalog["source_paths"])}
Available entity names (for resolving pronouns): {json.dumps(route_catalog["entity_names"])}

Recent conversation:
{history_text}

Recent structured entity memory:
{entity_memory_text}

User question:
{normalized_user_message}
""".strip()

        try:
            raw_plan = self.llm_callable(planning_prompt).strip()
            parsed_plan = self.parse_json_object(raw_plan)
        except Exception:
            route = self.default_query_route(normalized_user_message)
            route.update({
                "rewritten_query": normalized_user_message,
                "use_entity_registry": False,
                "use_document_registry": False,
                "entity_filter_type": "",
                "top_k": self.config.top_k,
                "needs_clarification": False,
                "clarifying_question": "",
                "clarification_options": [],
            })
            return route

        # Normalise the route fields through existing validation logic
        normalized = self.normalize_query_plan(parsed_plan, route_catalog, normalized_user_message)

        # Carry through the new fields that normalize_query_plan doesn't know about
        normalized["use_entity_registry"] = bool(parsed_plan.get("use_entity_registry", False))
        normalized["use_document_registry"] = bool(parsed_plan.get("use_document_registry", False))
        normalized["entity_filter_type"] = str(parsed_plan.get("entity_filter_type", "")).strip()
        normalized["top_k"] = int(parsed_plan.get("top_k", self.config.top_k))

        return normalized

    # ------------------------------------------------------------------ #
    #  Registry answer helpers (simplified — no regex routing inside)     #
    # ------------------------------------------------------------------ #

    def answer_from_entity_registry(self, rewritten_query: str, query_plan: dict) -> dict:
        """Answer listing/enumeration questions from the in-memory entity registry."""
        entity_filter_type = query_plan.get("entity_filter_type", "")
        entities = self.filter_entities_by_route(query_plan)

        if entity_filter_type:
            entities = [e for e in entities if e.get("entity_type") == entity_filter_type]

        if not entities:
            return {
                "answer": "I do not have enough information in the corpus to answer that.",
                "sources": [],
                "needs_clarification": False,
                "clarification_options": [],
            }

        # Build a human-readable label for the preamble
        label_map = {
            "staff_member": "staff members",
            "board_member": "board members",
            "affiliate": "affiliates",
            "person": "students and interns",
            "project": "projects or initiatives",
        }
        label = label_map.get(entity_filter_type, "people or entities")
        lowered_query = rewritten_query.lower()
        if not entity_filter_type and any(
            token in lowered_query for token in ("including staff", "including affiliates", "including students", "board")
        ):
            grouped: dict[str, list[dict]] = {
                "Staff": [],
                "Affiliates": [],
                "Students and Interns": [],
                "Board Members": [],
            }
            for entity in entities:
                entity_type = entity.get("entity_type", "")
                if entity_type == "staff_member":
                    grouped["Staff"].append(entity)
                elif entity_type in {"affiliate", "visiting_scholar"}:
                    grouped["Affiliates"].append(entity)
                elif entity_type == "person":
                    grouped["Students and Interns"].append(entity)
                elif entity_type == "board_member":
                    grouped["Board Members"].append(entity)

            lines = ["Here is an overview of people involved with SSL by group:"]
            sources: list[dict] = []
            seen_sources: set[tuple[str, str]] = set()
            for group_name, group_entities in grouped.items():
                if not group_entities:
                    continue
                lines.append("")
                lines.append(f"{group_name} ({len(group_entities)}):")
                for entity in group_entities[:8]:
                    role = self.extract_entity_role(entity)
                    if role:
                        lines.append(f"- {entity['section_name']} — {role}")
                    else:
                        lines.append(f"- {entity['section_name']}")
                    key = (entity.get("title", ""), entity.get("source_path", ""))
                    if key not in seen_sources:
                        seen_sources.add(key)
                        sources.append({
                            "title": entity.get("title", "Untitled source"),
                            "url": entity.get("source_url", "URL not provided"),
                            "source_path": entity.get("source_path", "Unknown source"),
                        })

            if len(lines) > 1:
                return {
                    "answer": "\n".join(lines).strip(),
                    "sources": sources,
                    "needs_clarification": False,
                    "clarification_options": [],
                }
        max_listed = min(len(entities), 20)
        listed = entities[:max_listed]

        include_roles = entity_filter_type in {"board_member", "staff_member", "affiliate"}
        lines = [f"I found {len(entities)} {label} in the corpus."]
        lines.append("")
        for idx, entity in enumerate(listed, start=1):
            role = self.extract_entity_role(entity) if include_roles else ""
            if role:
                lines.append(f"{idx}. {entity['section_name']} — {role} [{idx}]")
            else:
                lines.append(f"{idx}. {entity['section_name']} [{idx}]")

        if len(entities) > max_listed:
            lines.append(f"\nShowing the first {max_listed}; there are {len(entities) - max_listed} more.")

        seen: set[tuple] = set()
        sources: list[dict] = []
        for entity in listed:
            key = (entity.get("title", ""), entity.get("source_path", ""))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "title": entity.get("title", "Untitled source"),
                    "url": entity.get("source_url", "URL not provided"),
                    "source_path": entity.get("source_path", "Unknown source"),
                })

        return {
            "answer": "\n".join(lines).strip(),
            "sources": sources,
            "needs_clarification": False,
            "clarification_options": [],
        }

    def answer_from_document_registry(self, rewritten_query: str, query_plan: dict) -> dict:
        """Answer document listing/counting questions from the in-memory document registry."""
        documents = self.filter_documents_by_route(query_plan)

        if not documents:
            return {
                "answer": "I do not have enough information in the corpus to answer that.",
                "sources": [],
                "needs_clarification": False,
                "clarification_options": [],
            }

        lowered = rewritten_query.lower()
        if any(t in lowered for t in ("publication", "publications")):
            label = "publication source documents"
        elif any(t in lowered for t in ("report", "reports")):
            label = "report source documents"
        else:
            label = "source documents"

        max_listed = min(len(documents), 20)
        listed = documents[:max_listed]
        lines = [f"I found {len(documents)} {label} in the corpus.", ""]
        for idx, doc in enumerate(listed, start=1):
            lines.append(f"{idx}. {doc['title']} [{idx}]")

        if len(documents) > max_listed:
            lines.append(f"\nShowing the first {max_listed}.")

        sources = [
            {
                "title": doc["title"],
                "url": doc.get("source_url", "URL not provided"),
                "source_path": doc["source_path"],
            }
            for doc in listed
        ]

        return {
            "answer": "\n".join(lines).strip(),
            "sources": sources,
            "needs_clarification": False,
            "clarification_options": [],
        }

    # ------------------------------------------------------------------ #
    #  Prompt builder                                                      #
    # ------------------------------------------------------------------ #

    def build_prompt(
        self,
        user_message: str,
        retrieved_context: list[str],
        recent_history: Optional[list[ConversationTurn]] = None,
        rewritten_query: Optional[str] = None,
    ) -> str:
        n_sources = len(retrieved_context)
        if retrieved_context:
            numbered_blocks = [f"[{i}]\n{block}" for i, block in enumerate(retrieved_context, start=1)]
            retrieved_text = "\n\n".join(numbered_blocks)
        else:
            retrieved_text = "No relevant context found."

        history_text = format_recent_history(recent_history or [])
        history_section = f"\nRecent conversation:\n{history_text}\n" if history_text else ""

        rewrite_section = ""
        if rewritten_query and rewritten_query.strip().lower() != user_message.strip().lower():
            rewrite_section = f"\nResolved retrieval query:\n{rewritten_query}\n"

        return f"""
You are the Sustainable Solutions Lab assistant. Answer only from the retrieved context below.
If the answer is not supported by the context, say you do not have enough information.
There are exactly {n_sources} source blocks, numbered [1] through [{n_sources}].
Never cite a number outside this range. Never invent citation numbers.
Do not include inline bracket citations in prose; rely on the returned source list for attribution.
Do not mention a person name unless that exact name appears in retrieved context.
For current roles or titles, prefer the most recent source (Staff page over old annual reports).
{history_section}{rewrite_section}
Retrieved context:
{retrieved_text}

Question:
{user_message}
""".strip()

    # ------------------------------------------------------------------ #
    #  Source extractor                                                    #
    # ------------------------------------------------------------------ #

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
            sources.append({
                "title": title,
                "url": source_url or "URL not provided",
                "source_path": source_path,
            })
        return sources

    def maybe_attach_route_debug(self, response: dict, query_route: Optional[dict], rewritten_query: str) -> dict:
        if not self.config.route_debug_enabled:
            return response
        route = query_route or self.default_query_route(rewritten_query)
        targets_used = {
            "target_titles": route.get("target_titles", []) or [],
            "target_categories": route.get("target_categories", []) or [],
            "target_folders": route.get("target_folders", []) or [],
            "target_source_paths": route.get("target_source_paths", []) or [],
        }
        response["route_debug"] = {
            "question_type": route.get("question_type", "specific_fact"),
            "routing_mode": route.get("routing_mode", "global"),
            "targets_used": targets_used,
            "rewritten_query": rewritten_query,
        }
        return response

    def answer_mentions_unsupported_name(self, answer: str, retrieved_context: list[str], user_message: str) -> bool:
        answer_names = self.extract_query_named_phrases(answer)
        if not answer_names:
            return False
        normalized_context = self.normalize_entity_name("\n".join(retrieved_context))
        normalized_user = self.normalize_entity_name(user_message)
        for name in answer_names:
            normalized_name = self.normalize_entity_name(name)
            if not normalized_name:
                continue
            if normalized_name in normalized_user:
                continue
            if normalized_name not in normalized_context:
                return True
        return False

    def should_retry_with_fallback_route(self, answer: str, rewritten_query: str, query_plan: dict, retrieved_context: list[str]) -> bool:
        lowered_answer = answer.lower()
        if "i do not have enough information" in lowered_answer:
            return True
        if not retrieved_context:
            return True
        if self.answer_mentions_unsupported_name(answer, retrieved_context, rewritten_query):
            return True
        return False

    def build_fallback_route(self, rewritten_query: str, query_plan: dict) -> dict:
        fallback = dict(query_plan)
        fallback["use_entity_registry"] = False
        fallback["use_document_registry"] = False
        fallback["routing_mode"] = "soft"
        fallback["top_k"] = max(int(query_plan.get("top_k", self.config.top_k)), 8)

        lowered_query = rewritten_query.lower()
        board_titles = [
            title for title in {entity.get("title", "").strip() for entity in self.entity_registry}
            if title and "board" in title.lower()
        ]
        if self.role_group_in_query(lowered_query, "board") and board_titles:
            fallback["target_titles"] = list(dict.fromkeys((fallback.get("target_titles", []) or []) + board_titles))
        if self.role_group_in_query(lowered_query, "affiliate"):
            affiliate_titles = [
                title for title in {entity.get("title", "").strip() for entity in self.entity_registry}
                if title and "affiliate" in title.lower()
            ]
            fallback["target_titles"] = list(dict.fromkeys((fallback.get("target_titles", []) or []) + affiliate_titles))
        return fallback

    # ------------------------------------------------------------------ #
    #  Main answer entry point                                            #
    # ------------------------------------------------------------------ #

    def answer(self, user_message: str, recent_history: Optional[list[ConversationTurn]] = None) -> dict:
        recent_history = recent_history or []
        normalized_user_message = self.normalize_query_aliases(user_message)
        structured_follow_up = self.resolve_recent_entity_follow_up(normalized_user_message, recent_history)

        if structured_follow_up and structured_follow_up.get("needs_clarification"):
            return self.maybe_attach_route_debug({
                "answer": structured_follow_up.get("clarifying_question", "Can you clarify what you mean?"),
                "sources": [],
                "needs_clarification": True,
                "clarification_for": user_message,
                "clarification_options": structured_follow_up.get("clarification_options", []),
            }, structured_follow_up.get("query_route"), user_message)

        resolved_query = structured_follow_up.get("rewritten_query", normalized_user_message) if structured_follow_up else normalized_user_message

        # Single LLM call replaces all regex classification
        query_plan = self.plan_query_with_llm(resolved_query, recent_history)

        if structured_follow_up and structured_follow_up.get("query_route"):
            query_plan = self.merge_query_routes(
                query_plan,
                structured_follow_up["query_route"],
                replace_targets=True,
            )

        query_plan["rewritten_query"] = resolved_query
        query_plan = self.apply_generic_plan_adjustments(query_plan, resolved_query)

        if self.is_group_selection_follow_up(user_message) or self.is_specific_entity_detail_query(resolved_query):
            query_plan["use_entity_registry"] = False
            query_plan["use_document_registry"] = False

        # Handle clarification requests from the planner
        if query_plan.get("needs_clarification"):
            return self.maybe_attach_route_debug({
                "answer": query_plan.get("clarifying_question", "Can you clarify what you mean?"),
                "sources": [],
                "needs_clarification": True,
                "clarification_for": user_message,
                "clarification_options": query_plan.get("clarification_options", []),
            }, query_plan, resolved_query)

        rewritten_query = query_plan.get("rewritten_query", resolved_query)
        top_k = query_plan.get("top_k", self.config.top_k)

        # Entity registry path — listing people/projects
        if query_plan.get("use_entity_registry"):
            return self.maybe_attach_route_debug(
                self.answer_from_entity_registry(rewritten_query, query_plan),
                query_plan,
                rewritten_query,
            )

        # Document registry path — listing publication titles
        if query_plan.get("use_document_registry"):
            return self.maybe_attach_route_debug(
                self.answer_from_document_registry(rewritten_query, query_plan),
                query_plan,
                rewritten_query,
            )

        role_lookup_result = self.answer_role_lookup_from_entity_registry(rewritten_query, query_plan)
        if role_lookup_result:
            return self.maybe_attach_route_debug(role_lookup_result, query_plan, rewritten_query)

        # Standard retrieval path
        retrieved_context, retrieved_metadata, _ = self.retrieve_context(
            rewritten_query,
            top_k=top_k,
            query_route=query_plan,
        )

        prompt = self.build_prompt(
            user_message=user_message,
            retrieved_context=retrieved_context,
            recent_history=recent_history,
            rewritten_query=rewritten_query,
        )

        answer_text = self.llm_callable(prompt).strip()
        if self.should_retry_with_fallback_route(answer_text, rewritten_query, query_plan, retrieved_context):
            fallback_route = self.build_fallback_route(rewritten_query, query_plan)
            fallback_context, fallback_metadata, _ = self.retrieve_context(
                rewritten_query,
                top_k=fallback_route.get("top_k", top_k),
                query_route=fallback_route,
            )
            if fallback_context:
                fallback_prompt = self.build_prompt(
                    user_message=user_message,
                    retrieved_context=fallback_context,
                    recent_history=recent_history,
                    rewritten_query=rewritten_query,
                )
                fallback_answer = self.llm_callable(fallback_prompt).strip()
                if fallback_answer and not self.answer_mentions_unsupported_name(fallback_answer, fallback_context, rewritten_query):
                    answer_text = fallback_answer
                    retrieved_metadata = fallback_metadata
                    query_plan = fallback_route

        return self.maybe_attach_route_debug({
            "answer": answer_text,
            "sources": self.extract_sources(retrieved_metadata),
            "needs_clarification": False,
            "clarification_options": [],
        }, query_plan, rewritten_query)


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
    app = Flask(__name__, static_folder="staticfuture")
    app.config["JSON_SORT_KEYS"] = False

    @app.get("/")
    def index():
        return render_template("indexfuture.html")

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
