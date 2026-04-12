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

    def detect_local_query_route(self, query: str) -> dict:
        lowered_query = query.lower()
        route = self.default_query_route(query)
        matched_reasons: list[str] = []
        target_titles: set[str] = set()
        target_categories: set[str] = set()
        target_folders: set[str] = set()
        target_source_paths: set[str] = set()

        def apply_scope(
            *,
            titles: Optional[list[str]] = None,
            categories: Optional[list[str]] = None,
            folders: Optional[list[str]] = None,
            source_paths: Optional[list[str]] = None,
            question_type: Optional[str] = None,
            prefer_summary: Optional[bool] = None,
            reason: str,
        ) -> None:
            if titles:
                target_titles.update(titles)
            if categories:
                target_categories.update(categories)
            if folders:
                target_folders.update(folders)
            if source_paths:
                target_source_paths.update(source_paths)
            if question_type and route["question_type"] == "specific_fact":
                route["question_type"] = question_type
            if prefer_summary is True:
                route["prefer_summary"] = True
            matched_reasons.append(reason)

        if any(term in lowered_query for term in ("project", "projects", "initiative", "initiatives", "program", "programs")):
            apply_scope(
                titles=["Projects"],
                folders=["Annual Reports"],
                question_type="broad_overview" if any(term in lowered_query for term in ("what are", "overview", "current")) else "specific_fact",
                prefer_summary=True,
                reason="project-related sources",
            )

        if any(term in lowered_query for term in ("staff", "team", "employee", "employees")):
            apply_scope(
                titles=["Staff", "SSLAbout"],
                question_type="people_lookup",
                prefer_summary=True,
                reason="staff-related sources",
            )

        if any(term in lowered_query for term in ("student", "students", "intern", "interns", "fellow", "fellows", "alumni")):
            apply_scope(
                titles=["StudentsInterns", "AnnualReport2021"],
                question_type="people_lookup",
                prefer_summary=False,
                reason="student and intern sources",
            )

        if any(term in lowered_query for term in ("board", "leadership", "leader", "leaders", "advisory")):
            apply_scope(
                titles=["BoardOfDirectors", "SSLAbout", "AnnualReport2021"],
                question_type="people_lookup",
                prefer_summary=True,
                reason="board and leadership sources",
            )

        if any(term in lowered_query for term in ("affiliate", "affiliates", "faculty affiliate", "university affiliate")):
            apply_scope(
                titles=["UniversityAffiliates", "AnnualReport2021"],
                question_type="people_lookup",
                prefer_summary=True,
                reason="affiliate sources",
            )

        if any(term in lowered_query for term in ("publication", "publications", "paper", "papers", "report", "reports")):
            apply_scope(
                categories=["Publications", "Annual Reports"],
                folders=["Publications", "Annual Reports"],
                question_type="publication_inventory" if any(term in lowered_query for term in ("list", "name all", "how many")) else "specific_fact",
                prefer_summary=True,
                reason="publication and report sources",
            )

        if any(term in lowered_query for term in ("contact", "email", "phone", "address", "location", "located", "where is")):
            apply_scope(
                titles=["SSLAbout", "Staff"],
                question_type="contact",
                prefer_summary=False,
                reason="contact and about sources",
            )

        if any(term in lowered_query for term in ("what we do", "categories of work", "main categories of work")):
            apply_scope(
                titles=["SSLAbout"],
                question_type="specific_fact",
                prefer_summary=True,
                reason="SSL about section sources",
            )

        if any(term in lowered_query for term in ("mission", "vision", "year in review", "what does ssl do")):
            apply_scope(
                titles=["SSLAbout", "AnnualReport2021"],
                folders=["Annual Reports"],
                question_type="specific_fact",
                prefer_summary=True,
                reason="mission and about section sources",
            )

        if any(term in lowered_query for term in ("research background", "bio", "biography", "background")):
            apply_scope(
                titles=["Staff", "StudentsInterns", "UniversityAffiliates", "BoardOfDirectors"],
                question_type="people_lookup",
                prefer_summary=False,
                reason="person biography sources",
            )

        if any(term in lowered_query for term in ("cape cod", "rail", "railway", "massdot", "train line", "rail resilience")):
            apply_scope(
                titles=["Projects", "StudentsInterns", "AnnualReport2021"],
                question_type="people_lookup" if any(term in lowered_query for term in ("student", "students", "intern", "interns", "person", "people")) or self.is_group_selection_follow_up(query) else "specific_fact",
                prefer_summary=False,
                reason="cape cod rail overlap sources",
            )

        if "northeast climate justice research collaborative" in lowered_query:
            apply_scope(
                titles=["Projects", "SSLAbout"],
                question_type="specific_fact",
                prefer_summary=False,
                reason="collaborative-specific sources",
            )

        if "climate adaptation forum" in lowered_query:
            apply_scope(
                titles=["Projects", "AnnualReport2021"],
                question_type="specific_fact",
                prefer_summary=False,
                reason="forum-specific sources",
            )

        if any(term in lowered_query for term in ("climate careers curricula initiative", "c3i", "c3 initiative")):
            apply_scope(
                titles=["Projects"],
                question_type="specific_fact",
                prefer_summary=False,
                reason="c3 initiative sources",
            )

        if "cliir" in lowered_query or "climate inequality and integrative resilience" in lowered_query:
            apply_scope(
                titles=["Projects", "StudentsInterns"],
                question_type="specific_fact",
                prefer_summary=False,
                reason="cliir-related sources",
            )

        if "vishal verma" in lowered_query:
            apply_scope(
                source_paths=["SEED_DOCUMENTS/StudentsInterns.txt"],
                question_type="people_lookup",
                prefer_summary=False,
                reason="Vishal Verma entity source",
            )

        if "carlos velásquez" in lowered_query or "carlos velasquez" in lowered_query:
            apply_scope(
                source_paths=["SEED_DOCUMENTS/Projects.txt"],
                question_type="people_lookup",
                prefer_summary=False,
                reason="Carlos Velásquez entity source",
            )

        if "rebecca herst" in lowered_query:
            apply_scope(
                source_paths=["SEED_DOCUMENTS/Annual Reports/AnnualReport2021.txt"],
                question_type="people_lookup",
                prefer_summary=False,
                reason="Rebecca Herst entity source",
            )

        if "b. r. balachandran" in lowered_query or "balachandran" in lowered_query:
            apply_scope(
                titles=["Staff"],
                question_type="people_lookup",
                prefer_summary=False,
                reason="Balachandran staff source",
            )

        if "rosalyn negron" in lowered_query:
            apply_scope(
                titles=["Staff", "StudentsInterns", "AnnualReport2021"],
                question_type="people_lookup",
                prefer_summary=False,
                reason="Rosalyn Negron overlap sources",
            )

        for entity in self.entity_registry:
            section_name = entity.get("section_name", "").strip()
            if not section_name:
                continue
            normalized_name = section_name.lower()
            if len(normalized_name) < 5 or normalized_name not in lowered_query:
                continue

            entity_type = entity.get("entity_type", "")
            apply_scope(
                titles=[entity.get("title", "")] if entity.get("title") else None,
                categories=[entity.get("category", "")] if entity.get("category") else None,
                folders=[entity.get("folder_label", "")] if entity.get("folder_label") else None,
                source_paths=[entity.get("source_path", "")] if entity.get("source_path") else None,
                question_type="people_lookup" if entity_type != "project" else "specific_fact",
                prefer_summary=False,
                reason=f"entity match for {section_name}",
            )

        candidate_phrases = self.extract_query_named_phrases(query)
        for phrase in candidate_phrases:
            matched_source_paths = sorted(
                {
                    record.get("metadata", {}).get("source_path", "")
                    for record in self.search_records
                    if phrase.lower() in record.get("document", "").lower() and record.get("metadata", {}).get("source_path")
                }
            )
            if matched_source_paths:
                apply_scope(
                    source_paths=matched_source_paths,
                    question_type="people_lookup" if any(term in lowered_query for term in ("who is", "what does", "say about")) else "specific_fact",
                    prefer_summary=False,
                    reason=f"exact phrase match for {phrase}",
                )

        if target_titles or target_categories or target_folders or target_source_paths:
            route.update(
                {
                    "routing_mode": "soft",
                    "target_titles": sorted(target_titles),
                    "target_categories": sorted(target_categories),
                    "target_folders": sorted(target_folders),
                    "target_source_paths": sorted(target_source_paths),
                    "reason": "Local first-pass multi-label route: " + ", ".join(dict.fromkeys(matched_reasons)),
                }
            )

        return route

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

    def plan_query_with_llm(
        self,
        user_message: str,
        recent_history: Optional[list[ConversationTurn]] = None,
    ) -> dict:
        retrieval_query = user_message
        if not self.search_records:
            default_route = self.default_query_route(retrieval_query)
            default_route.update(
                {
                    "rewritten_query": retrieval_query,
                    "needs_clarification": False,
                    "clarifying_question": "",
                    "clarification_options": [],
                }
            )
            return default_route

        route_catalog = self.get_route_catalog()
        history_text = format_recent_history(recent_history or []) or "No recent conversation."
        entity_memory_text = self.format_recent_entity_memory(recent_history)
        planning_prompt = f"""
You are planning retrieval for a Sustainable Solutions Lab RAG system.
Your job is to do all of the following in one pass:
1. Resolve ambiguous follow-up references when possible.
2. Rewrite the user question into a standalone retrieval query when needed.
3. Decide whether clarification is still required.
4. Choose the best retrieval scope from the available corpus metadata.

Return valid JSON only with this schema:
{{
  "rewritten_query": "",
  "question_type": "specific_fact",
  "routing_mode": "hard",
  "prefer_summary": false,
  "target_titles": [],
  "target_categories": [],
  "target_folders": [],
  "target_source_paths": [],
  "needs_clarification": false,
  "clarifying_question": "",
  "clarification_options": [],
  "reason": "short explanation"
}}

Allowed question_type values:
- broad_overview
- specific_fact
- list_inventory
- people_lookup
- follow_up
- contact
- publication_inventory
- comparison
- unknown

Allowed routing_mode values:
- hard
- soft
- global

Important rules:
- First try to resolve the question silently from recent conversation.
- If one referent is clearly most likely, rewrite the query and do not ask a clarification question.
- If multiple plausible referents remain, set needs_clarification to true and provide one short clarifying question plus 2-4 short user-facing options.
- Only choose targets that exist in the available metadata lists.
- For people follow-ups, prefer the most likely person/source area from recent conversation.
- For publications inventory questions, prefer Publications or Annual Reports scopes rather than global retrieval.
- If the question is broad, set prefer_summary to true.
- If the question is about a specific person, project, or document, set prefer_summary to false.
- When needs_clarification is true, still provide your best partial routing if you can.

Recent conversation:
{history_text}

Recent structured entity memory:
{entity_memory_text}

Original user question:
{user_message}

Available titles:
{json.dumps(route_catalog["titles"], indent=2)}

Available categories:
{json.dumps(route_catalog["categories"], indent=2)}

Available folders:
{json.dumps(route_catalog["folders"], indent=2)}

Available source paths:
{json.dumps(route_catalog["source_paths"], indent=2)}

Available entity names:
{json.dumps(route_catalog["entity_names"], indent=2)}
""".strip()

        try:
            raw_plan = self.llm_callable(planning_prompt).strip()
            parsed_plan = self.parse_json_object(raw_plan)
        except Exception:
            default_route = self.default_query_route(retrieval_query)
            default_route.update(
                {
                    "rewritten_query": retrieval_query,
                    "needs_clarification": False,
                    "clarifying_question": "",
                    "clarification_options": [],
                }
            )
            return default_route

        return self.normalize_query_plan(parsed_plan, route_catalog, retrieval_query)

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

    def find_matching_entities(self, user_message: str, entities: Optional[list[dict]] = None) -> list[dict]:
        lowered_query = user_message.lower()
        matched_entities: list[dict] = []
        for entity in entities or self.entity_registry:
            section_name = entity.get("section_name", "").strip()
            if not section_name:
                continue

            full_name = section_name.lower()
            normalized_name = self.normalize_entity_name(section_name)
            if full_name in lowered_query or (normalized_name and normalized_name in self.normalize_entity_name(user_message)):
                matched_entities.append(entity)

        return matched_entities

    def is_person_entity_type(self, entity_type: str) -> bool:
        return entity_type in {"person", "staff_member", "board_member", "affiliate", "visiting_scholar"}

    def build_recent_entity_memory(self, recent_history: Optional[list[ConversationTurn]]) -> list[dict]:
        if not recent_history or not self.entity_registry:
            return []

        memories: list[dict] = []
        for turn_index, turn in enumerate(recent_history, start=1):
            turn_entities_by_id: dict[str, dict] = {}
            for speaker in ("user", "assistant"):
                text = (turn.get(speaker) or "").strip()
                if not text:
                    continue
                for entity in self.find_matching_entities(text):
                    unit_id = entity.get("unit_id", "").strip()
                    if not unit_id:
                        continue
                    turn_entities_by_id.setdefault(unit_id, entity)

            if not turn_entities_by_id:
                continue

            all_entities = list(turn_entities_by_id.values())
            person_entities = [
                entity
                for entity in all_entities
                if self.is_person_entity_type(entity.get("entity_type", ""))
            ]
            memories.append(
                {
                    "turn_index": turn_index,
                    "all_entities": all_entities,
                    "person_entities": person_entities,
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
            entities = ", ".join(entity.get("section_name", "") for entity in memory["all_entities"] if entity.get("section_name"))
            if people:
                lines.append(f"Turn {memory['turn_index']} people: {people}")
            elif entities:
                lines.append(f"Turn {memory['turn_index']} entities: {entities}")

        return "\n".join(lines) if lines else "No recent entity memory."

    def build_entity_follow_up_rewrite(self, user_message: str, entity_name: str) -> str:
        stripped_message = user_message.strip()
        lowered_message = stripped_message.lower()
        if not stripped_message:
            return entity_name

        rewritten = stripped_message
        substitution_patterns = [
            (r"\bthat person\b", entity_name),
            (r"\bthis person\b", entity_name),
            (r"\bthat one\b", entity_name),
            (r"\bthis one\b", entity_name),
            (r"\bthose people\b", entity_name),
            (r"\bthese people\b", entity_name),
            (r"\btheir\b", f"{entity_name}'s"),
            (r"\bthem\b", entity_name),
            (r"\bthey\b", entity_name),
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

        return f"{entity_name}: {stripped_message}"

    def resolve_recent_entity_follow_up(self, user_message: str, recent_history: Optional[list[ConversationTurn]]) -> Optional[dict]:
        if not recent_history or not self.is_ambiguous_query(user_message):
            return None

        memories = self.build_recent_entity_memory(recent_history)
        if not memories:
            return None

        recent_person_memories = [memory for memory in reversed(memories) if memory["person_entities"]]
        if not recent_person_memories:
            return None

        candidate_memory = recent_person_memories[0]
        person_entities = candidate_memory["person_entities"]
        unique_people = list(
            {
                entity.get("section_name", "").strip(): entity
                for entity in person_entities
                if entity.get("section_name", "").strip()
            }.values()
        )

        if len(unique_people) == 1:
            entity = unique_people[0]
            rewritten_query = self.build_entity_follow_up_rewrite(user_message, entity["section_name"])
            query_route = self.detect_local_query_route(rewritten_query)
            return {
                "resolved": True,
                "rewritten_query": rewritten_query,
                "query_route": query_route,
            }

        lowered_query = user_message.lower()
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
                    "routing_mode": "soft",
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
            return None

        return {
            "resolved": False,
            "needs_clarification": True,
            "clarifying_question": "Which person do you mean?",
            "clarification_options": options,
        }

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

    def entity_matches_query_focus(self, entity: dict, user_message: str) -> bool:
        lowered_query = user_message.lower()
        source_text = self.strip_embedding_labels(entity.get("detail_text", "") or entity.get("summary_text", "")).lower()

        focus_groups = [
            {
                "query_terms": ("rail", "railway", "massdot", "cape cod", "train line", "rail resilience", "railway resilience"),
                "source_terms": (
                    "rail",
                    "railway",
                    "massdot",
                    "cape cod",
                    "train line",
                    "rail resilience",
                    "rail safety",
                    "climate resilience on cape cod",
                    "safety and resilience in coastal massachusetts",
                    "cape main line",
                ),
            },
            {
                "query_terms": ("cliir", "climate inequality", "integrative resilience"),
                "source_terms": ("cliir", "climate inequality", "integrative resilience"),
            },
            {
                "query_terms": ("collaborative", "northeast climate justice research collaborative"),
                "source_terms": ("collaborative", "northeast climate justice research collaborative"),
            },
            {
                "query_terms": ("forum", "climate adaptation forum"),
                "source_terms": ("forum", "climate adaptation forum"),
            },
            {
                "query_terms": ("c3i", "climate careers curricula initiative"),
                "source_terms": ("c3i", "climate careers curricula initiative"),
            },
        ]
        for focus_group in focus_groups:
            if any(term in lowered_query for term in focus_group["query_terms"]):
                return any(term in source_text for term in focus_group["source_terms"])

        query_terms = [term for term in self.tokenize_for_bm25(user_message) if len(term) > 3]
        strong_terms = [term for term in query_terms if term not in {"which", "them", "those", "these", "student", "students", "intern", "interns", "person", "people", "working"}]
        return bool(strong_terms) and sum(1 for term in strong_terms if term in source_text) >= 1

    def find_phrase_matched_entities(self, user_message: str, entities: Optional[list[dict]] = None) -> list[dict]:
        candidate_entities = entities or self.entity_registry
        phrases = [
            phrase
            for phrase in self.extract_query_named_phrases(user_message)
            if self.is_probable_person_name(phrase)
        ]
        if not phrases:
            return []

        matched_entities: list[dict] = []
        seen_unit_ids: set[str] = set()
        for entity in candidate_entities:
            source_text = self.strip_embedding_labels(entity.get("detail_text", "") or entity.get("summary_text", "")).lower()
            if not source_text:
                continue
            if not any(phrase.lower() in source_text for phrase in phrases):
                continue

            unit_id = entity.get("unit_id", "")
            if unit_id and unit_id in seen_unit_ids:
                continue
            if unit_id:
                seen_unit_ids.add(unit_id)
            matched_entities.append(entity)

        return matched_entities

    def find_exact_or_phrase_matched_entities(self, user_message: str, entities: Optional[list[dict]] = None) -> list[dict]:
        matched_entities = self.find_matching_entities(user_message, entities)
        seen_unit_ids = {
            entity.get("unit_id", "")
            for entity in matched_entities
            if entity.get("unit_id", "")
        }
        for entity in self.find_phrase_matched_entities(user_message, entities):
            unit_id = entity.get("unit_id", "")
            if unit_id and unit_id in seen_unit_ids:
                continue
            if unit_id:
                seen_unit_ids.add(unit_id)
            matched_entities.append(entity)
        return matched_entities

    def collapse_entities_by_normalized_name(self, entities: list[dict]) -> list[dict]:
        collapsed_entities: dict[str, dict] = {}
        for entity in entities:
            section_name = entity.get("section_name", "").strip()
            if not section_name:
                continue
            normalized_name = self.normalize_entity_name(section_name)
            if not normalized_name:
                continue

            current_best = collapsed_entities.get(normalized_name)
            candidate_score = (
                1 if entity.get("detail_text") else 0,
                1 if entity.get("summary_text") else 0,
                len(entity.get("section_name", "")),
            )
            if current_best is None:
                collapsed_entities[normalized_name] = entity
                continue

            current_score = (
                1 if current_best.get("detail_text") else 0,
                1 if current_best.get("summary_text") else 0,
                len(current_best.get("section_name", "")),
            )
            if candidate_score > current_score:
                collapsed_entities[normalized_name] = entity

        return list(collapsed_entities.values())

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

    def find_best_section_entity(self, user_message: str, query_route: Optional[dict]) -> Optional[dict]:
        section_entities = [
            entity for entity in self.filter_entities_by_route(query_route) if entity.get("entity_type") == "section"
        ]
        if not section_entities:
            return None

        lowered_query = user_message.lower()
        query_terms = [term for term in self.tokenize_for_bm25(user_message) if len(term) > 2]
        best_entity = None
        best_score = float("-inf")

        for entity in section_entities:
            section_name = entity.get("section_name", "").strip().lower()
            source_text = self.strip_embedding_labels(entity.get("detail_text", "") or entity.get("summary_text", "")).lower()
            score = 0.0

            if "what does ssl do" in lowered_query or "what we do" in lowered_query:
                if "what we do" in section_name:
                    score += 3.0
            if "categories of work" in lowered_query or "main categories of work" in lowered_query:
                if "what we do" in section_name:
                    score += 3.0
            if "mission" in lowered_query:
                if "who we are" in section_name or "what we do" in section_name:
                    score += 2.5
                if "mission is to" in source_text:
                    score += 2.0
            if "vision" in lowered_query and "vision" in section_name:
                score += 3.0
            if "contact" in lowered_query and "contact us" in section_name:
                score += 3.0

            score += sum(0.15 for term in query_terms if term in source_text or term in section_name)

            if score > best_score:
                best_score = score
                best_entity = entity

        return best_entity if best_score > 0 else None

    def extract_section_headings(self, section_text: str) -> list[str]:
        headings: list[str] = []
        for line in section_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if lowered in {"what we do", "who we are", "our vision", "contact us"}:
                continue
            if stripped.endswith(".") or stripped.endswith(":"):
                continue
            if len(stripped) > 90:
                continue
            if stripped.isupper():
                continue
            word_count = len(stripped.split())
            if 3 <= word_count <= 8 and any(character.isupper() for character in stripped):
                headings.append(stripped)
        return list(dict.fromkeys(headings))

    def extract_mission_statement(self, section_text: str) -> str:
        mission_match = re.search(r"mission is to:\s*(.+)", section_text, re.IGNORECASE)
        if not mission_match:
            return ""

        lines = [mission_match.group(1).strip()]
        for line in section_text[mission_match.end():].splitlines():
            stripped = line.strip()
            if not stripped:
                if lines:
                    break
                continue
            if re.match(r"^[12]\)", stripped):
                lines.append(stripped)
            elif lines and stripped[0].islower():
                lines[-1] = f"{lines[-1]} {stripped}"
            else:
                break

        return "\n".join(line for line in lines if line)

    def requested_people_groups(self, user_message: str, query_route: Optional[dict]) -> set[str]:
        lowered_query = user_message.lower()
        requested_groups: set[str] = set()

        if any(term in lowered_query for term in ("staff", "team", "employee", "employees")):
            requested_groups.add("staff_member")
        if any(term in lowered_query for term in ("affiliate", "affiliates", "faculty affiliate", "university affiliate")):
            requested_groups.update({"affiliate", "visiting_scholar"})
        if any(term in lowered_query for term in ("student", "students", "intern", "interns", "fellow", "fellows", "alumni")):
            requested_groups.add("person")
        if any(term in lowered_query for term in ("board", "leadership", "board member", "board members")):
            requested_groups.add("board_member")

        titles = set((query_route or {}).get("target_titles", []))
        if "Staff" in titles:
            requested_groups.add("staff_member")
        if "UniversityAffiliates" in titles:
            requested_groups.update({"affiliate", "visiting_scholar"})
        if "StudentsInterns" in titles:
            requested_groups.add("person")
        if "BoardOfDirectors" in titles:
            requested_groups.add("board_member")

        return requested_groups

    def is_multi_group_people_overview(self, user_message: str, query_route: Optional[dict]) -> bool:
        lowered_query = user_message.lower()
        overview_markers = ("overview", "people involved", "who are the people", "tell me about the people")
        requested_groups = self.requested_people_groups(user_message, query_route)
        return len(requested_groups) >= 2 and (
            any(marker in lowered_query for marker in overview_markers) or "including" in lowered_query
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

    def extract_project_access_bullets(self, project_text: str) -> list[str]:
        lines = [line.strip() for line in project_text.splitlines()]
        access_items: list[str] = []
        collecting = False
        for line in lines:
            lowered = line.lower()
            if "grants you access to" in lowered:
                collecting = True
                continue
            if not collecting:
                continue
            if not line:
                if access_items:
                    break
                continue
            if line.startswith("##"):
                break
            if line.lower().startswith("we welcome"):
                break
            access_items.append(line.lstrip("-* ").strip())

        return [item for item in access_items if item]

    def should_use_section_registry(self, user_message: str, query_route: Optional[dict]) -> bool:
        if not self.entity_registry:
            return False

        lowered_query = user_message.lower()
        section_markers = (
            "what does ssl do",
            "what we do",
            "categories of work",
            "main categories of work",
            "mission",
            "vision",
            "year in review",
            "contact",
        )
        return any(marker in lowered_query for marker in section_markers)

    def answer_from_section_registry(self, user_message: str, query_route: Optional[dict]) -> Optional[dict]:
        section = self.find_best_section_entity(user_message, query_route)
        if not section:
            return None

        section_text = self.strip_embedding_labels(section.get("detail_text", "") or section.get("summary_text", ""))
        lowered_query = user_message.lower()
        reply = ""

        if any(marker in lowered_query for marker in ("what does ssl do", "what we do", "categories of work", "main categories of work")):
            headings = self.extract_section_headings(section_text)
            if headings:
                reply_lines = ["SSL describes its work in these main categories:"]
                for index, heading in enumerate(headings, start=1):
                    reply_lines.append(f"{index}. {heading} [1]")
                reply = "\n".join(reply_lines)

        elif "mission" in lowered_query:
            mission_statement = self.extract_mission_statement(section_text)
            if mission_statement:
                reply = f"SSL described its mission this way:\n{mission_statement} [1]"

        elif "vision" in lowered_query:
            reply = f"{section_text} [1]"

        elif "contact" in lowered_query:
            reply = f"{section_text} [1]"

        if not reply:
            return None

        return {
            "reply": reply,
            "sources": [
                {
                    "title": section.get("title", "Untitled source"),
                    "url": section.get("source_url", "URL not provided"),
                    "source_path": section.get("source_path", "Unknown source"),
                }
            ],
            "needs_clarification": False,
            "clarification_options": [],
        }

    def infer_entity_inventory_type(self, user_message: str, query_route: Optional[dict]) -> str:
        lowered_query = user_message.lower()
        titles = set((query_route or {}).get("target_titles", []))
        if self.is_multi_group_people_overview(user_message, query_route):
            return ""
        if any(term in lowered_query for term in ("student", "students", "intern", "interns", "alumni", "person", "people")):
            return "person"
        if "projects" in lowered_query or "initiatives" in lowered_query or "programs" in lowered_query or "Projects" in titles:
            return "project"
        if "board" in lowered_query or "leadership" in lowered_query or "BoardOfDirectors" in titles:
            return "board_member"
        if "affiliate" in lowered_query or "UniversityAffiliates" in titles:
            return "affiliate"
        if "staff" in lowered_query or "team" in lowered_query or "Staff" in titles:
            return "staff_member"
        if "StudentsInterns" in titles:
            return "person"
        return ""

    def should_use_entity_registry(self, user_message: str, query_route: Optional[dict]) -> bool:
        if not self.entity_registry:
            return False

        lowered_query = user_message.lower()
        question_type = (query_route or {}).get("question_type", "")
        enumeration_markers = (
            "who are",
            "list",
            "name all",
            "name several",
            "several members",
            "which members",
            "which one of them",
            "which of them",
            "which of those",
            "which of the",
            "which student",
            "which intern",
            "which person",
            "who among",
            "all of them",
            "what are the current",
            "what are our current",
            "current projects",
        )
        count_markers = ("how many", "count")
        category_terms = ("staff", "board", "students", "interns", "affiliates", "projects", "initiatives", "programs")
        matched_entities = self.find_exact_or_phrase_matched_entities(user_message)

        if self.is_multi_group_people_overview(user_message, query_route):
            return True

        if matched_entities and all(entity.get("entity_type") != "project" for entity in matched_entities):
            return True

        if (
            len(matched_entities) == 1
            and matched_entities[0].get("entity_type") == "project"
            and any(term in lowered_query for term in ("benefit", "benefits", "join", "joining", "access", "membership", "member"))
        ):
            return True

        if self.is_specific_entity_detail_query(user_message):
            return False

        if question_type == "people_lookup" and any(marker in lowered_query for marker in enumeration_markers):
            return True

        if any(marker in lowered_query for marker in enumeration_markers) and any(term in lowered_query for term in category_terms):
            return True

        if any(marker in lowered_query for marker in count_markers) and any(term in lowered_query for term in category_terms) and not matched_entities:
            return True

        if question_type == "broad_overview" and any(term in lowered_query for term in ("projects", "initiatives", "staff", "board", "affiliates")):
            return True

        return False

    def answer_multi_group_people_overview(self, entities: list[dict], user_message: str) -> Optional[dict]:
        requested_groups = self.requested_people_groups(user_message, None)
        if len(requested_groups) < 2:
            return None

        group_definitions = [
            ("staff_member", "Staff"),
            ("affiliate", "Affiliates"),
            ("visiting_scholar", "Affiliates"),
            ("person", "Students and Interns"),
            ("board_member", "Board Members"),
        ]
        grouped_entities: dict[str, list[dict]] = {}
        for entity in entities:
            entity_type = entity.get("entity_type", "")
            if entity_type not in requested_groups:
                continue
            label = dict(group_definitions).get(entity_type)
            if not label:
                continue
            grouped_entities.setdefault(label, [])
            grouped_entities[label].append(entity)

        if len(grouped_entities) < 2:
            return None

        lines = ["Here is an overview of the people involved with SSL across the requested groups:"]
        sources: list[dict] = []
        source_seen: set[tuple[str, str]] = set()
        citation_index = 1

        for label in ["Staff", "Affiliates", "Students and Interns", "Board Members"]:
            group_entities = grouped_entities.get(label, [])
            if not group_entities:
                continue
            names = [entity.get("section_name", "") for entity in group_entities if entity.get("section_name")]
            preview = ", ".join(names[:4])
            if len(names) > 4:
                preview = f"{preview}, and {len(names) - 4} more"
            lines.append(f"- {label}: {len(names)} people, including {preview} [{citation_index}]")

            source = {
                "title": group_entities[0].get("title", "Untitled source"),
                "url": group_entities[0].get("source_url", "URL not provided"),
                "source_path": group_entities[0].get("source_path", "Unknown source"),
            }
            source_key = (source["title"], source["source_path"])
            if source_key not in source_seen:
                source_seen.add(source_key)
                sources.append(source)
            citation_index += 1

        return {
            "reply": "\n".join(lines),
            "sources": sources,
            "needs_clarification": False,
            "clarification_options": [],
        }

    def answer_from_entity_registry(self, user_message: str, query_route: Optional[dict]) -> dict:
        entities = self.filter_entities_by_route(query_route)
        lowered_query = user_message.lower()
        if not entities:
            return {
                "reply": "I do not have enough information in the entity registry to answer that.",
                "sources": [],
                "needs_clarification": False,
                "clarification_options": [],
            }

        aggregated_people_overview = self.answer_multi_group_people_overview(entities, user_message)
        if aggregated_people_overview:
            return aggregated_people_overview

        exact_matches = self.collapse_entities_by_normalized_name(
            self.find_exact_or_phrase_matched_entities(user_message, entities)
        )

        if any(term in lowered_query for term in ("benefit", "benefits", "join", "joining", "access", "membership", "member")):
            project_matches = [entity for entity in exact_matches if entity.get("entity_type") == "project"]
            if len(project_matches) == 1:
                project_text = self.strip_embedding_labels(project_matches[0].get("detail_text", "") or project_matches[0].get("summary_text", ""))
                access_items = self.extract_project_access_bullets(project_text)
                if access_items:
                    reply_lines = ["Joining the Northeast Climate Justice Research Collaborative gives members access to:"]
                    for index, item in enumerate(access_items, start=1):
                        reply_lines.append(f"{index}. {item} [1]")
                    return {
                        "reply": "\n".join(reply_lines),
                        "sources": [
                            {
                                "title": project_matches[0].get("title", "Untitled source"),
                                "url": project_matches[0].get("source_url", "URL not provided"),
                                "source_path": project_matches[0].get("source_path", "Unknown source"),
                            }
                        ],
                        "needs_clarification": False,
                        "clarification_options": [],
                    }

        if (
            len(exact_matches) == 1
            and not any(marker in lowered_query for marker in ("list", "name all", "name several", "who are", "how many", "count"))
        ):
            entity = exact_matches[0]
            entity_type = entity.get("entity_type")
            if (
                entity_type == "project"
                and any(term in lowered_query for term in ("benefit", "benefits", "join", "joining", "access", "membership", "member"))
            ):
                project_text = self.strip_embedding_labels(entity.get("detail_text", "") or entity.get("summary_text", ""))
                access_items = self.extract_project_access_bullets(project_text)
                if access_items:
                    reply_lines = ["Joining the Northeast Climate Justice Research Collaborative gives members access to:"]
                    for index, item in enumerate(access_items, start=1):
                        reply_lines.append(f"{index}. {item} [1]")
                    return {
                        "reply": "\n".join(reply_lines),
                        "sources": [
                            {
                                "title": entity.get("title", "Untitled source"),
                                "url": entity.get("source_url", "URL not provided"),
                                "source_path": entity.get("source_path", "Unknown source"),
                            }
                        ],
                        "needs_clarification": False,
                        "clarification_options": [],
                    }

            if entity_type == "project":
                exact_matches = []
            else:
                summary_text = self.strip_embedding_labels(entity.get("summary_text", "") or entity.get("detail_text", ""))
                if summary_text:
                    reply = f"{summary_text} [1]"
                    return {
                        "reply": reply,
                        "sources": [
                            {
                                "title": entity.get("title", "Untitled source"),
                                "url": entity.get("source_url", "URL not provided"),
                                "source_path": entity.get("source_path", "Unknown source"),
                            }
                        ],
                        "needs_clarification": False,
                        "clarification_options": [],
                    }

        requested_entity_type = self.infer_entity_inventory_type(user_message, query_route)
        if requested_entity_type:
            filtered_entities = [
                entity for entity in entities if entity.get("entity_type") == requested_entity_type
            ]
            if filtered_entities:
                entities = filtered_entities

        if self.is_group_selection_follow_up(user_message):
            focused_entities = [entity for entity in entities if self.entity_matches_query_focus(entity, user_message)]
            if focused_entities:
                entities = focused_entities

        count_only = any(marker in lowered_query for marker in ("how many", "count"))
        max_listed = min(len(entities), 20)
        listed_entities = entities[:max_listed]

        if requested_entity_type == "project":
            label = "projects or initiatives"
        elif requested_entity_type == "board_member":
            label = "board members"
        elif requested_entity_type == "affiliate":
            label = "affiliates"
        elif requested_entity_type == "staff_member":
            label = "staff members"
        else:
            label = "people or entities"

        lines = [f"I found {len(entities)} {label} in the matched corpus scope."]
        include_roles = requested_entity_type in {"board_member", "staff_member", "affiliate"} or "role" in lowered_query
        if not count_only or len(entities) <= 20:
            lines.append("")
            for index, entity in enumerate(listed_entities, start=1):
                role = self.extract_entity_role(entity) if include_roles else ""
                if role:
                    lines.append(f"{index}. {entity['section_name']} — {role} [{index}]")
                else:
                    lines.append(f"{index}. {entity['section_name']} [{index}]")

        if len(entities) > max_listed:
            lines.append("")
            lines.append(f"I listed the first {max_listed} entities above.")

        sources = [
            {
                "title": entity.get("title", "Untitled source"),
                "url": entity.get("source_url", "URL not provided"),
                "source_path": entity.get("source_path", "Unknown source"),
            }
            for entity in listed_entities
        ]

        deduped_sources: list[dict] = []
        seen_source_keys: set[tuple[str, str]] = set()
        for source in sources:
            key = (source["title"], source["source_path"])
            if key in seen_source_keys:
                continue
            seen_source_keys.add(key)
            deduped_sources.append(source)

        return {
            "reply": "\n".join(lines).strip(),
            "sources": deduped_sources,
            "needs_clarification": False,
            "clarification_options": [],
        }

    def should_use_document_registry(self, user_message: str, query_route: Optional[dict]) -> bool:
        if not self.document_registry:
            return False

        lowered_query = user_message.lower()
        question_type = (query_route or {}).get("question_type", "")
        inventory_markers = (
            "list",
            "name all",
            "all of them",
            "how many",
            "count",
            "which documents",
            "which publications",
            "what publications",
            "what reports",
        )

        if question_type in {"publication_inventory", "list_inventory"}:
            return True

        if any(marker in lowered_query for marker in inventory_markers) and any(
            term in lowered_query for term in ("publication", "publications", "report", "reports", "document", "documents")
        ):
            return True

        return False

    def answer_from_document_registry(self, user_message: str, query_route: Optional[dict]) -> dict:
        documents = self.filter_documents_by_route(query_route)
        lowered_query = user_message.lower()
        if not documents:
            return {
                "reply": "I do not have enough information in the document registry to answer that.",
                "sources": [],
                "needs_clarification": False,
                "clarification_options": [],
            }

        count_only = any(marker in lowered_query for marker in ("how many", "count"))
        max_listed = min(len(documents), 20)
        listed_documents = documents[:max_listed]

        lines = []
        if any(term in lowered_query for term in ("publication", "publications")):
            label = "publication source documents"
        elif any(term in lowered_query for term in ("report", "reports")):
            label = "report source documents"
        else:
            label = "source documents"

        lines.append(f"I found {len(documents)} {label} in the matched corpus scope.")
        if not count_only or len(documents) <= 20:
            lines.append("")
            for index, document in enumerate(listed_documents, start=1):
                lines.append(f"{index}. {document['title']} [{index}]")

        if len(documents) > max_listed:
            lines.append("")
            lines.append(f"I listed the first {max_listed} documents above.")

        sources = [
            {
                "title": document["title"],
                "url": document.get("source_url", "URL not provided"),
                "source_path": document["source_path"],
            }
            for document in listed_documents
        ]

        return {
            "reply": "\n".join(lines).strip(),
            "sources": sources,
            "needs_clarification": False,
            "clarification_options": [],
        }

    def build_prompt(
        self,
        user_message: str,
        retrieved_context: list[str],
        recent_history: Optional[list[ConversationTurn]] = None,
        rewritten_query: Optional[str] = None,
    ) -> str:
        if retrieved_context:
            numbered_blocks = [f"[{index}]\n{block}" for index, block in enumerate(retrieved_context, start=1)]
            retrieved_text = "\n\n".join(numbered_blocks)
        else:
            retrieved_text = "No relevant context found."
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
When you state facts, include inline citations using the retrieved source numbers like [1] or [2].
Only cite numbers that appear in the retrieved context.
For list answers, include citations on each bullet when possible.
{history_section}{rewritten_query_section}

Retrieved context:
{retrieved_text}

Question:
{user_message}
""".strip()

    def assess_retrieval_confidence(
        self,
        user_message: str,
        query_route: dict,
        retrieved_context: list[str],
        retrieved_metadata: list[dict],
        retrieval_diagnostics: dict,
        recent_history: Optional[list[ConversationTurn]] = None,
    ) -> dict:
        recent_history = recent_history or []
        ambiguous = self.is_ambiguous_query(user_message)
        question_type = query_route.get("question_type", "specific_fact")
        broad_query = query_route.get("prefer_summary", False) or question_type in {
            "broad_overview",
            "list_inventory",
            "publication_inventory",
            "comparison",
        }

        score = 0.0
        reasons: list[str] = []

        if retrieved_context:
            score += 0.4
        else:
            reasons.append("no_context")

        selected_count = retrieval_diagnostics.get("selected_count", 0)
        distinct_source_count = retrieval_diagnostics.get("distinct_source_count", 0)
        top_score = float(retrieval_diagnostics.get("top_score", 0.0))
        score_gap = float(retrieval_diagnostics.get("score_gap", 0.0))

        if broad_query:
            if selected_count >= 3:
                score += 0.15
            else:
                reasons.append("limited_context_coverage")
            if distinct_source_count >= 2:
                score += 0.1
            else:
                reasons.append("narrow_source_coverage")
        else:
            if selected_count >= 1:
                score += 0.15
            if distinct_source_count >= 1:
                score += 0.1

        if top_score >= 0.65:
            score += 0.15
        else:
            reasons.append("low_top_candidate_score")

        if score_gap >= 0.12:
            score += 0.1
        else:
            reasons.append("weak_score_gap")

        if query_route.get("routing_mode") != "global":
            score += 0.05
        elif ambiguous or recent_history:
            reasons.append("global_route_on_contextual_query")

        if ambiguous:
            score -= 0.2
            reasons.append("ambiguous_query")
        if ambiguous and not recent_history:
            score -= 0.1
            reasons.append("ambiguous_without_history")

        if question_type in {"people_lookup", "follow_up"} and distinct_source_count > 2:
            score -= 0.1
            reasons.append("diffuse_people_sources")
        if question_type in {"publication_inventory", "list_inventory"} and selected_count < 4:
            score -= 0.15
            reasons.append("insufficient_enumeration_coverage")

        normalized_score = max(0.0, min(score, 1.0))
        threshold = 0.55 if ambiguous else 0.5
        return {
            "score": round(normalized_score, 3),
            "is_low_confidence": normalized_score < threshold,
            "reasons": list(dict.fromkeys(reasons)),
        }

    def answer(self, user_message: str, recent_history: Optional[list[ConversationTurn]] = None) -> dict:
        recent_history = recent_history or []
        structured_follow_up = self.resolve_recent_entity_follow_up(user_message, recent_history)
        if structured_follow_up and structured_follow_up.get("needs_clarification"):
            return {
                "reply": structured_follow_up.get("clarifying_question", "Can you clarify what you mean?"),
                "sources": [],
                "needs_clarification": True,
                "clarification_for": user_message,
                "clarification_options": structured_follow_up.get("clarification_options", []),
            }

        rewritten_query = structured_follow_up.get("rewritten_query", user_message) if structured_follow_up else user_message
        is_follow_up_ambiguous = self.is_ambiguous_query(user_message)
        query_route = structured_follow_up.get("query_route") if structured_follow_up else self.detect_local_query_route(rewritten_query)

        if self.should_use_section_registry(rewritten_query, query_route):
            section_result = self.answer_from_section_registry(rewritten_query, query_route)
            if section_result:
                return section_result

        if self.should_use_entity_registry(rewritten_query, query_route):
            return self.answer_from_entity_registry(rewritten_query, query_route)

        if self.should_use_document_registry(rewritten_query, query_route):
            return self.answer_from_document_registry(rewritten_query, query_route)

        retrieval_k = self.choose_top_k(query_route)
        retrieved_context, retrieved_metadata, retrieval_diagnostics = self.retrieve_context(
            rewritten_query,
            top_k=retrieval_k,
            query_route=query_route,
        )
        confidence = self.assess_retrieval_confidence(
            user_message=rewritten_query,
            query_route=query_route,
            retrieved_context=retrieved_context,
            retrieved_metadata=retrieved_metadata,
            retrieval_diagnostics=retrieval_diagnostics,
            recent_history=recent_history,
        )
        query_plan = None

        if confidence["is_low_confidence"]:
            query_plan = self.plan_query_with_llm(user_message=user_message, recent_history=recent_history)
            rewritten_query = query_plan.get("rewritten_query", user_message)

            if query_plan.get("needs_clarification"):
                return {
                    "reply": query_plan.get("clarifying_question", "Can you clarify what you mean?"),
                    "sources": [],
                    "needs_clarification": True,
                    "clarification_for": user_message,
                    "clarification_options": query_plan.get("clarification_options", []),
                }

            if self.should_use_section_registry(rewritten_query, query_plan):
                section_result = self.answer_from_section_registry(rewritten_query, query_plan)
                if section_result:
                    return section_result

            if self.should_use_entity_registry(rewritten_query, query_plan):
                return self.answer_from_entity_registry(rewritten_query, query_plan)

            if self.should_use_document_registry(rewritten_query, query_plan):
                return self.answer_from_document_registry(rewritten_query, query_plan)

            retrieval_k = self.choose_top_k(query_plan)
            retrieved_context, retrieved_metadata, retrieval_diagnostics = self.retrieve_context(
                rewritten_query,
                top_k=retrieval_k,
                query_route=query_plan,
            )

        if self.should_ask_clarifying_question(
            original_query=user_message,
            rewritten_query=rewritten_query,
            retrieved_context=retrieved_context,
            retrieved_metadata=retrieved_metadata,
        ):
            fallback_question = (query_plan or {}).get("clarifying_question") or self.build_generic_clarifying_question(
                user_message=user_message,
                query_plan=query_plan,
            )
            return {
                "reply": fallback_question,
                "sources": self.extract_sources(retrieved_metadata),
                "needs_clarification": True,
                "clarification_for": user_message,
                "clarification_options": (query_plan or {}).get("clarification_options", []),
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
            "needs_clarification": False,
            "clarification_options": [],
        }

    def choose_top_k(self, query_route: Optional[dict] = None) -> int:
        if not query_route:
            return self.config.top_k

        if query_route.get("question_type") in {"broad_overview", "list_inventory", "publication_inventory", "comparison"}:
            return 8

        return 8 if query_route.get("prefer_summary") else self.config.top_k

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

    def should_ask_clarifying_question(
        self,
        original_query: str,
        rewritten_query: str,
        retrieved_context: list[str],
        retrieved_metadata: list[dict],
    ) -> bool:
        if not retrieved_context:
            return True

        if self.is_group_selection_follow_up(original_query):
            return False

        ambiguous = self.is_ambiguous_query(original_query)
        distinct_sources = {
            (metadata.get("source_path"), metadata.get("title"))
            for metadata in retrieved_metadata
            if metadata
        }
        weak_context = len(retrieved_context) < 2 or len(distinct_sources) == 1
        rewrite_failed = ambiguous and rewritten_query.strip().lower() == original_query.strip().lower()
        return weak_context and (ambiguous or rewrite_failed)

    def build_generic_clarifying_question(self, user_message: str, query_plan: Optional[dict] = None) -> str:
        question_type = (query_plan or {}).get("question_type", "")
        if question_type in {"people_lookup", "follow_up"}:
            return "Can you clarify which person you mean?"
        if question_type in {"publication_inventory", "list_inventory"}:
            return "Can you clarify which set of documents you want me to list?"
        if question_type == "contact":
            return "Can you clarify whether you want contact information, location, or both?"
        return f"Can you clarify what you mean by \"{user_message}\"?"

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
