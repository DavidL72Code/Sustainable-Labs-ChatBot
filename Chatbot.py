from __future__ import annotations

import json
import os
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
    embedding_model_name: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
    web_host: str = os.getenv("CHATBOT_HOST", "127.0.0.1")
    web_port: int = int(os.getenv("CHATBOT_PORT", "8000"))


class SourceDocument(dict):
    pass


class RetrievalChatbot:
    def __init__(self, llm_callable: LLMCallable, config: Optional[ChatbotConfig] = None) -> None:
        self.config = config or ChatbotConfig()
        self.llm_callable = llm_callable
        self.embedder = SentenceTransformer(self.config.embedding_model_name)
        self.client = chromadb.PersistentClient(path=self.config.persist_directory)
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Collection:
        return self.client.get_or_create_collection(name=self.config.collection_name)

    def chunk_documents(self, documents: list[str]) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ".", " "],
        )
        return splitter.split_text("\n\n".join(documents))

    def split_document_into_chunks(self, text: str) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
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

            chunks = self.split_document_into_chunks(text)
            if not chunks:
                continue

            embeddings = self.embedder.encode(chunks, convert_to_numpy=True)

            for index, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{document['source_path']}::chunk-{index}"
                if chunk_id in existing_ids:
                    continue

                new_ids.append(chunk_id)
                new_chunks.append(chunk_text)
                new_embeddings.append(embedding.tolist())
                new_metadatas.append(
                    {
                        "source_path": document["source_path"],
                        "source_url": document["source_url"],
                        "title": document["title"],
                        "category": document["category"],
                        "document_type": document["document_type"],
                        "chunk_index": index,
                    }
                )

        if new_ids:
            self.collection.add(
                ids=new_ids,
                documents=new_chunks,
                embeddings=new_embeddings,
                metadatas=new_metadatas,
            )

    def retrieve_context(self, query: str, top_k: Optional[int] = None) -> list[str]:
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k or self.config.top_k,
            include=["documents", "metadatas"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        context_blocks: list[str] = []

        for chunk_text, metadata in zip(documents, metadatas):
            metadata = metadata or {}
            source_url = metadata.get("source_url", "URL not provided")
            title = metadata.get("title", "Untitled source")
            source_path = metadata.get("source_path", "Unknown source")
            chunk_index = metadata.get("chunk_index", "?")
            context_blocks.append(
                f"Title: {title}\n"
                f"Source URL: {source_url}\n"
                f"Source Path: {source_path}\n"
                f"Chunk Index: {chunk_index}\n\n"
                f"{chunk_text}"
            )

        return context_blocks

    def build_prompt(self, user_message: str, retrieved_context: list[str]) -> str:
        retrieved_text = "\n\n".join(retrieved_context) if retrieved_context else "No relevant context found."
        return f"""
You are the Sustainable Labs retrieval assistant. Answer only from the provided retrieved context.
If the answer is not supported by the context, say you do not have enough information.

Retrieved context:
{retrieved_text}

Question:
{user_message}
""".strip()

    def answer(self, user_message: str) -> str:
        retrieved_context = self.retrieve_context(user_message)
        prompt = self.build_prompt(user_message=user_message, retrieved_context=retrieved_context)
        return self.llm_callable(prompt).strip()


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
        if not user_message:
            return jsonify({"error": "Message is required."}), 400

        try:
            reply = chatbot.answer(user_message)
        except Exception as exc:  # pragma: no cover - user-facing error path
            return jsonify({"error": str(exc)}), 500

        return jsonify({"reply": reply})

    return app


def main() -> None:
    app = create_app()
    config = ChatbotConfig()
    app.run(debug=True, host=config.web_host, port=config.web_port)


if __name__ == "__main__":
    main()
