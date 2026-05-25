---
title: SSL Research Assistant
emoji: 🌱
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Sustainable Labs ChatBot

A RAG (Retrieval-Augmented Generation) chatbot for the UMass Boston Sustainable Solutions Lab. Built by Team 1 "RAG's to Riches".

> The YAML frontmatter above is consumed by Hugging Face Spaces (Docker SDK). GitHub treats it as metadata and ignores it in rendering.

---

## 1. What We Built & How It Works

The chatbot answers questions about SSL research projects, publications, staff, initiatives, funding, and community partnerships using only the lab's own source documents. Everything the model says is grounded in retrieved chunks — no free-form invention.

### User-Facing Features

- **Grounded answers** drawn directly from SSL source documents (annual reports, project pages, publications, staff bios).
- **Streaming responses** — text appears token by token as Gemini generates it, using Server-Sent Events.
- **Suggested questions** — six clickable pill buttons on first load to help users get started.
- **Recent questions sidebar** — in-session navigation that scrolls the chat back to a previous question on click.
- **Content filter** — blocks profanity, hate speech, threats, and SSL/UMB-targeted harassment with a custom whitelist for legitimate academic terms (e.g. `assessment`, `massachusetts`, bird species) and a custom block list for org-specific phrases.
- **Friendly error handling** — Gemini 503/429 errors surface as "high demand, try again" instead of raw stack traces.
- **Analytics dashboard** at `/dashboard` for reviewing chat history, source mappings, retrieval diagnostics, confidence scores, latency, and evaluation results.

### The Retrieval Pipeline

The interesting work happens *before* the LLM is called. A user question goes through these stages:

#### a) Intent Classification & Query Routing

When a question comes in, the chatbot first figures out **what kind of question it is** and **which slice of the corpus is most relevant**. This is done in three layers:

1. **Keyword-based local router** ([`detect_local_query_route`](Chatbot.py#L1388)) — a fast, deterministic classifier that tags the question with:
   - A **question type**: `broad_overview`, `specific_fact`, `people_lookup`, `publication_inventory`, or `list_inventory`.
   - A **scope**: which document titles, categories, and folders to filter retrieval to (e.g. "staff" → `Staff`, `SSLAbout`; "board" → `BoardOfDirectors`; "publications" → `Publications`).
   - A **`prefer_summary` hint** that biases ranking toward short summary chunks vs. detail chunks.
2. **Heuristic LLM-planning gate** ([`should_use_llm_planning`](Chatbot.py#L104)) — decides whether the question is ambiguous enough to deserve a more expensive LLM-powered planning call. Skips the LLM when confidence is decent, the query is short, targets are already found, or the topic is obviously clear. Saves tokens and latency on easy questions.
3. **LLM query planner** ([`plan_query_with_llm`](Chatbot.py#L1722)) — only invoked when the gate says the heuristic wasn't enough. Given a catalog of titles, categories, folders, and entity names, Gemini picks the right routing scope itself.

#### b) Ensemble / Hybrid Search

Retrieval runs two search engines in parallel and fuses their results:

1. **Dense retrieval** ([`retrieve_dense_candidates`](Chatbot.py#L1093)) — semantic vector search over ChromaDB using `all-MiniLM-L6-v2` embeddings. Strong at paraphrase and conceptual matches.
2. **BM25 sparse retrieval** ([`retrieve_bm25_candidates`](Chatbot.py#L1135)) — lexical keyword search. Strong at exact terms, names, and acronyms that vector search sometimes misses.
3. **Reciprocal Rank Fusion** ([`fuse_candidates`](Chatbot.py#L1183)) — combines the two ranked lists with the RRF formula (`weight / (60 + rank)`). Weights are adaptive ([`get_hybrid_weights`](Chatbot.py#L1228)): summary-preferred queries favor dense, fact-lookup queries favor BM25, hard-routed queries weight both equally.
4. **Metadata-aware reranking** ([`rerank_candidates`](Chatbot.py#L1308)) — boosts candidates whose source path, title, category, or folder matches the route, plus exact-term hits in body or section name. The final ordering is what gets sent to the LLM.

#### c) Adaptive Candidate Pool

[`choose_candidate_pool`](Chatbot.py#L141) sizes the candidate pool based on the question type: broad overviews pull more candidates for coverage, specific-fact questions pull fewer to save retrieval work (roughly 15–20% reduction over a fixed multiplier).

#### d) Entity Resolution & Follow-ups

A separate entity registry is built at ingest time from staff, board, affiliate, and project sections. At query time the bot tries to:
- Match exact and phrased person/project names ([`find_phrase_matched_entities`](Chatbot.py#L2421)).
- Resolve pronouns and "what about her?" follow-ups against recent conversation turns ([`resolve_recent_entity_follow_up`](Chatbot.py#L2179)).
- Detect multi-group people-overview asks vs. specific-entity-detail asks so the response shape matches the question.

#### e) Generation

Retrieved chunks are formatted into a prompt and sent to Gemini via a **singleton client** (created once at startup, reused across requests). `max_output_tokens` is set to 1024 — appropriate for RAG and faster than the default 8192. The response is streamed back to the client over SSE.

### Document Ingestion

At first run, [`SEED_DOCUMENTS/`](SEED_DOCUMENTS/) is parsed into structured units:
- Project pages get split per project ([`split_project_sections`](Chatbot.py#L455)).
- Staff/board/affiliate pages get split per person with name detection ([`split_people_sections`](Chatbot.py#L590)).
- Slide decks get split per slide ([`split_slide_sections`](Chatbot.py#L547)).
- Everything else is chunked with `RecursiveCharacterTextSplitter`.

Each chunk is embedded and stored in ChromaDB with rich metadata (title, category, folder, source path, section name, chunk level). The metadata is what makes routing and reranking possible.

---

## 2. Tech Stack

### Backend
| Technology | Role |
|---|---|
| **Python 3** | Language |
| **Flask** | Web server, REST API, SSE streaming |
| **Google Gemini** (`google-genai`) | LLM for answer generation and query planning. Default model: `gemini-3.1-flash-lite` |
| **ChromaDB** | Local vector store for embeddings + metadata |
| **sentence-transformers** (`all-MiniLM-L6-v2`) | Local embedding model — runs on CPU, no API calls |
| **BM25** (custom implementation) | Sparse lexical retrieval, paired with dense for hybrid search |
| **langchain-text-splitters** | `RecursiveCharacterTextSplitter` for chunking |
| **pypdf** | PDF document ingestion |
| **better-profanity** | Content filtering with custom whitelist/blocklist |
| **python-dotenv** | Local environment variable loading |

### Frontend
| Technology | Role |
|---|---|
| **HTML / CSS / vanilla JS** | No frameworks — keeps the UI fast and dependency-free |
| **Server-Sent Events (SSE)** | Token-by-token streaming from Gemini |
| **`fetch` + `ReadableStream`** | Client-side stream consumption |
| **Markdown rendering** | Applied once a streamed response completes |
| **UMass Boston / SSL color scheme** | Navy/blue gradient header, yellow accents, UMB logo |

### Dev & Evaluation
| Technology | Role |
|---|---|
| **`run_questions_eval.py`** | Batch evaluation harness over `questions.json` |
| **`benchmark_planner.py`, `benchmark_stream.py`, `benchmark_suggestions.py`** | Performance benchmarks for planner, streaming, and suggestion latency |
| **`question_eval_iter*.json`** | Iterative evaluation snapshots used to track regressions and improvements |
| **Analytics dashboard** | Live diagnostics: per-interaction trace JSON, retrieval diagnostics, source usage, corpus coverage, problem cases (blocked, clarification, error, low-confidence), evaluation summary with score key |

### Evaluation Score Key
- `correctness_vs_corpus`: 1–5 rating for how well the answer matches the SSL corpus reference.
- `citations`: 1–5 rating for whether returned sources are useful and relevant support.
- `hallucinated`: `yes` means the evaluator found unsupported or clearly incorrect facts.
- `answered_question`: `yes` means the answer directly addressed the question that was asked.
- `right_citations`: `yes` means the cited or returned sources match the relevant corpus sources.

---

## 3. Deployment

### Local Development

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set your Gemini API key**
   ```bash
   export GEMINI_API_KEY=your_key_here
   ```
3. **Run the server**
   ```bash
   python3 Chatbot.py
   ```
4. Open `http://localhost:7860` for the chat UI.
5. Open `http://localhost:7860/dashboard` for the analytics dashboard.

The default port is `7860` (Hugging Face Spaces convention). Override with `PORT` or `CHATBOT_PORT` if needed. Set `CHATBOT_HOST=127.0.0.1` to bind to localhost only.

### Split Deployment: Hugging Face (backend) + Vercel (frontend)

To avoid the Hugging Face Spaces iframe chrome, the backend and frontend deploy separately:

```
[ Vercel: static site ]  ── fetch ──►  [ Hugging Face Space (Docker): Flask API ]
   frontend/index.html                   /api/chat, /api/suggestions
   frontend/static/                      ChromaDB + Gemini
```

#### Backend on Hugging Face Spaces (Docker SDK)

The repo is already wired up for this:
- [`Dockerfile`](Dockerfile) — installs deps, pre-downloads the embedding model, exposes port `7860`, runs `python Chatbot.py`.
- [`.dockerignore`](.dockerignore) — excludes dev/benchmark artifacts from the image.
- README frontmatter at the top of this file declares `sdk: docker` + `app_port: 7860`.
- [`Chatbot.py`](Chatbot.py) wires up `flask-cors` and reads `CORS_ORIGINS` from env.

Steps:

1. Create a new Hugging Face Space and pick **Docker** as the SDK.
2. Push this repo to the Space's git remote.
3. In **Space Settings → Variables and secrets**, set:
   - `GEMINI_API_KEY` — your Gemini key (secret).
   - `CORS_ORIGINS` — comma-separated list of allowed origins, e.g. `https://your-app.vercel.app,http://localhost:5173`. Defaults to `*` if unset.
   - Optionally `GEMINI_MODEL` to override the default model.
4. Wait for the Space to build. The endpoint is `https://<user>-<space>.hf.space`.
5. On free Spaces the filesystem is ephemeral. Either commit a prebuilt `chroma_db/` (uncomment the `COPY chroma_db/` line in the Dockerfile), rebuild from `SEED_DOCUMENTS/` at startup, or buy Persistent Storage.

#### Frontend on Vercel (static site)

The [`frontend/`](frontend/) directory is a ready-to-deploy static site.

1. Edit [`frontend/index.html`](frontend/index.html) and replace the `window.API_BASE` placeholder with your HF Space URL:
   ```html
   <script>
     window.API_BASE = "https://<user>-<space>.hf.space";
   </script>
   ```
2. Deploy:
   ```bash
   cd frontend
   npx vercel            # preview
   npx vercel --prod     # production
   ```
   Or import the `frontend/` folder in the Vercel dashboard — no build step needed, it's pure static.
3. After the first Vercel deploy, copy the production URL and add it to `CORS_ORIGINS` on your Hugging Face Space.
4. Open the Vercel URL — clean UI, no Hugging Face border.

#### Things to Watch

- **Cold starts** — free HF Spaces sleep after inactivity. First request after sleep takes 30–60s while ChromaDB and the embedding model reload. The Dockerfile pre-downloads the model to remove that part of the delay.
- **CORS** — `flask-cors` is wired to `/api/*` only. SSE works cross-origin as long as your Vercel domain is in `CORS_ORIGINS`.
- **Dashboard persistence** — interaction logs live on disk. On ephemeral filesystems they reset on every restart. The dashboard is still served by the backend at `<hf-space-url>/dashboard` and the frontend's Dashboard link points there.
- **`GEMINI_API_KEY`** — never commit it. Use Space secrets only.
