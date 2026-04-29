# Sustainable Labs ChatBot

A RAG (Retrieval-Augmented Generation) chatbot for the UMass Boston Sustainable Solutions Lab. Built by Team 1 "RAG's to Riches".

---

## Features

- Answers questions about SSL research projects, publications, staff, initiatives, funding, and community partnerships
- Responses are drawn directly from SSL source documents via semantic search
- Recent questions sidebar for in-session navigation
- Streaming responses — text appears token by token as Gemini generates it
- Content filter blocking profanity, hate speech, threats, and SSL/UMB-targeted harassment

---

## Setup

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

4. Open `http://localhost:5000` in your browser.

---

## Changes & Updates (UI Fork)

### UI Overhaul
- Redesigned the full interface to match the UMass Boston / SSL color scheme (navy, blue, yellow accent)
- Added animated gradient header with UMass Boston logo
- Added links to the SSL website, UMass Boston homepage, and a mailto button for `ssl@umb.edu`
- Rounded buttons to match the UMB site style
- Fixed color inconsistencies across components

### Recent Questions Sidebar
- Added a sidebar that logs questions asked during the current session
- Clicking a question scrolls the chat back to that message

### Layout & Sizing
- Condensed the hero section (title + description) to give the chat window more vertical space
- Reduced hero font size and padding
- Compacted the footer into a single slim row with the Email Us button moved to the left

### Suggested Questions
- Six clickable suggested questions appear below the welcome message on load
- Styled as pill buttons in the SSL blue color scheme
- Clicking one submits it as a normal message and dismisses the suggestions
- Suggestions also dismiss automatically when the user types their own question

### Welcome Message
- Expanded the opening assistant message to introduce the bot, clarify what it can help with, and include contact info (`ssl@umb.edu` and the SSL website)

### Content Filter
- Added a profanity and harmful content filter using the `better-profanity` library
- Covers: profanity, slurs, hate speech, threats, explicit content, and SSL/UMB-targeted phrases
- Custom whitelist for legitimate academic/research terms that would otherwise be false positives (e.g. `sex`, `dam`, `weed`, `assessment`, `massachusetts`, bird species names, etc.)
- Custom block list for org-specific harmful phrases (e.g. "ssl is a scam", "climate change is fake", "kill yourself")
- Blocked messages are shown as a polite refusal in the chat and are **not** added to the recent questions sidebar

### Streaming Responses
- Responses now stream token by token instead of waiting for the full reply
- Backend uses Gemini's `generate_content_stream` API via SSE (Server-Sent Events)
- Frontend reads the stream with `fetch` + `ReadableStream` and updates the chat bubble in real time
- Markdown rendering is applied once the stream completes

### API Performance Improvements
- **Singleton Gemini client** — the `genai.Client` is now created once at startup and reused across all requests, eliminating repeated authentication overhead
- **Reduced `max_output_tokens`** — lowered from 8192 to 1024, appropriate for a RAG chatbot and reduces unnecessary generation time

### Error Handling
- Gemini API 503/high-demand and 429/quota errors now show a friendly message ("The assistant is experiencing high demand right now. Please wait a moment and try again.") instead of a raw error
- All other unexpected errors fall back to a generic "Something went wrong. Please try again." message

### Bug Fixes
- Sources no longer appear when the bot asks for clarification — fixed in both the backend (clarification return paths now always send `sources: []`) and the frontend (sources suppressed client-side when `needs_clarification` is true). **Note: clarification suppression is not fully working yet and is still being investigated.**

### Upstream Merge
- Merged `DavidL72Code/Sustainable-Labs-ChatBot` pipeline update (`b717be9`) into this UI fork
- Resolved conflict in `Chatbot.py`: adopted upstream's `MAX_CHROMA_BATCH_SIZE` class constant (5000) over the local hardcoded value (500)
- Resolved conflict in `styles.css`: kept both upstream's `.option-bubble` styles and existing UI styles

---

## Dependencies

| Package | Purpose |
|---|---|
| `flask` | Web server |
| `google-genai` | Gemini LLM API |
| `chromadb` | Vector store for embeddings |
| `sentence-transformers` | Local embedding model (`all-MiniLM-L6-v2`) |
| `langchain-text-splitters` | Document chunking |
| `pypdf` | PDF ingestion |
| `python-dotenv` | Environment variable loading |
| `better-profanity` | Content filtering |
