const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const chatMessages = document.getElementById("chatMessages");
const messageTemplate = document.getElementById("messageTemplate");
const loadingTemplate = document.getElementById("loadingTemplate");
const suggestionsTemplate = document.getElementById("suggestionsTemplate");
const statusDot = document.querySelector(".status-dot");
const sidebarList = document.getElementById("sidebarList");
let messageCounter = 0;

function assistantLabelMarkup(label) {
  return `
    <span class="assistant-icon" aria-hidden="true">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
      </svg>
    </span>
    ${escapeHtml(label)}
  `;
}

function addSidebarEntry(text, messageId) {
  if (!sidebarList) return null;

  const empty = sidebarList.querySelector(".sidebar-empty");
  if (empty) empty.remove();

  const item = document.createElement("li");
  item.className = "sidebar-item";

  const button = document.createElement("button");
  button.type = "button";
  button.className = "sidebar-link";
  button.title = text;
  button.textContent = text;
  button.addEventListener("click", () => {
    const target = document.getElementById(messageId);
    if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  item.appendChild(button);
  sidebarList.prepend(item);

  const items = sidebarList.querySelectorAll(".sidebar-item");
  if (items.length > 10) {
    items[items.length - 1].remove();
  }

  return item;
}

function setStatus(processing) {
  if (!statusDot) return;
  statusDot.classList.toggle("processing", processing);
}

const recentHistory = [];
const recentHistoryWindow = 4;

const suggestedQuestionsEl = document.getElementById("suggestedQuestions");
if (suggestedQuestionsEl) {
  suggestedQuestionsEl.querySelectorAll(".suggested-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const question = btn.textContent.trim();
      suggestedQuestionsEl.remove();
      submitMessageFlow(question);
    });
  });
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderInlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/__(.+?)__/g, "<strong>$1</strong>");
}

function stripInlineCitations(text) {
  return text
    .replace(/\s*\[(?:\d+(?:\s*,\s*\d+)*)\]/g, "")
    .replace(/[ \t]+\./g, ".")
    .replace(/[ \t]+,/g, ",")
    .replace(/[ \t]+:/g, ":")
    .replace(/\(\s+/g, "(")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\n[ \t]+/g, "\n")
    .trim();
}

function renderAssistantContent(content) {
  const lines = stripInlineCitations(content).split("\n");
  const htmlParts = [];
  let currentParagraph = [];
  let currentList = [];

  function flushParagraph() {
    if (currentParagraph.length === 0) return;
    htmlParts.push(`<p>${currentParagraph.join("<br>")}</p>`);
    currentParagraph = [];
  }

  function flushList() {
    if (currentList.length === 0) return;
    htmlParts.push(`<ul>${currentList.map((item) => `<li>${item}</li>`).join("")}</ul>`);
    currentList = [];
  }

  lines.forEach((line) => {
    const bulletMatch = line.match(/^\s*[*-]\s+(.*)$/);
    if (bulletMatch) {
      flushParagraph();
      currentList.push(renderInlineMarkdown(bulletMatch[1]));
      return;
    }
    if (line.trim() === "") {
      flushParagraph();
      flushList();
      return;
    }
    flushList();
    currentParagraph.push(renderInlineMarkdown(line));
  });

  flushParagraph();
  flushList();

  return htmlParts.join("") || `<p>${renderInlineMarkdown(content)}</p>`;
}

function buildSourcesNode(sources) {
  const sourcesNode = document.createElement("div");
  sourcesNode.className = "message-sources";

  const sourcesLabel = document.createElement("span");
  sourcesLabel.className = "sources-label";
  sourcesLabel.textContent = "Sources";
  sourcesNode.appendChild(sourcesLabel);

  sources.forEach((source) => {
    const sourceLink = document.createElement("a");
    sourceLink.className = "source-chip";
    sourceLink.target = "_blank";
    sourceLink.rel = "noreferrer";
    sourceLink.href = source.url !== "URL not provided" ? source.url : "#";
    sourceLink.textContent = source.title;
    if (source.url === "URL not provided") {
      sourceLink.classList.add("source-chip-disabled");
      sourceLink.removeAttribute("href");
    }
    sourcesNode.appendChild(sourceLink);
  });

  return sourcesNode;
}

function buildClarificationReply(option, originalQuestion) {
  const opt = (option || "").trim();
  const q = (originalQuestion || "").trim();
  if (!opt) return q;
  if (!q) return `Tell me more about ${opt}.`;

  const lowerQuestion = q.toLowerCase();
  const optHasProjectWord = /\b(project|initiative|program)\b/i.test(opt);
  const replacement =
    /\b(project|initiative|program)\b/.test(lowerQuestion) && !optHasProjectWord
      ? `the ${opt} project`
      : opt;

  if (/\bthis project\b/i.test(q)) return q.replace(/\bthis project\b/gi, replacement);
  if (/\bthat project\b/i.test(q)) return q.replace(/\bthat project\b/gi, replacement);
  if (/\bthis initiative\b/i.test(q)) return q.replace(/\bthis initiative\b/gi, replacement);
  if (/\bthat initiative\b/i.test(q)) return q.replace(/\bthat initiative\b/gi, replacement);
  if (/\bthis program\b/i.test(q)) return q.replace(/\bthis program\b/gi, replacement);
  if (/\bthat program\b/i.test(q)) return q.replace(/\bthat program\b/gi, replacement);

  return `Regarding my earlier question "${q}", I meant ${replacement}.`;
}

function appendMessage(role, label, content, sources = [], clarificationOptions = [], clarificationFor = "", onOptionSelect = null) {
  const fragment = messageTemplate.content.cloneNode(true);
  const messageNode = fragment.querySelector(".message");
  const labelNode = fragment.querySelector(".message-label");
  const bubbleNode = fragment.querySelector(".message-bubble");

  messageNode.classList.add(role);

  let sidebarItem = null;
  if (role === "user") {
    const id = `msg-${++messageCounter}`;
    messageNode.id = id;
    sidebarItem = addSidebarEntry(content, id);
  }
  if (role === "assistant") {
    labelNode.innerHTML = assistantLabelMarkup(label);
  } else {
    labelNode.textContent = label;
  }

  if (role === "assistant") {
    bubbleNode.innerHTML = renderAssistantContent(content);
  } else {
    bubbleNode.textContent = content;
  }

  chatMessages.appendChild(fragment);
  const liveNode = chatMessages.lastElementChild;

  if (role === "assistant" && sources.length > 0) {
    liveNode.appendChild(buildSourcesNode(sources));
  }

  if (role === "assistant" && clarificationOptions.length > 0) {
    const optionsNode = document.createElement("div");
    optionsNode.className = "message-options";
    clarificationOptions.forEach((option) => {
      const optionButton = document.createElement("button");
      optionButton.type = "button";
      optionButton.className = "option-bubble";
      optionButton.textContent = option;
      optionButton.addEventListener("click", () => {
        if (typeof onOptionSelect === "function") {
          onOptionSelect(buildClarificationReply(option, clarificationFor));
        }
      });
      optionsNode.appendChild(optionButton);
    });
    liveNode.appendChild(optionsNode);
  }

  chatMessages.scrollTop = chatMessages.scrollHeight;
  return sidebarItem;
}

function appendStreamingBubble(label) {
  const fragment = messageTemplate.content.cloneNode(true);
  const messageNode = fragment.querySelector(".message");
  const labelNode = fragment.querySelector(".message-label");
  const bubbleNode = fragment.querySelector(".message-bubble");

  messageNode.classList.add("assistant");
  labelNode.innerHTML = assistantLabelMarkup(label);
  bubbleNode.textContent = "";

  chatMessages.appendChild(fragment);
  const liveNode = chatMessages.lastElementChild;
  const liveBubble = liveNode.querySelector(".message-bubble");

  let rawText = "";

  return {
    addChunk(text) {
      rawText += text;
      liveBubble.textContent = rawText;
      chatMessages.scrollTop = chatMessages.scrollHeight;
    },
    finalize(sources = []) {
      liveBubble.innerHTML = renderAssistantContent(rawText);
      if (sources.length > 0) {
        liveNode.appendChild(buildSourcesNode(sources));
      }
      chatMessages.scrollTop = chatMessages.scrollHeight;
      return rawText;
    },
  };
}

function appendLoading() {
  const fragment = loadingTemplate.content.cloneNode(true);
  chatMessages.appendChild(fragment);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return chatMessages.lastElementChild;
}


function renderSuggestions(suggestions, targetNode) {
  const chips = suggestionsTemplate.content.cloneNode(true);
  const list = chips.querySelector(".suggestion-chips-list");

  suggestions.forEach((text) => {
    const btn = document.createElement("button");
    btn.className = "suggestion-chip";
    btn.type = "button";
    btn.textContent = text;
    btn.addEventListener("click", () => {
      messageInput.value = text;
      chatForm.requestSubmit();
    });
    list.appendChild(btn);
  });

  targetNode.appendChild(chips);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}


async function streamMessage(message, onEvent) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      recent_history: recentHistory.slice(-recentHistoryWindow),
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    let errMsg = "Chat request failed.";
    try { errMsg = JSON.parse(text).error || errMsg; } catch {}
    throw new Error(errMsg);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      try {
        onEvent(JSON.parse(line.slice(6)));
      } catch {}
    }
  }
}

function restoreSidebarPlaceholder() {
  if (!sidebarList) return;

  if (!sidebarList.querySelector(".sidebar-item")) {
    const empty = document.createElement("li");
    empty.className = "sidebar-empty";
    empty.textContent = "Your questions will appear here.";
    sidebarList.appendChild(empty);
  }
}

async function submitMessageFlow(message, displayMessage = message) {
  if (!message) return;

  const suggestedEl = document.getElementById("suggestedQuestions");
  if (suggestedEl) suggestedEl.remove();
  chatMessages.querySelectorAll(".suggestion-chips").forEach(el => el.remove());

  const sidebarItem = appendMessage("user", "You", displayMessage);
  messageInput.value = "";
  messageInput.focus();
  sendButton.disabled = true;

  let loadingNode = appendLoading();
  setStatus(true);

  let streaming = null;
  let pendingSources = [];
  let fullReply = "";
  let suggestionAnchor = null;

  try {
    await streamMessage(message, (event) => {
      if (event.done && event.reply !== undefined) {
        // Early return: clarification, registry answer, or blocked message
        if (loadingNode) { loadingNode.remove(); loadingNode = null; }

        if (event.blocked && sidebarItem) {
          sidebarItem.remove();
          restoreSidebarPlaceholder();
        }

        appendMessage(
          "assistant",
          "Sustainable Labs",
          event.reply,
          event.needs_clarification ? [] : (event.sources || []),
          event.clarification_options || [],
          event.clarification_for || message,
          async (clarifiedMessage) => {
            if (sendButton.disabled) return;
            await submitMessageFlow(clarifiedMessage, clarifiedMessage);
          }
        );

        fullReply = event.reply;
        if (!event.blocked) {
          recentHistory.push({ user: message, assistant: fullReply });
          if (recentHistory.length > recentHistoryWindow) {
            recentHistory.splice(0, recentHistory.length - recentHistoryWindow);
          }
        }
      } else if (event.type === "meta") {
        pendingSources = event.sources || [];
      } else if (event.type === "delta") {
        fullReply += event.delta || "";
        if (!streaming) {
          if (loadingNode) { loadingNode.remove(); loadingNode = null; }
          streaming = appendStreamingBubble("Sustainable Labs");
        }
        streaming.addChunk(event.delta);
      } else if (event.type === "done") {
        if (streaming) {
          streaming.finalize(pendingSources);
          streaming = null;
          recentHistory.push({ user: message, assistant: fullReply });
          if (recentHistory.length > recentHistoryWindow) {
            recentHistory.splice(0, recentHistory.length - recentHistoryWindow);
          }
        }
        // Unlock UI immediately — suggestions will still arrive after this
        setStatus(false);
        sendButton.disabled = false;
        suggestionAnchor = chatMessages.lastElementChild;
      } else if (event.type === "suggestions") {
        if (suggestionAnchor && event.suggestions && event.suggestions.length > 0) {
          renderSuggestions(event.suggestions, suggestionAnchor);
        }
      } else if (event.type === "error") {
        if (loadingNode) { loadingNode.remove(); loadingNode = null; }
        appendMessage("assistant", "Sustainable Labs", event.error || "An error occurred.");
      }
    });
  } catch (error) {
    if (loadingNode) { loadingNode.remove(); loadingNode = null; }
    if (streaming) { streaming.finalize([]); streaming = null; }
    appendMessage("assistant", "Sustainable Labs", error.message);
  } finally {
    setStatus(false);
    sendButton.disabled = false;
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  await submitMessageFlow(message);
});
