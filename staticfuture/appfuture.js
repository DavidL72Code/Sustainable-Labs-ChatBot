const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const chatMessages = document.getElementById("chatMessages");
const messageTemplate = document.getElementById("messageTemplate");
const loadingTemplate = document.getElementById("loadingTemplate");
const statusDot = document.querySelector(".status-dot");
const sidebarList = document.getElementById("sidebarList");
const recentHistory = [];
const recentHistoryWindow = 4;
let messageCounter = 0;

function toText(value, fallback = "") {
  if (value === null || value === undefined) {
    return fallback;
  }
  return String(value);
}

function escapeHtml(value) {
  return toText(value)
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
  return toText(text)
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
    if (currentParagraph.length === 0) {
      return;
    }
    htmlParts.push(`<p>${currentParagraph.join("<br>")}</p>`);
    currentParagraph = [];
  }

  function flushList() {
    if (currentList.length === 0) {
      return;
    }
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

function buildClarificationReply(option, originalQuestion) {
  const opt = (option || "").trim();
  const q = (originalQuestion || "").trim();
  if (!opt) {
    return q;
  }
  if (!q) {
    return `Tell me more about ${opt}.`;
  }

  const optHasProjectWord = /\bproject\b/i.test(opt);
  const replacement =
    (/\b(project|initiative|program)\b/.test(lower) && !optHasProjectWord) ? `the ${opt} project` : opt;

  if (/\bthis project\b/i.test(q)) {
    return q.replace(/\bthis project\b/gi, replacement);
  }
  if (/\bthat project\b/i.test(q)) {
    return q.replace(/\bthat project\b/gi, replacement);
  }
  if (/\bthis initiative\b/i.test(q)) {
    return q.replace(/\bthis initiative\b/gi, replacement);
  }
  if (/\bthat initiative\b/i.test(q)) {
    return q.replace(/\bthat initiative\b/gi, replacement);
  }
  if (/\bthis program\b/i.test(q)) {
    return q.replace(/\bthis program\b/gi, replacement);
  }
  if (/\bthat program\b/i.test(q)) {
    return q.replace(/\bthat program\b/gi, replacement);
  }

  return `Tell me more about ${replacement}.`;
}

function addSidebarEntry(text, messageId) {
  if (!sidebarList) {
    return;
  }

  const empty = sidebarList.querySelector(".sidebar-empty");
  if (empty) {
    empty.remove();
  }

  const item = document.createElement("li");
  item.className = "sidebar-item";

  const button = document.createElement("button");
  button.type = "button";
  button.className = "sidebar-link";
  button.title = text;
  button.textContent = text;
  button.addEventListener("click", () => {
    const target = document.getElementById(messageId);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  });

  item.appendChild(button);
  sidebarList.prepend(item);

  const items = sidebarList.querySelectorAll(".sidebar-item");
  if (items.length > 10) {
    items[items.length - 1].remove();
  }
}

function setStatus(processing) {
  if (!statusDot) {
    return;
  }
  statusDot.classList.toggle("processing", processing);
}

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

function appendMessage(role, label, content, sources = [], clarificationOptions = [], clarificationFor = "", onOptionSelect = null) {
  const fragment = messageTemplate.content.cloneNode(true);
  const messageNode = fragment.querySelector(".message");
  const labelNode = fragment.querySelector(".message-label");
  const bubbleNode = fragment.querySelector(".message-bubble");

  messageNode.classList.add(role);

  if (role === "user") {
    const id = `msg-${++messageCounter}`;
    messageNode.id = id;
    addSidebarEntry(content, id);
  }

  if (role === "assistant") {
    labelNode.innerHTML = assistantLabelMarkup(label);
    bubbleNode.innerHTML = renderAssistantContent(content);
  } else {
    labelNode.textContent = label;
    bubbleNode.textContent = content;
  }

  if (role === "assistant" && sources.length > 0) {
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

    messageNode.appendChild(sourcesNode);
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

    messageNode.appendChild(optionsNode);
  }

  chatMessages.appendChild(fragment);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendLoading() {
  const fragment = loadingTemplate.content.cloneNode(true);
  const loadingNode = fragment.querySelector("#loadingMessage");
  chatMessages.appendChild(fragment);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return loadingNode;
}

async function sendMessage(message) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      recent_history: recentHistory.slice(-recentHistoryWindow),
    }),
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Chat request failed.");
  }

  return {
    ...payload,
    answer: toText(payload.answer ?? payload.reply, "I couldn't generate a response."),
    sources: Array.isArray(payload.sources) ? payload.sources : [],
    clarification_options: Array.isArray(payload.clarification_options) ? payload.clarification_options : [],
    clarification_for: toText(payload.clarification_for, ""),
  };
}

async function submitMessageFlow(message, displayMessage = message) {
  if (!message) {
    return;
  }

  appendMessage("user", "You", displayMessage);
  messageInput.value = "";
  messageInput.focus();
  sendButton.disabled = true;
  setStatus(true);

  const loadingNode = appendLoading();

  try {
    const result = await sendMessage(message);
    loadingNode.remove();

    appendMessage(
      "assistant",
      "SSL Assistant",
      result.answer,
      result.sources || [],
      result.clarification_options || [],
      result.clarification_for || "",
      (clarificationReply) => {
        submitMessageFlow(clarificationReply, clarificationReply);
      },
    );

    recentHistory.push({
      user: message,
      assistant: result.answer,
    });

    if (recentHistory.length > recentHistoryWindow) {
      recentHistory.splice(0, recentHistory.length - recentHistoryWindow);
    }
  } catch (error) {
    loadingNode.remove();
    appendMessage("assistant", "SSL Assistant", error.message || "Something went wrong.");
  } finally {
    sendButton.disabled = false;
    setStatus(false);
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) {
    messageInput.focus();
    return;
  }

  await submitMessageFlow(message);
});
