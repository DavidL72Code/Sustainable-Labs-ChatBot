const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const chatMessages = document.getElementById("chatMessages");
const messageTemplate = document.getElementById("messageTemplate");
const loadingTemplate = document.getElementById("loadingTemplate");
const recentHistory = [];
const recentHistoryWindow = 4;

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
  const trimmedQuestion = (originalQuestion || "").trim();
  if (!trimmedQuestion) {
    return option;
  }

  return `Regarding my earlier question "${trimmedQuestion}", I meant ${option}.`;
}

function appendMessage(role, label, content, sources = [], clarificationOptions = [], clarificationFor = "", onOptionSelect = null) {
  const fragment = messageTemplate.content.cloneNode(true);
  const messageNode = fragment.querySelector(".message");
  const labelNode = fragment.querySelector(".message-label");
  const bubbleNode = fragment.querySelector(".message-bubble");

  messageNode.classList.add(role);
  labelNode.textContent = label;
  if (role === "assistant") {
    bubbleNode.innerHTML = renderAssistantContent(content);
  } else {
    bubbleNode.textContent = content;
  }

  if (role === "assistant" && sources.length > 0) {
    const sourcesNode = document.createElement("div");
    sourcesNode.className = "message-sources";

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

  return payload;
}

async function submitMessageFlow(message, displayMessage = message) {
  if (!message) {
    return;
  }

  appendMessage("user", "You", displayMessage);
  messageInput.value = "";
  messageInput.focus();
  sendButton.disabled = true;

  const loadingNode = appendLoading();

  try {
    const result = await sendMessage(message);
    loadingNode.remove();
    appendMessage(
      "assistant",
      "Sustainable Labs",
      result.reply,
      result.sources || [],
      result.clarification_options || [],
      result.clarification_for || message,
      async (clarifiedMessage) => {
        if (sendButton.disabled) {
          return;
        }
        await submitMessageFlow(clarifiedMessage, clarifiedMessage);
      }
    );
    recentHistory.push({
      user: message,
      assistant: result.reply,
    });
    if (recentHistory.length > recentHistoryWindow) {
      recentHistory.splice(0, recentHistory.length - recentHistoryWindow);
    }
  } catch (error) {
    loadingNode.remove();
    appendMessage("assistant", "Sustainable Labs", error.message);
  } finally {
    sendButton.disabled = false;
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const message = messageInput.value.trim();
  await submitMessageFlow(message);
});
