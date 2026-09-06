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
try {
  if (window.API_BASE) {
    window.localStorage.setItem("ssl_api_base", window.API_BASE.replace(/\/+$/, ""));
  }
} catch {}

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

function sidebarSessionRow(id, title) {
  // One row per session, labelled with the first message of that session —
  // the same shape the server already stores (visitor_conversations.title).
  const item = document.createElement("li");
  item.className = "sidebar-item";
  item.dataset.conversationId = id;

  const open = document.createElement("button");
  open.type = "button";
  open.className = "sidebar-link";
  open.title = title;
  open.textContent = title;
  open.addEventListener("click", () => openSavedConversation(id));

  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "sidebar-delete";
  remove.title = "Delete this chat";
  remove.setAttribute("aria-label", `Delete ${title}`);
  remove.textContent = "\u00d7";
  remove.addEventListener("click", async (event) => {
    event.stopPropagation();
    if (!window.confirm(`Delete "${title}"?\n\nThis cannot be undone.`)) return;
    const done = await fetch(apiUrl(`/api/visitor/conversations/${encodeURIComponent(id)}`), {
      method: "DELETE",
      credentials: "include",
    }).catch(() => null);
    if (!done || !done.ok) return;
    item.remove();
    if (conversationId === id) startNewConversation();
    if (!sidebarList.querySelector(".sidebar-item")) restoreSidebarPlaceholder();
  });

  item.append(open, remove);
  return item;
}

function addSessionToSidebar(id, title) {
  if (!sidebarList || !id) return;
  if (sidebarList.querySelector(`[data-conversation-id="${CSS.escape(id)}"]`)) return;
  const empty = sidebarList.querySelector(".sidebar-empty");
  if (empty) empty.remove();
  sidebarList.prepend(sidebarSessionRow(id, title || "New chat"));
  markActiveConversation(id);
}

function setStatus(processing) {
  if (!statusDot) return;
  statusDot.classList.toggle("processing", processing);
}

let conversationId = "";
let firstMessageOfSession = "";

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

function sanitizeSourceUrl(value) {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed || trimmed === "URL not provided") return null;

  try {
    const parsed = new URL(trimmed, window.location.origin);
    if (parsed.protocol === "http:" || parsed.protocol === "https:") {
      return parsed.href;
    }
  } catch {}

  return null;
}

function buildSourcesNode(sources) {
  const sourcesNode = document.createElement("div");
  sourcesNode.className = "message-sources";

  const sourcesLabel = document.createElement("span");
  sourcesLabel.className = "sources-label";
  sourcesLabel.textContent = "Sources";
  sourcesNode.appendChild(sourcesLabel);

  const seenSources = new Set();
  const uniqueSources = sources.filter((source) => {
    const key = [
      source.source_path || "",
      source.url || "",
      source.title || "",
    ].join("|").toLowerCase();
    if (seenSources.has(key)) return false;
    seenSources.add(key);
    return true;
  });

  uniqueSources.forEach((source) => {
    const sourceLink = document.createElement("a");
    sourceLink.className = "source-chip";
    sourceLink.target = "_blank";
    sourceLink.rel = "noreferrer";
    const safeUrl = sanitizeSourceUrl(source.url);
    sourceLink.textContent = source.title;
    if (safeUrl) {
      sourceLink.href = safeUrl;
    } else {
      sourceLink.classList.add("source-chip-disabled");
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
    // Anchor for in-page scrolling only. The sidebar lists sessions, not
    // individual questions: it used to prepend a row per message and cap at
    // ten, which meant a signed-in visitor's saved sessions were pushed out by
    // their own questions as soon as they started typing.
    messageNode.id = `msg-${++messageCounter}`;
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
  // The message that opens a session becomes its sidebar title, the same way
  // the server titles visitor_conversations.
  if (!conversationId) firstMessageOfSession = message;
  const response = await fetch((window.API_BASE || "") + "/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      conversation_id: conversationId,
    }),
  });

  const responseConversationId = response.headers.get("X-Conversation-Id");
  if (responseConversationId) {
    const isNewSession = responseConversationId && responseConversationId !== conversationId;
    conversationId = responseConversationId;
    // A brand-new session appears in the sidebar straight away, titled with
    // the message that started it — matching how the server titles it.
    if (isNewSession && signedIn) {
      addSessionToSidebar(responseConversationId, firstMessageOfSession || "New chat");
    }
  }

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
        }
        // Unlock UI immediately — suggestions will still arrive after this
        setStatus(false);
        sendButton.disabled = false;
        suggestionAnchor = chatMessages.lastElementChild;
      } else if (event.type === "session_full") {
        // The answer was still delivered; only saving stopped. Offer the new
        // session rather than letting history silently stop being recorded.
        openSessionFullDialog();
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

/* ---------------------------------------------------------------------------
 * Optional visitor accounts.
 *
 * Signing in only controls whether this visitor's own history is saved. Without
 * an account the assistant works exactly as before and nothing is stored, so
 * the UI never blocks the chat behind a login.
 * ------------------------------------------------------------------------- */
const accountArea = document.getElementById("accountArea");
const newChatButton = document.getElementById("newChatButton");
const confirmOverlay = document.getElementById("confirmOverlay");
const sessionFullOverlay = document.getElementById("sessionFullOverlay");
const sessionFullClose = document.getElementById("sessionFullClose");
const sessionFullNew = document.getElementById("sessionFullNew");
const confirmClose = document.getElementById("confirmClose");
const confirmOk = document.getElementById("confirmOk");
const accountEmail = document.getElementById("accountEmail");
const signInButton = document.getElementById("signInButton");
const signOutButton = document.getElementById("signOutButton");
const authOverlay = document.getElementById("authOverlay");
const authForm = document.getElementById("authForm");
const authEmail = document.getElementById("authEmail");
const authPassword = document.getElementById("authPassword");
const authError = document.getElementById("authError");
const authTitle = document.getElementById("authTitle");
const authSubmit = document.getElementById("authSubmit");
const authSwitchText = document.getElementById("authSwitchText");
const authSwitchButton = document.getElementById("authSwitchButton");
const authClose = document.getElementById("authClose");
const sidebarNote = document.getElementById("sidebarNote");
const forgotLink = document.getElementById("forgotLink");
const authEmailField = document.getElementById("authEmailField");
const authPasswordField = document.getElementById("authPasswordField");

let authMode = "login";
let signedIn = false;

function apiUrl(path) {
  // Served by Flask: no API_BASE, so relative paths are right. Served as the
  // static site on Vercel: API_BASE points at the deployed backend.
  const base = (typeof window !== "undefined" && window.API_BASE)
    ? String(window.API_BASE).replace(/\/+$/, "")
    : "";
  return `${base}${path}`;
}

// "recover" asks for an email only; "reset" sets a new password after the user
// arrives from the emailed link.
const AUTH_MODES = {
  login:   { title: "Sign in",           submit: "Sign in",          switchText: "New here?",                switchLabel: "Create an account", autocomplete: "current-password" },
  signup:  { title: "Create an account", submit: "Create account",   switchText: "Already have an account?", switchLabel: "Sign in",           autocomplete: "new-password" },
  recover: { title: "Reset your password", submit: "Email me a link", switchText: "Remembered it?",          switchLabel: "Sign in",           autocomplete: "current-password" },
  reset:   { title: "Choose a new password", submit: "Save password", switchText: "",                        switchLabel: "",                  autocomplete: "new-password" },
};

let recoveryToken = "";
let recoveryRefresh = "";

function setAuthMode(mode) {
  authMode = mode;
  const spec = AUTH_MODES[mode] || AUTH_MODES.login;
  authTitle.textContent = spec.title;
  authSubmit.textContent = spec.submit;
  authSwitchText.textContent = spec.switchText;
  authSwitchButton.textContent = spec.switchLabel;
  authSwitchButton.hidden = !spec.switchLabel;
  authPassword.setAttribute("autocomplete", spec.autocomplete);
  // Recovery collects an email only; the reset step collects a password only.
  // Toggle the wrapper divs, not parentElement -- the inputs sit directly in
  // the form, so hiding a "parent" would hide the whole dialog.
  if (authEmailField) authEmailField.hidden = mode === "reset";
  authEmail.required = mode !== "reset";
  if (authPasswordField) authPasswordField.hidden = mode === "recover";
  authPassword.required = mode !== "recover";
  if (forgotLink) forgotLink.hidden = mode !== "login";
  authError.hidden = true;
}

function openAuth() {
  setAuthMode("login");
  authOverlay.hidden = false;
  authEmail.focus();
}

function closeAuth() {
  authOverlay.hidden = true;
  authForm.reset();
  authError.hidden = true;
}

function showAuthError(message) {
  authError.textContent = message;
  authError.hidden = false;
}

function openConfirmDialog() {
  // A dialog of its own, not a line inside the sign-in form. Signing up
  // succeeded and the next step happens in the user's inbox, so the message
  // should not sit in the same place as "wrong password" on a form they have
  // just finished with.
  if (!confirmOverlay) return;
  closeAuth();
  confirmOverlay.hidden = false;
  confirmOk?.focus();
}

function closeConfirmDialog() {
  if (confirmOverlay) confirmOverlay.hidden = true;
}

function openSessionFullDialog() {
  if (!sessionFullOverlay) return;
  sessionFullOverlay.hidden = false;
  sessionFullNew?.focus();
}

function closeSessionFullDialog() {
  if (sessionFullOverlay) sessionFullOverlay.hidden = true;
}

function renderAccountState(email) {
  signedIn = Boolean(email);
  accountArea.hidden = false;
  accountEmail.textContent = email || "";
  accountEmail.hidden = !signedIn;
  signInButton.hidden = signedIn;
  signOutButton.hidden = !signedIn;
  if (sidebarNote) sidebarNote.hidden = signedIn;
}

async function refreshAccountState() {
  try {
    const response = await fetch(apiUrl("/api/visitor/session"), { credentials: "include" });
    if (!response.ok) return;
    const data = await response.json();
    // Accounts are only offered when the deployment has Supabase configured.
    if (!data.available) {
      accountArea.hidden = true;
      if (sidebarNote) sidebarNote.hidden = true;
      return;
    }
    renderAccountState(data.signed_in ? data.email : "");
    if (data.signed_in) await loadSavedConversations();
  } catch {
    /* Accounts are optional; a failure here must never block the chat. */
  }
}

function markActiveConversation(activeId) {
  if (!sidebarList) return;
  sidebarList.querySelectorAll(".sidebar-item").forEach((item) => {
    item.classList.toggle("is-active", item.dataset.conversationId === activeId);
  });
}

function startNewConversation() {
  // A fresh session: clear the id so the backend allocates a new one, empty
  // the transcript, and drop the active highlight. The saved chats stay.
  conversationId = "";
  firstMessageOfSession = "";
  if (chatMessages) chatMessages.innerHTML = "";
  markActiveConversation("");
}

async function loadSavedConversations() {
  if (!signedIn || !sidebarList) return;
  try {
    const response = await fetch(apiUrl("/api/visitor/conversations"), { credentials: "include" });
    if (!response.ok) return;
    const { conversations = [] } = await response.json();
    if (!conversations.length) return;
    sidebarList.innerHTML = "";
    conversations.forEach((conversation) => {
      sidebarList.appendChild(
        sidebarSessionRow(conversation.id, conversation.title || "Saved chat")
      );
    });
    markActiveConversation(conversationId);
  } catch {
    /* Leave whatever the sidebar already shows. */
  }
}

async function openSavedConversation(savedId) {
  try {
    const response = await fetch(
      apiUrl(`/api/visitor/conversations/${encodeURIComponent(savedId)}`),
      { credentials: "include" }
    );
    if (!response.ok) return;
    const { messages = [] } = await response.json();
    // Adopt the saved id so the next message continues this conversation.
    // Without this the transcript was rendered but the id stayed on whatever
    // was current, so replying to an old chat silently began a new one.
    conversationId = savedId;
    markActiveConversation(savedId);
    chatMessages.innerHTML = "";
    messages.forEach((message) => {
      appendMessage(
        message.role === "user" ? "user" : "assistant",
        message.role === "user" ? "You" : "Sustainable Labs",
        message.content || "",
        message.sources || []
      );
    });
    chatMessages.scrollTop = chatMessages.scrollHeight;
  } catch {
    /* Ignore: the live chat still works. */
  }
}

async function submitRecover(email) {
  await fetch(apiUrl("/api/visitor/recover"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ email, redirect_to: window.location.origin + window.location.pathname }),
  });
  // The reply is deliberately identical whether or not the address has an
  // account, so the message here has to be too.
  showAuthError("If that address has an account, a reset link is on its way.");
  setAuthMode("login");
}

async function submitReset(password) {
  const response = await fetch(apiUrl("/api/visitor/reset-password"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ access_token: recoveryToken, refresh_token: recoveryRefresh, password }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    showAuthError(data.error || "Could not set that password.");
    return;
  }
  recoveryToken = "";
  recoveryRefresh = "";
  renderAccountState(data.email || "");
  closeAuth();
  await loadSavedConversations();
}

authForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const email = authEmail.value.trim();
  const password = authPassword.value;

  if (authMode === "recover") {
    if (!email) { showAuthError("Enter the email address for your account."); return; }
    authSubmit.disabled = true;
    try { await submitRecover(email); } finally { authSubmit.disabled = false; }
    return;
  }
  if (authMode === "reset") {
    if (password.length < 8) { showAuthError("Use a password of at least 8 characters."); return; }
    authSubmit.disabled = true;
    try { await submitReset(password); } finally { authSubmit.disabled = false; }
    return;
  }

  if (!email || password.length < 8) {
    showAuthError("Enter your email and a password of at least 8 characters.");
    return;
  }
  authSubmit.disabled = true;
  try {
    const response = await fetch(apiUrl(`/api/visitor/${authMode === "signup" ? "signup" : "login"}`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ email, password }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      showAuthError(data.error || "That did not work. Please try again.");
      return;
    }
    if (data.confirm_email) {
      setAuthMode("login");
      openConfirmDialog();
      return;
    }
    renderAccountState(data.email || email);
    closeAuth();
    await loadSavedConversations();
  } catch {
    showAuthError("Could not reach the server. Please try again.");
  } finally {
    authSubmit.disabled = false;
  }
});

sessionFullNew?.addEventListener("click", () => {
  closeSessionFullDialog();
  startNewConversation();
  messageInput?.focus();
});
sessionFullClose?.addEventListener("click", closeSessionFullDialog);
sessionFullOverlay?.addEventListener("click", (event) => {
  if (event.target === sessionFullOverlay) closeSessionFullDialog();
});

confirmOk?.addEventListener("click", closeConfirmDialog);
confirmClose?.addEventListener("click", closeConfirmDialog);
confirmOverlay?.addEventListener("click", (event) => {
  if (event.target === confirmOverlay) closeConfirmDialog();
});
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (confirmOverlay && !confirmOverlay.hidden) closeConfirmDialog();
  if (sessionFullOverlay && !sessionFullOverlay.hidden) closeSessionFullDialog();
});

newChatButton?.addEventListener("click", () => {
  startNewConversation();
  messageInput?.focus();
});

signInButton?.addEventListener("click", openAuth);
authClose?.addEventListener("click", closeAuth);
authSwitchButton?.addEventListener("click", () =>
  setAuthMode(authMode === "signup" ? "login" : authMode === "recover" ? "login" : "signup"));
forgotLink?.addEventListener("click", () => setAuthMode("recover"));

// Supabase returns the user here with the recovery token in the URL fragment.
// Catch it on load and go straight to choosing a new password.
function handleRecoveryRedirect() {
  const hash = window.location.hash || "";
  if (!hash.includes("type=recovery")) return false;
  const params = new URLSearchParams(hash.replace(/^#/, ""));
  recoveryToken = params.get("access_token") || "";
  recoveryRefresh = params.get("refresh_token") || "";
  // Clear the fragment so the token is not left sitting in the address bar.
  history.replaceState(null, "", window.location.pathname + window.location.search);
  if (!recoveryToken) return false;
  accountArea.hidden = false;
  authOverlay.hidden = false;
  setAuthMode("reset");
  authPassword.focus();
  return true;
}
authOverlay?.addEventListener("click", (event) => {
  if (event.target === authOverlay) closeAuth();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && authOverlay && !authOverlay.hidden) closeAuth();
});

signOutButton?.addEventListener("click", async () => {
  await fetch(apiUrl("/api/visitor/logout"), { method: "POST", credentials: "include" });
  renderAccountState("");
  if (sidebarList) {
    sidebarList.innerHTML = '<li class="sidebar-empty">Your questions will appear here.</li>';
  }
});

if (!handleRecoveryRedirect()) refreshAccountState();
