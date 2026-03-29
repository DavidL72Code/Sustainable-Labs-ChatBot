const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const chatMessages = document.getElementById("chatMessages");
const messageTemplate = document.getElementById("messageTemplate");
const loadingTemplate = document.getElementById("loadingTemplate");

function appendMessage(role, label, content) {
  const fragment = messageTemplate.content.cloneNode(true);
  const messageNode = fragment.querySelector(".message");
  const labelNode = fragment.querySelector(".message-label");
  const bubbleNode = fragment.querySelector(".message-bubble");

  messageNode.classList.add(role);
  labelNode.textContent = label;
  bubbleNode.textContent = content;

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
    }),
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Chat request failed.");
  }

  return payload.reply;
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const message = messageInput.value.trim();
  if (!message) {
    return;
  }

  appendMessage("user", "You", message);
  messageInput.value = "";
  messageInput.focus();
  sendButton.disabled = true;

  const loadingNode = appendLoading();

  try {
    const reply = await sendMessage(message);
    loadingNode.remove();
    appendMessage("assistant", "Sustainable Labs", reply);
  } catch (error) {
    loadingNode.remove();
    appendMessage("assistant", "Sustainable Labs", error.message);
  } finally {
    sendButton.disabled = false;
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
});
