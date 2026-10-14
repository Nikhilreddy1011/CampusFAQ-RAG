/**
 * Campus FAQ Chat Assistant Client Script
 * Handles question submission, semantic answer rendering, citation inspection,
 * and conversational state.
 */

document.addEventListener("DOMContentLoaded", () => {
  const chatMessages = document.getElementById("chat-messages");
  const chatForm = document.getElementById("chat-form");
  const queryInput = document.getElementById("student-query-input");
  const sendBtn = document.getElementById("send-btn");
  const loadingIndicator = document.getElementById("loading-indicator");
  const clearChatBtn = document.getElementById("clear-chat-btn");
  const exampleChips = document.querySelectorAll(".example-chip");

  let isSubmitting = false;

  // Auto-resize textarea as student types
  queryInput.addEventListener("input", () => {
    queryInput.style.height = "auto";
    queryInput.style.height = Math.max(58, Math.min(queryInput.scrollHeight, 150)) + "px";
  });

  // Enter to send (Shift+Enter for newline)
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isSubmitting) {
        submitQuestion();
      }
    }
  });

  // Form submit
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    if (!isSubmitting) {
      submitQuestion();
    }
  });

  // Example chips click
  exampleChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const query = chip.getAttribute("data-query");
      if (query && !isSubmitting) {
        queryInput.value = query;
        submitQuestion();
      }
    });
  });

  // Clear chat button
  clearChatBtn.addEventListener("click", () => {
    chatMessages.innerHTML = `
      <div class="message-row bot-row">
        <div class="avatar bot-avatar" aria-hidden="true">
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"/>
          </svg>
        </div>
        <div class="message-bubble bot-bubble">
          <div class="message-meta">
            <span class="sender-name">Campus Assistant</span>
            <span class="time-stamp">Just now</span>
          </div>
          <div class="message-text">
            Conversation reset. How can I assist you with campus policies or services?
          </div>
        </div>
      </div>
    `;
    queryInput.focus();
  });

  async function submitQuestion() {
    const query = queryInput.value.trim();
    if (!query) return;

    isSubmitting = true;
    sendBtn.disabled = true;
    queryInput.value = "";
    queryInput.style.height = "auto";

    // Append student's message
    appendUserMessage(query);

    // Show loading
    loadingIndicator.classList.remove("hidden");
    scrollToBottom();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error (${response.status})`);
      }

      const data = await response.json();
      appendBotResponse(data);
    } catch (err) {
      appendErrorMessage(err.message || "Failed to contact campus helpdesk service.");
    } finally {
      loadingIndicator.classList.add("hidden");
      isSubmitting = false;
      sendBtn.disabled = false;
      queryInput.focus();
      scrollToBottom();
    }
  }

  function appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row user-row";
    row.innerHTML = `
      <div class="avatar user-avatar" aria-hidden="true">You</div>
      <div class="message-bubble user-bubble">
        <div class="message-meta">
          <span class="sender-name">Student</span>
          <span class="time-stamp">${getCurrentTime()}</span>
        </div>
        <div class="message-text">${escapeHtml(text)}</div>
      </div>
    `;
    chatMessages.appendChild(row);
  }

  function appendBotResponse(data) {
    const row = document.createElement("div");
    row.className = "message-row bot-row";

    const formattedAnswer = formatAnswerText(data.answer);
    const hasCitations = data.citations && data.citations.length > 0;

    let citationsHtml = "";
    if (hasCitations) {
      const citationCards = data.citations
        .map(
          (c) => `
        <div class="source-card" id="source-card-${c.id}">
          <div class="source-card-header">
            <span class="source-title">
              <span class="citation-tag">[${c.id}]</span>
              ${escapeHtml(c.source_label)}
            </span>
            <span class="source-similarity">${c.similarity_percent}% match</span>
          </div>
          <div class="source-excerpt">${escapeHtml(c.excerpt)}</div>
        </div>
      `
        )
        .join("");

      citationsHtml = `
        <div class="sources-container">
          <button class="sources-toggle-btn" type="button" aria-expanded="false" onclick="toggleSources(this)">
            <svg style="width:14px;height:14px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
            </svg>
            <span>View Supporting Sources (${data.citations.length})</span>
          </button>
          <div class="sources-list hidden">
            ${citationCards}
          </div>
        </div>
      `;
    }

    const modelLabel = data.synthesizer === "llama3.2" ? "Grounded • Llama 3.2" : "Grounded Answer";
    const badgeHtml = data.grounded
      ? `<span class="badge-grounded">
           <svg style="width:12px;height:12px;" fill="none" stroke="currentColor" viewBox="0 0 24 24">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
           </svg>
           ${modelLabel}
         </span>`
      : `<span class="badge-fallback">
           <svg style="width:12px;height:12px;" fill="none" stroke="currentColor" viewBox="0 0 24 24">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
           </svg>
           Not Found in Documents
         </span>`;

    row.innerHTML = `
      <div class="avatar bot-avatar" aria-hidden="true">
        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"/>
        </svg>
      </div>
      <div class="message-bubble bot-bubble">
        <div class="message-meta">
          <span class="sender-name">Campus Assistant</span>
          <span class="time-stamp">${getCurrentTime()}</span>
        </div>
        <div class="message-text">${formattedAnswer}</div>
        <div class="grounded-badge-container">
          ${badgeHtml}
          <span class="search-latency">Retrieved in ${data.search_time_ms} ms</span>
        </div>
        ${citationsHtml}
      </div>
    `;

    chatMessages.appendChild(row);
  }

  function appendErrorMessage(msg) {
    const row = document.createElement("div");
    row.className = "message-row bot-row";
    row.innerHTML = `
      <div class="avatar bot-avatar" aria-hidden="true">
        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
        </svg>
      </div>
      <div class="message-bubble bot-bubble">
        <div class="message-meta">
          <span class="sender-name">Campus Assistant</span>
          <span class="time-stamp">${getCurrentTime()}</span>
        </div>
        <div class="message-text" style="color:var(--danger)">
          <strong>Notice:</strong> ${escapeHtml(msg)}
        </div>
      </div>
    `;
    chatMessages.appendChild(row);
  }

  function formatAnswerText(text) {
    let clean = escapeHtml(text);
    // Convert **bold** to <strong>
    clean = clean.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    // Convert citations like [1] or [2] into clickable citation tags
    clean = clean.replace(/\[(\d+)\]/g, (match, p1) => {
      return `<button class="citation-tag" type="button" aria-label="View citation ${p1}" onclick="focusCitation(${p1})">[${p1}]</button>`;
    });
    // Convert bullet lists
    clean = clean.replace(/\n• /g, "<br>• ");
    return clean;
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function getCurrentTime() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});

// Global helpers for inline citations and accordion toggling
function toggleSources(button) {
  const list = button.nextElementSibling;
  const isHidden = list.classList.contains("hidden");
  if (isHidden) {
    list.classList.remove("hidden");
    button.setAttribute("aria-expanded", "true");
    button.querySelector("span").textContent = button
      .querySelector("span")
      .textContent.replace("View", "Hide");
  } else {
    list.classList.add("hidden");
    button.setAttribute("aria-expanded", "false");
    button.querySelector("span").textContent = button
      .querySelector("span")
      .textContent.replace("Hide", "View");
  }
}

function focusCitation(citationId) {
  const targetCard = document.getElementById(`source-card-${citationId}`);
  if (targetCard) {
    const list = targetCard.closest(".sources-list");
    if (list && list.classList.contains("hidden")) {
      const toggleBtn = list.previousElementSibling;
      if (toggleBtn) toggleSources(toggleBtn);
    }
    targetCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    targetCard.style.outline = "2px solid var(--accent)";
    targetCard.style.transition = "outline 0.3s ease";
    setTimeout(() => {
      targetCard.style.outline = "none";
    }, 2000);
  }
}
