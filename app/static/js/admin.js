/**
 * Campus FAQ Admin Portal Client Script
 * Handles document upload (drag-and-drop), sample ingestion, chunk inspection,
 * and document deletion.
 */

document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const progressContainer = document.getElementById("upload-progress-container");
  const progressBar = document.getElementById("upload-progress-bar");
  const statusText = document.getElementById("upload-status-text");
  const alertBox = document.getElementById("admin-alert");
  const loadSamplesBtn = document.getElementById("load-samples-btn");
  const chunksModal = document.getElementById("chunks-modal");
  const modalCloseBtn = document.getElementById("modal-close-btn");
  const modalTitle = document.getElementById("modal-title");
  const modalSubtitle = document.getElementById("modal-subtitle");
  const modalBody = document.getElementById("modal-body");

  // Drop zone events
  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      handleFileUpload(fileInput.files[0]);
    }
  });

  // Load Samples Button
  loadSamplesBtn.addEventListener("click", async () => {
    loadSamplesBtn.disabled = true;
    showAlert("Loading and indexing sample campus FAQ documents...", "info");

    try {
      const res = await fetch("/api/admin/load-samples", { method: "POST" });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Failed to load sample documents.");
      }

      showAlert(data.message, "success");
      setTimeout(() => window.location.reload(), 1200);
    } catch (err) {
      showAlert(err.message, "error");
      loadSamplesBtn.disabled = false;
    }
  });

  // Document upload handler
  async function handleFileUpload(file) {
    const allowed = [".pdf", ".docx", ".txt"];
    const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
    if (!allowed.includes(ext)) {
      showAlert(`Unsupported file type '${ext}'. Please choose a .pdf, .docx, or .txt file.`, "error");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    progressContainer.classList.remove("hidden");
    progressBar.style.width = "40%";
    statusText.textContent = `Extracting and embedding '${file.name}' with FastEmbed...`;

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || `Upload failed (${res.status})`);
      }

      progressBar.style.width = "100%";
      statusText.textContent = "Completed!";
      showAlert(`Indexed '${data.filename}' successfully into ${data.chunks_indexed} chunks.`, "success");

      setTimeout(() => window.location.reload(), 1000);
    } catch (err) {
      showAlert(err.message, "error");
      progressContainer.classList.add("hidden");
      progressBar.style.width = "0%";
    } finally {
      fileInput.value = "";
    }
  }

  // Delegate clicks for View Chunks and Delete buttons
  document.addEventListener("click", async (e) => {
    // Delete Document
    if (e.target.closest(".delete-doc-btn")) {
      const btn = e.target.closest(".delete-doc-btn");
      const docId = btn.getAttribute("data-doc-id");
      const filename = btn.getAttribute("data-filename");

      if (!confirm(`Delete '${filename}' and all of its indexed chunks from the database?`)) {
        return;
      }

      btn.disabled = true;
      try {
        const res = await fetch(`/api/documents/${docId}`, { method: "DELETE" });
        const data = await res.json();

        if (!res.ok) {
          throw new Error(data.detail || "Deletion failed.");
        }

        showAlert(data.message, "success");
        const row = document.getElementById(`doc-row-${docId}`);
        if (row) row.remove();
        updateStatsAfterDelete();
      } catch (err) {
        showAlert(err.message, "error");
        btn.disabled = false;
      }
    }

    // View Chunks
    if (e.target.closest(".view-chunks-btn")) {
      const btn = e.target.closest(".view-chunks-btn");
      const docId = btn.getAttribute("data-doc-id");
      const filename = btn.getAttribute("data-filename");
      openChunksModal(docId, filename);
    }
  });

  async function openChunksModal(docId, filename) {
    modalTitle.textContent = filename;
    modalSubtitle.textContent = "Loading chunks...";
    modalBody.innerHTML = "<p>Loading document chunks from SQLite...</p>";
    chunksModal.classList.remove("hidden");

    try {
      const res = await fetch(`/api/documents/${docId}/chunks`);
      const data = await res.json();

      if (!res.ok) throw new Error("Could not load chunks.");

      modalSubtitle.textContent = `Total chunks: ${data.chunks.length} | Format: ${data.document.file_type}`;
      modalBody.innerHTML = data.chunks
        .map(
          (c) => `
        <div class="chunk-card">
          <div class="chunk-meta">
            <span>Chunk #${c.chunk_index + 1}</span>
            <span>Page ${c.page_number} • Section: ${escapeHtml(c.section_heading)}</span>
          </div>
          <div class="chunk-body">${escapeHtml(c.content)}</div>
        </div>
      `
        )
        .join("");
    } catch (err) {
      modalBody.innerHTML = `<p style="color:var(--danger)">Failed to load chunks: ${escapeHtml(err.message)}</p>`;
    }
  }

  modalCloseBtn.addEventListener("click", () => chunksModal.classList.add("hidden"));
  chunksModal.addEventListener("click", (e) => {
    if (e.target === chunksModal) chunksModal.classList.add("hidden");
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !chunksModal.classList.contains("hidden")) {
      chunksModal.classList.add("hidden");
    }
  });

  function showAlert(message, type = "info") {
    alertBox.textContent = message;
    alertBox.className = `admin-alert ${type}`;
    alertBox.classList.remove("hidden");
    if (type !== "info") {
      setTimeout(() => alertBox.classList.add("hidden"), 5000);
    }
  }

  function updateStatsAfterDelete() {
    const totalDocsEl = document.getElementById("stat-total-docs");
    const docsCountEl = document.getElementById("docs-count");
    if (totalDocsEl) {
      const current = parseInt(totalDocsEl.textContent, 10) || 1;
      totalDocsEl.textContent = Math.max(0, current - 1);
    }
    if (docsCountEl) {
      const current = parseInt(docsCountEl.textContent, 10) || 1;
      docsCountEl.textContent = Math.max(0, current - 1);
    }
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
