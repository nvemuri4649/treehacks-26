/**
 * Privacy Shield — Chatbot Frontend
 *
 * Connects to the FastAPI WebSocket backend, sends user messages
 * (text + optional image), and displays responses with status indicators.
 */

(() => {
    "use strict";

    // --- DOM elements ---
    const chatMessages  = document.getElementById("chat-messages");
    const messageInput  = document.getElementById("message-input");
    const sendBtn       = document.getElementById("send-btn");
    const modelSelect   = document.getElementById("model-select");
    const newSessionBtn = document.getElementById("new-session-btn");
    const imageInput    = document.getElementById("image-input");
    const imagePreview  = document.getElementById("image-preview");
    const previewImg    = document.getElementById("preview-img");
    const removeImageBtn= document.getElementById("remove-image");
    const statusBar     = document.getElementById("status-bar");
    const statusText    = document.getElementById("status-text");

    // --- State ---
    let ws = null;
    let sessionId = null;
    let pendingImage = null;   // { base64, mimeType, dataUrl }
    let isProcessing = false;

    // --- Session management ---

    async function createSession() {
        const resp = await fetch("/api/session", { method: "POST" });
        const data = await resp.json();
        return data.session_id;
    }

    async function initSession() {
        // Close existing WebSocket
        if (ws) {
            ws.close();
            ws = null;
        }

        sessionId = await createSession();
        connectWebSocket();
        clearChat();
    }

    // --- WebSocket ---

    function connectWebSocket() {
        const proto = location.protocol === "https:" ? "wss:" : "ws:";
        ws = new WebSocket(`${proto}//${location.host}/ws/${sessionId}`);

        ws.onopen = () => {
            console.log("[WS] Connected, session:", sessionId);
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            handleServerMessage(msg);
        };

        ws.onclose = () => {
            console.log("[WS] Disconnected");
        };

        ws.onerror = (err) => {
            console.error("[WS] Error:", err);
            showError("Connection error. Please refresh the page.");
        };
    }

    function handleServerMessage(msg) {
        switch (msg.type) {
            case "status":
                showStatus(msg.stage);
                break;

            case "response":
                hideStatus();
                addMessage("assistant", msg.text, msg.model);
                setProcessing(false);
                break;

            case "error":
                hideStatus();
                showError(msg.message);
                setProcessing(false);
                break;

            default:
                console.warn("[WS] Unknown message type:", msg.type);
        }
    }

    // --- Status indicator ---

    const stageLabels = {
        sanitizing: "Dereferencing personal information locally...",
        glazing:    "Glazing uploaded image with adversarial protection...",
        thinking:   "Cloud model is thinking (sanitized data only)...",
        restoring:  "Re-referencing your information locally...",
    };

    function showStatus(stage) {
        statusBar.classList.remove("hidden");
        statusBar.dataset.stage = stage;
        statusText.textContent = stageLabels[stage] || stage;
    }

    function hideStatus() {
        statusBar.classList.add("hidden");
    }

    // --- Chat UI ---

    function clearChat() {
        chatMessages.innerHTML = `
            <div class="welcome-message">
                <svg class="welcome-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                </svg>
                <h2>Your conversations, protected.</h2>
                <p>Personal information is dereferenced and images are glazed locally before reaching any cloud model.
                   Your data never leaves your machine unprotected.</p>
                <div class="welcome-features">
                    <div class="feature-pill">PII Dereferencing</div>
                    <div class="feature-pill">Image Glazing</div>
                    <div class="feature-pill">Local Processing</div>
                </div>
            </div>
        `;
    }

    function removeWelcome() {
        const welcome = chatMessages.querySelector(".welcome-message");
        if (welcome) welcome.remove();
    }

    function addMessage(role, text, model) {
        removeWelcome();

        const wrapper = document.createElement("div");
        wrapper.className = `message ${role}`;

        // If user message has an image, show it
        if (role === "user" && pendingImage) {
            const img = document.createElement("img");
            img.className = "message-image";
            img.src = pendingImage.dataUrl;
            wrapper.appendChild(img);
        }

        const bubble = document.createElement("div");
        bubble.className = "message-bubble";
        bubble.textContent = text;
        wrapper.appendChild(bubble);

        if (model) {
            const meta = document.createElement("div");
            meta.className = "message-meta";
            meta.textContent = model;
            wrapper.appendChild(meta);
        }

        chatMessages.appendChild(wrapper);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Clear pending image after user message is rendered
        if (role === "user") {
            clearImagePreview();
        }
    }

    function showError(text) {
        removeWelcome();

        const wrapper = document.createElement("div");
        wrapper.className = "message error assistant";

        const bubble = document.createElement("div");
        bubble.className = "message-bubble";
        bubble.textContent = text;
        wrapper.appendChild(bubble);

        chatMessages.appendChild(wrapper);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // --- Image handling ---

    function handleImageSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            const dataUrl = e.target.result;
            const base64 = dataUrl.split(",")[1];
            const mimeType = file.type || "image/png";

            pendingImage = { base64, mimeType, dataUrl };

            previewImg.src = dataUrl;
            imagePreview.classList.remove("hidden");
        };
        reader.readAsDataURL(file);
    }

    function clearImagePreview() {
        pendingImage = null;
        imagePreview.classList.add("hidden");
        previewImg.src = "";
        imageInput.value = "";
    }

    // --- Send message ---

    function sendMessage() {
        const text = messageInput.value.trim();
        if (!text || isProcessing || !ws || ws.readyState !== WebSocket.OPEN) return;

        const payload = {
            text: text,
            model: modelSelect.value,
        };

        if (pendingImage) {
            payload.image = pendingImage.base64;
            payload.mime_type = pendingImage.mimeType;
        }

        // Show user message in UI
        addMessage("user", text);

        ws.send(JSON.stringify(payload));
        messageInput.value = "";
        autoResizeTextarea();
        setProcessing(true);
    }

    function setProcessing(active) {
        isProcessing = active;
        sendBtn.disabled = active;
        messageInput.disabled = active;
    }

    // --- Auto-resize textarea ---

    function autoResizeTextarea() {
        messageInput.style.height = "auto";
        messageInput.style.height = Math.min(messageInput.scrollHeight, 140) + "px";
    }

    // --- Event listeners ---

    sendBtn.addEventListener("click", sendMessage);

    messageInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    messageInput.addEventListener("input", autoResizeTextarea);

    imageInput.addEventListener("change", handleImageSelect);
    removeImageBtn.addEventListener("click", clearImagePreview);

    newSessionBtn.addEventListener("click", () => {
        initSession();
    });

    // --- Boot ---
    initSession();
})();
