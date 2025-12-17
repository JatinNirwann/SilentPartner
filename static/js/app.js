const fileInput = document.getElementById('file-input');
const uploadBtn = document.getElementById('upload-btn');
const sendBtn = document.getElementById('send-btn');
const userInput = document.getElementById('user-input');
const chatHistory = document.getElementById('chat-history');
const docListContainer = document.getElementById('doc-list-container');
const clearHistoryBtn = document.getElementById('clear-history-btn');

// Auto-resize textarea
userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Upload functionality
uploadBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async (e) => {
    if (e.target.files.length > 0) {
        const file = e.target.files[0];
        const formData = new FormData();
        formData.append('file', file);

        addSystemMessage(`Uploading ${file.name}...`);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (response.ok) {
                addSystemMessage(`Successfully processed ${data.filename}!`);
                loadDocuments();
            } else {
                addSystemMessage(`Error: ${data.error}`);
            }
        } catch (error) {
            addSystemMessage(`Upload failed: ${error.message}`);
        }
        
        fileInput.value = ''; // Reset
    }
});

// Chat functionality
async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // Add user message
    addUserMessage(text);
    userInput.value = '';
    userInput.style.height = 'auto';

    // Loading indicator
    const loadingId = addLoadingIndicator();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: text })
        });
        const data = await response.json();
        
        removeMessage(loadingId);
        
        if (data.response) {
            addSystemMessage(data.response);
        } else {
            addSystemMessage("Error: No response received.");
        }
    } catch (error) {
        removeMessage(loadingId);
        addSystemMessage(`Error: ${error.message}`);
    }
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// UI Helper Functions
function addUserMessage(text) {
    const div = document.createElement('div');
    div.className = 'message user';
    div.innerHTML = `
        <div class="content"><p>${text}</p></div>
        <div class="avatar"><i data-lucide="user"></i></div>
    `;
    chatHistory.appendChild(div);
    scrollToBottom();
    lucide.createIcons();
}

function addSystemMessage(markdownText) {
    const div = document.createElement('div');
    div.className = 'message system';
    const htmlContent = marked.parse(markdownText);
    div.innerHTML = `
        <div class="avatar"><i data-lucide="bot"></i></div>
        <div class="content">${htmlContent}</div>
    `;
    chatHistory.appendChild(div);
    scrollToBottom();
    lucide.createIcons();
}

function addLoadingIndicator() {
    const id = 'loading-' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'message system';
    div.innerHTML = `
        <div class="avatar"><i data-lucide="loader-2" class="animate-spin"></i></div>
        <div class="content"><p>Thinking...</p></div>
    `;
    chatHistory.appendChild(div);
    scrollToBottom();
    lucide.createIcons();
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollToBottom() {
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// Document Management
async function loadDocuments() {
    try {
        const response = await fetch('/documents');
        const data = await response.json();
        docListContainer.innerHTML = '';
        
        if (data.documents && data.documents.length > 0) {
            data.documents.forEach(doc => {
                const item = document.createElement('div');
                item.className = 'doc-item';
                item.innerHTML = `
                    <i data-lucide="file-text"></i>
                    <span>${doc.filename}</span>
                    <i data-lucide="x" class="delete-doc" onclick="deleteDoc(${doc.id}, event)"></i>
                `;
                docListContainer.appendChild(item);
            });
        } else {
            docListContainer.innerHTML = '<p style="padding:1rem; color:var(--text-secondary); font-size:0.9rem">No documents uploaded</p>';
        }
        lucide.createIcons();
    } catch (e) {
        console.error("Failed to load docs", e);
    }
}

window.deleteDoc = async (id, event) => {
    event.stopPropagation();
    if (confirm("Delete this document?")) {
        await fetch(`/documents/${id}`, { method: 'DELETE' });
        loadDocuments();
    }
}

// Initial Load
loadDocuments();

// History Management
async function loadHistory() {
    try {
        const response = await fetch('/chats');
        const data = await response.json();
        if (data.chats) {
            data.chats.forEach(chat => {
                addUserMessage(chat.query);
                addSystemMessage(chat.response);
            });
        }
    } catch (e) {}
}

clearHistoryBtn.addEventListener('click', async () => {
    if (confirm("Clear chat history?")) {
        await fetch('/chats', { method: 'DELETE' });
        // Clear UI except welcome message
        chatHistory.innerHTML = `
            <div class="message system">
                <div class="avatar">
                    <i data-lucide="bot"></i>
                </div>
                <div class="content">
                    <p>Welcome to Nexus. Upload a document to get started or ask me anything based on your knowledge base.</p>
                </div>
            </div>
        `;
        lucide.createIcons();
    }
});
