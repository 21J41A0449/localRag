/**
 * DocuMind AI - Agentic RAG Frontend
 * Features: Advanced parsing status, reasoning trace display, query complexity
 */

// DOM Elements
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');
const documentsList = document.getElementById('documentsList');
const docCount = document.getElementById('docCount');
const statusCard = document.getElementById('statusCard');
const statusText = document.getElementById('statusText');
const messagesContainer = document.getElementById('messagesContainer');
const queryForm = document.getElementById('queryForm');
const questionInput = document.getElementById('questionInput');
const submitBtn = document.getElementById('submitBtn');

// State
let isProcessing = false;
let systemCapabilities = {
    advanced_parsing: false,
    agentic_mode: false
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    loadDocuments();
    setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        handleFiles(e.dataTransfer.files);
    });

    queryForm.addEventListener('submit', handleQuerySubmit);
}

// Health Check
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();

        systemCapabilities.advanced_parsing = data.advanced_parsing;
        systemCapabilities.agentic_mode = data.agentic_mode;

        if (data.ollama_connected) {
            statusCard.className = 'status-card connected';
            let statusLabel = 'Ollama Connected';
            if (data.agentic_mode) statusLabel += ' • Agentic';
            if (data.advanced_parsing) statusLabel += ' • Advanced';
            statusText.textContent = statusLabel;
        } else {
            statusCard.className = 'status-card disconnected';
            statusText.textContent = 'Ollama Disconnected';
        }
    } catch (error) {
        statusCard.className = 'status-card disconnected';
        statusText.textContent = 'Connection Error';
        console.error('Health check failed:', error);
    }
}

// File Upload
function handleFileSelect(e) {
    handleFiles(e.target.files);
}

async function handleFiles(files) {
    const pdfFiles = Array.from(files).filter(f => f.name.toLowerCase().endsWith('.pdf'));

    if (pdfFiles.length === 0) {
        showUploadMessage('Please select PDF files only', 'error');
        return;
    }

    const formData = new FormData();
    pdfFiles.forEach(file => formData.append('files', file));

    const parsingType = systemCapabilities.advanced_parsing ? 'advanced' : 'basic';
    showUploadMessage(`Uploading ${pdfFiles.length} file(s) with ${parsingType} parsing...`, 'info');

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            let msg = `✓ Uploaded ${data.files_uploaded.length} file(s). Processing ${data.total_chunks} chunks`;
            if (data.metadata && data.metadata.length > 0) {
                const meta = data.metadata[0];
                if (meta.has_tables) msg += ' • Tables detected';
                if (meta.has_images) msg += ' • Images detected';
            }
            showUploadMessage(msg, 'success');
            setTimeout(loadDocuments, 2000);
            setTimeout(() => { uploadProgress.innerHTML = ''; }, 5000);
        } else {
            showUploadMessage(`Error: ${data.detail}`, 'error');
        }
    } catch (error) {
        showUploadMessage(`Upload failed: ${error.message}`, 'error');
        console.error('Upload error:', error);
    }

    fileInput.value = '';
}

function showUploadMessage(message, type) {
    uploadProgress.innerHTML = `<div class="upload-progress-item ${type}">${message}</div>`;
}

// Load Documents
async function loadDocuments() {
    try {
        const response = await fetch('/documents');
        const data = await response.json();

        docCount.textContent = data.documents.length;

        if (data.documents.length === 0) {
            documentsList.innerHTML = `
                <div class="empty-state">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>
                        <polyline points="13 2 13 9 20 9"/>
                    </svg>
                    <p>No documents yet</p>
                    <span>Upload PDFs to get started</span>
                </div>
            `;
            return;
        }

        documentsList.innerHTML = data.documents.map(doc => {
            let badges = '';
            if (doc.has_tables) badges += '<span class="doc-badge">📊</span>';
            if (doc.has_images) badges += '<span class="doc-badge">🖼️</span>';
            if (doc.pages) badges += `<span class="doc-badge">${doc.pages}p</span>`;

            return `
                <div class="document-item">
                    <div class="document-icon">PDF</div>
                    <div class="document-info">
                        <div class="document-name" title="${doc.filename}">${doc.filename}</div>
                        <div class="document-meta">
                            <span class="document-size">${formatFileSize(doc.size_bytes)}</span>
                            ${badges}
                        </div>
                    </div>
                    <button class="document-delete" onclick="deleteDocument('${doc.filename}')" title="Delete">
                        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="3 6 5 6 21 6"/>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                        </svg>
                    </button>
                </div>
            `;
        }).join('');
    } catch (error) {
        console.error('Failed to load documents:', error);
    }
}

// Delete Document
async function deleteDocument(filename) {
    if (!confirm(`Delete "${filename}"?`)) return;

    try {
        const response = await fetch(`/documents/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            loadDocuments();
        } else {
            const data = await response.json();
            alert(`Error: ${data.detail}`);
        }
    } catch (error) {
        console.error('Delete error:', error);
        alert('Failed to delete document');
    }
}

// Query Handling
async function handleQuerySubmit(e) {
    e.preventDefault();

    const question = questionInput.value.trim();
    if (!question || isProcessing) return;

    isProcessing = true;
    setLoading(true);

    // Remove welcome card
    const welcomeCard = messagesContainer.querySelector('.welcome-card');
    if (welcomeCard) welcomeCard.remove();

    // Add user message
    addMessage(question, 'user');
    questionInput.value = '';

    // Add loading indicator
    const loadingId = addLoadingMessage();

    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question,
                include_reasoning: true
            })
        });

        const data = await response.json();

        removeLoadingMessage(loadingId);

        if (response.ok) {
            addMessage(data.answer, 'assistant', data.sources, data.complexity, data.sub_queries);
        } else {
            addMessage(`Error: ${data.detail}`, 'assistant');
        }
    } catch (error) {
        removeLoadingMessage(loadingId);
        addMessage(`Error: ${error.message}`, 'assistant');
        console.error('Query error:', error);
    }

    isProcessing = false;
    setLoading(false);
}

// Message Functions
function addMessage(text, role, sources = [], complexity = null, subQueries = []) {
    const avatar = role === 'user' ? '👤' : '🤖';

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        const tags = sources.map(s =>
            `<span class="source-tag">${s.file} • Page ${s.page}</span>`
        ).join('');
        sourcesHtml = `<div class="message-sources"><span class="sources-label">Sources</span>${tags}</div>`;
    }

    let metaHtml = '';
    if (role === 'assistant') {
        let metaParts = [];
        if (complexity) metaParts.push(`<span class="complexity-badge ${complexity}">${complexity}</span>`);
        if (subQueries && subQueries.length > 0) {
            metaParts.push(`<span class="subquery-count">${subQueries.length} sub-queries</span>`);
        }
        if (metaParts.length > 0) {
            metaHtml = `<div class="message-meta">${metaParts.join('')}</div>`;
        }
    }

    const html = `
        <div class="message ${role}">
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                ${metaHtml}
                <div class="message-bubble">
                    <div class="message-text">${escapeHtml(text)}</div>
                    ${sourcesHtml}
                </div>
            </div>
        </div>
    `;

    messagesContainer.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
}

function addLoadingMessage() {
    const id = 'loading-' + Date.now();
    const agenticMsg = systemCapabilities.agentic_mode ? 'Analyzing & reasoning...' : 'Thinking...';
    const html = `
        <div class="message assistant" id="${id}">
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <div class="message-loading">
                    <div class="loading-dots">
                        <span></span><span></span><span></span>
                    </div>
                    <span class="loading-text">${agenticMsg}</span>
                </div>
            </div>
        </div>
    `;
    messagesContainer.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
    return id;
}

function removeLoadingMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// UI Helpers
function setLoading(loading) {
    submitBtn.disabled = loading;
    submitBtn.classList.toggle('loading', loading);
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Periodic health check
setInterval(checkHealth, 30000);
