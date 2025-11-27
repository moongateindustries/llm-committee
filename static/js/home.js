const form = document.getElementById('prompt-form');
const promptInput = document.getElementById('prompt-input');
const responsesContainer = document.getElementById('responses');
const errorEl = document.getElementById('prompt-error');
const providers = window.__LLM_PROVIDERS__ || [];

function renderLoading() {
    responsesContainer.innerHTML = '';
    providers.forEach((provider) => {
        const panel = document.createElement('div');
        panel.className = 'panel animate-pulse';
        panel.innerHTML = `
            <div class="flex justify-between items-center mb-2">
                <span class="text-slate-200 font-semibold">${provider.display_name || provider.name}</span>
                <span class="text-xs text-slate-400">waiting…</span>
            </div>
            <p class="text-slate-500">Awaiting response...</p>
        `;
        responsesContainer.appendChild(panel);
    });
}

function renderResponses(payload) {
    responsesContainer.innerHTML = '';
    Object.entries(payload.responses || {}).forEach(([name, info]) => {
        const panel = document.createElement('div');
        panel.className = 'panel space-y-2';
        const pretty = typeof info.response === 'object' ? JSON.stringify(info.response, null, 2) : info.response;
        panel.innerHTML = `
            <div class="flex justify-between items-center">
                <span class="text-slate-100 font-semibold">${name}</span>
                <span class="text-xs uppercase tracking-widest ${info.status === 'ok' ? 'text-emerald-300' : info.status === 'timeout' ? 'text-amber-300' : 'text-red-300'}">${info.status}</span>
            </div>
            <pre class="text-xs whitespace-pre-wrap">${pretty}</pre>
        `;
        responsesContainer.appendChild(panel);
    });
}

function showError(message) {
    errorEl.textContent = message;
    errorEl.classList.remove('hidden');
}

function clearError() {
    errorEl.textContent = '';
    errorEl.classList.add('hidden');
}

form?.addEventListener('submit', async (event) => {
    event.preventDefault();
    clearError();
    const prompt = promptInput.value.trim();
    if (!prompt) {
        showError('Prompt cannot be empty.');
        return;
    }
    renderLoading();
    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt })
        });
        if (!response.ok) {
            const detail = await response.json();
            showError(detail.detail || 'Unable to reach backend');
            return;
        }
        const payload = await response.json();
        renderResponses(payload);
    } catch (error) {
        showError('Request failed: ' + error.message);
    }
});
