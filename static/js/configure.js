const state = {
    providers: [],
    apiKeys: []
};

const providerListEl = document.getElementById('provider-list');
const reloadButton = document.getElementById('reload-config');
const reloadStatus = document.getElementById('reload-status');
const keysForm = document.getElementById('keys-form');
const keysStatus = document.getElementById('keys-status');
const keysGrid = document.getElementById('keys-grid');
const introspectForm = document.getElementById('provider-introspect-form');
const introspectStatus = document.getElementById('introspect-status');
const providerForm = document.getElementById('provider-form');
const providerStatus = document.getElementById('provider-form-status');
const providerNotes = document.getElementById('provider-notes');
const variableFields = document.getElementById('variable-fields');
const addVariableButton = document.getElementById('add-variable');
const cancelBuilderButton = document.getElementById('cancel-builder');
const requestTemplate = document.getElementById('request-template');
const builderBaseUrl = document.getElementById('builder-base-url');
const providerNameInput = document.getElementById('provider-name');
const providerDisplayNameInput = document.getElementById('provider-display-name');
const providerDescriptionInput = document.getElementById('provider-description');
const providerTimeoutInput = document.getElementById('provider-timeout');
const providerRequiresKeyInput = document.getElementById('provider-requires-key');
const detectUrlInput = document.getElementById('introspect-url');

let detectedSchema = null;

function setStatus(el, message, tone = 'info') {
    if (!el) return;
    const base = tone === 'error' ? 'text-sm text-red-300' : tone === 'success' ? 'text-sm text-emerald-300' : 'text-sm text-slate-200';
    el.textContent = message;
    el.className = base;
}

async function fetchConfig() {
    try {
        const response = await fetch('/api/config');
        if (!response.ok) {
            throw new Error('Unable to load configuration');
        }
        const payload = await response.json();
        state.providers = payload.providers || [];
        state.apiKeys = payload.api_keys || [];
        renderProviders();
        renderKeyInputs();
    } catch (error) {
        setStatus(reloadStatus, error.message, 'error');
    }
}

function renderProviders() {
    if (!providerListEl) return;
    providerListEl.innerHTML = '';
    if (!state.providers.length) {
        providerListEl.innerHTML = '<p class="text-sm text-slate-300">No providers saved yet. Use the form on the left to add one.</p>';
        return;
    }
    state.providers.forEach((provider) => {
        const card = document.createElement('div');
        card.className = 'border border-slate-700 rounded-xl p-4 bg-slate-950/60';
        const statusBadge = provider.mock ? '<span class="text-xs text-amber-300 ml-2">mock</span>' : '';
        card.innerHTML = `
            <div class="flex items-center justify-between">
                <div>
                    <p class="text-lg font-semibold text-slate-100">${provider.display_name || provider.name}${statusBadge}</p>
                    <p class="text-xs text-slate-400">${provider.base_url || ''}</p>
                </div>
                <button class="btn-secondary text-xs" data-remove-provider="${provider.name}">Remove</button>
            </div>
            <div class="text-xs text-slate-400 mt-2">
                <p>Timeout: ${provider.timeout || 30}s · Requires key: ${provider.requires_api_key !== false ? 'yes' : 'no'}</p>
                <p>${provider.description || ''}</p>
            </div>
        `;
        providerListEl.appendChild(card);
    });
}

function renderKeyInputs() {
    if (!keysGrid) return;
    keysGrid.innerHTML = '';
    if (!state.providers.length) {
        keysGrid.innerHTML = '<p class="text-sm text-slate-300">Add a provider to begin storing keys.</p>';
        return;
    }
    state.providers.forEach((provider) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'space-y-1';
        const configured = state.apiKeys.includes(provider.name);
        const badge = configured ? '<span class="text-emerald-300 text-xs">Saved</span>' : '<span class="text-xs text-slate-400">Not saved</span>';
        wrapper.innerHTML = `
            <div class="flex items-center justify-between">
                <span class="text-xs font-semibold tracking-widest text-slate-200">${provider.display_name || provider.name}</span>
                <div class="flex items-center gap-2">
                    ${badge}
                    ${configured ? `<button type="button" class="btn-secondary text-xs" data-clear-key="${provider.name}">Clear key</button>` : ''}
                </div>
            </div>
            <input type="password" class="key-input" data-provider="${provider.name}" placeholder="${configured ? 'Enter to replace saved key' : 'API key'}" />
        `;
        keysGrid.appendChild(wrapper);
    });
}

keysForm?.addEventListener('submit', async (event) => {
    event.preventDefault();
    const inputs = keysGrid.querySelectorAll('input[data-provider]');
    const keys = {};
    inputs.forEach((input) => {
        const value = input.value.trim();
        if (!value) return;
        keys[input.dataset.provider] = value;
    });
    if (!Object.keys(keys).length) {
        setStatus(keysStatus, 'Enter a key to save or use the Clear key buttons.', 'info');
        return;
    }
    setStatus(keysStatus, 'Saving keys…');
    try {
        const response = await fetch('/api/keys', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ keys })
        });
        if (!response.ok) {
            const detail = await response.json();
            throw new Error(detail.detail || 'Unable to save keys');
        }
        const payload = await response.json();
        state.apiKeys = payload.configured || [];
        renderKeyInputs();
        setStatus(keysStatus, 'Keys saved successfully', 'success');
    } catch (error) {
        setStatus(keysStatus, error.message, 'error');
    }
});

keysGrid?.addEventListener('click', async (event) => {
    const button = event.target.closest('[data-clear-key]');
    if (!button) return;
    const provider = button.dataset.clearKey;
    if (!provider) return;
    if (!confirm(`Remove the saved key for ${provider}?`)) {
        return;
    }
    button.disabled = true;
    setStatus(keysStatus, `Clearing key for ${provider}…`);
    try {
        const response = await fetch('/api/keys', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ keys: { [provider]: null } })
        });
        if (!response.ok) {
            const detail = await response.json();
            throw new Error(detail.detail || 'Unable to clear key');
        }
        const payload = await response.json();
        state.apiKeys = payload.configured || [];
        renderKeyInputs();
        setStatus(keysStatus, `Key removed for ${provider}`, 'success');
    } catch (error) {
        setStatus(keysStatus, error.message, 'error');
    } finally {
        button.disabled = false;
    }
});

reloadButton?.addEventListener('click', async () => {
    setStatus(reloadStatus, 'Reloading…');
    try {
        const response = await fetch('/api/reload-config', { method: 'POST' });
        if (!response.ok) {
            throw new Error('Unable to reload file');
        }
        const payload = await response.json();
        setStatus(reloadStatus, `Providers loaded: ${payload.count}`, 'success');
        await fetchConfig();
    } catch (error) {
        setStatus(reloadStatus, error.message, 'error');
    }
});

providerListEl?.addEventListener('click', async (event) => {
    const button = event.target.closest('[data-remove-provider]');
    if (!button) return;
    const name = button.dataset.removeProvider;
    if (!confirm(`Remove provider ${name}?`)) return;
    button.disabled = true;
    try {
        const response = await fetch(`/api/providers/${encodeURIComponent(name)}`, { method: 'DELETE' });
        if (!response.ok) {
            const detail = await response.json();
            throw new Error(detail.detail || 'Failed to delete provider');
        }
        await fetchConfig();
    } catch (error) {
        alert(error.message);
    } finally {
        button.disabled = false;
    }
});

function clearBuilder() {
    detectedSchema = null;
    providerForm?.classList.add('hidden');
    providerForm?.reset();
    if (variableFields) {
        variableFields.innerHTML = '';
    }
    providerNotes.textContent = '';
    requestTemplate.value = '';
    builderBaseUrl.value = '';
}

cancelBuilderButton?.addEventListener('click', () => {
    clearBuilder();
});

function renderVariableDefinitions(schema) {
    if (!variableFields) return;
    variableFields.innerHTML = '';
    const definitions = schema.variable_schema && schema.variable_schema.length
        ? schema.variable_schema
        : Object.keys(schema.variables || {}).map((key) => ({ key, label: key, type: 'text', required: false }));

    definitions.forEach((definition) => addVariableRow(definition.key, schema.variables?.[definition.key] ?? '', definition));
}

function addVariableRow(key = '', value = '', definition = {}) {
    if (!variableFields) return;
    const wrapper = document.createElement('div');
    wrapper.className = 'space-y-1';
    const isCustom = !definition || !definition.key;
    const label = definition.label || key || 'Custom variable';
    if (isCustom) {
        wrapper.innerHTML = `
            <label class="text-xs text-slate-300">${label}</label>
            <div class="grid gap-2 md:grid-cols-2">
                <input type="text" class="key-input" placeholder="Variable name" data-variable-name value="${key}" />
                <input type="text" class="key-input" placeholder="Value" data-variable-value value="${value}" />
            </div>
        `;
    } else {
        const inputType = definition.type === 'textarea' ? 'textarea' : 'input';
        const element = document.createElement(inputType);
        element.className = 'key-input';
        element.dataset.variableKey = definition.key;
        if (definition.placeholder) element.placeholder = definition.placeholder;
        if (definition.required) element.required = true;
        element.value = value ?? '';
        if (inputType === 'input' && definition.type === 'number') {
            element.type = 'number';
        }
        if (inputType === 'textarea') {
            element.rows = 3;
        }
        wrapper.innerHTML = `<label class="text-xs text-slate-300">${label}</label>`;
        wrapper.appendChild(element);
    }
    variableFields.appendChild(wrapper);
}

addVariableButton?.addEventListener('click', () => {
    if (providerForm?.classList.contains('hidden')) return;
    addVariableRow('', '', {});
});

introspectForm?.addEventListener('submit', async (event) => {
    event.preventDefault();
    const baseUrl = detectUrlInput.value.trim();
    if (!baseUrl) {
        setStatus(introspectStatus, 'Enter a valid URL', 'error');
        return;
    }
    setStatus(introspectStatus, 'Contacting endpoint…');
    try {
        const response = await fetch('/api/providers/introspect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ base_url: baseUrl })
        });
        if (!response.ok) {
            const detail = await response.json();
            throw new Error(detail.detail || 'Unable to introspect provider');
        }
        detectedSchema = await response.json();
        populateBuilder(detectedSchema);
        setStatus(introspectStatus, 'Template ready. Complete the form below.', 'success');
    } catch (error) {
        setStatus(introspectStatus, error.message, 'error');
    }
});

function populateBuilder(schema) {
    if (!providerForm) return;
    providerForm.classList.remove('hidden');
    setStatus(providerStatus, '');
    builderBaseUrl.value = schema.base_url || detectUrlInput.value.trim();
    providerNameInput.value = schema.suggested_name || '';
    providerDisplayNameInput.value = schema.suggested_display_name || '';
    providerDescriptionInput.value = schema.description || '';
    providerTimeoutInput.value = schema.timeout || 60;
    providerRequiresKeyInput.checked = schema.requires_api_key !== false;
    providerNotes.innerHTML = `<p class="text-sm text-slate-300">${schema.notes || ''}</p>`;
    requestTemplate.value = JSON.stringify(schema.request || {}, null, 2);
    renderVariableDefinitions(schema);
}

providerForm?.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!detectedSchema) {
        setStatus(providerStatus, 'Run detection first.', 'error');
        return;
    }
    let requestBlock;
    try {
        requestBlock = JSON.parse(requestTemplate.value || '{}');
    } catch (error) {
        setStatus(providerStatus, 'Request template must be valid JSON.', 'error');
        return;
    }
    const variables = {};
    if (variableFields) {
        variableFields.querySelectorAll('[data-variable-key]').forEach((input) => {
            if (!input.dataset.variableKey) return;
            const value = input.value.trim();
            if (value === '') {
                variables[input.dataset.variableKey] = '';
                return;
            }
            const rawValue = input.type === 'number' ? Number(value) : value;
            variables[input.dataset.variableKey] = rawValue;
        });
        const customNames = variableFields.querySelectorAll('[data-variable-name]');
        customNames.forEach((nameInput) => {
            const key = nameInput.value.trim();
            const valueInput = nameInput.parentElement.querySelector('[data-variable-value]');
            const value = valueInput ? valueInput.value.trim() : '';
            if (key) {
                variables[key] = value;
            }
        });
    }
    const payload = {
        name: providerNameInput.value.trim(),
        display_name: providerDisplayNameInput.value.trim(),
        base_url: builderBaseUrl.value.trim(),
        description: providerDescriptionInput.value.trim(),
        timeout: Number(providerTimeoutInput.value) || 60,
        requires_api_key: providerRequiresKeyInput.checked,
        request: requestBlock,
        variables,
        mock: false
    };
    if (!payload.name || !payload.base_url) {
        setStatus(providerStatus, 'Name and base URL are required.', 'error');
        return;
    }
    setStatus(providerStatus, 'Saving provider…');
    try {
        const response = await fetch('/api/providers', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            const detail = await response.json();
            throw new Error(detail.detail || 'Unable to save provider');
        }
        clearBuilder();
        detectUrlInput.value = '';
        await fetchConfig();
        setStatus(introspectStatus, 'Provider saved. Enter another URL to continue.', 'success');
    } catch (error) {
        setStatus(providerStatus, error.message, 'error');
    }
});

fetchConfig();
