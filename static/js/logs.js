const container = document.getElementById('logs-container');

async function loadLogs() {
    try {
        const response = await fetch('/api/logs');
        if (!response.ok) {
            container.textContent = 'Unable to load logs';
            return;
        }
        const payload = await response.json();
        if (!payload.logs || !payload.logs.length) {
            container.textContent = 'No log entries yet';
            return;
        }
        container.innerHTML = '<pre class="whitespace-pre-wrap text-xs">' + payload.logs.join('') + '</pre>';
    } catch (error) {
        container.textContent = 'Failed to load logs: ' + error.message;
    }
}

loadLogs();
setInterval(loadLogs, 5000);
