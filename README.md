# LLM Committee

A container-first FastAPI control panel that fans your prompts out to any number of LLM providers. Provider definitions live in a JSON file that can be edited from the UI, API keys persist inside a writable volume, and a small Tailwind UI exposes "Home", "Configure APIs", and "Logs" views.

## Highlights
- FastAPI backend with HTTP Basic auth (credentials via env vars) and `/api/*` endpoints for queries, key updates, config reloads, and log streaming.
- Provider routing is completely data-driven: edit `config/llm_config.json` or add providers from the Configure view. Changes are stored in `/app/state/providers.json` automatically.
- Frontend served from FastAPI static files featuring a terminal-style prompt, API key manager, and live log watcher.
- Dockerfile + docker-compose stack ready for `docker build` or `docker push` to Docker Hub. Containers keep `./logs`, `./config`, and `./state` mounted so configs and keys persist.

## Prerequisites
- Docker 20.10+ and docker-compose v2 (or Compose plugin)
- Optional: Python 3.11+ if you want to run without Docker

## Quickstart (Docker Compose)
```bash
# Clone and move into the project
 git clone <repo-url>
 cd llm-committee

# Optional: adjust config/llm_config.json before first launch

# Set UI credentials + host port, then start the stack
 export BASIC_AUTH_USER=admin
 export BASIC_AUTH_PASS=supersecret
 export APP_PORT=8080
 docker-compose up --build
```
Browse to `http://localhost:8080` and authenticate. If you remove every entry from `config/llm_config.json` (or delete the file entirely), the Home route automatically redirects to **Configure APIs** so you must define providers before using the prompt console. The template ships with a mock OpenAI provider, so you'll land directly on Home until you edit the config.

### Running Locally without Docker
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
BASIC_AUTH_USER=dev BASIC_AUTH_PASS=dev uvicorn main:app --reload
```
Front-end assets are served from `/static` exactly like they are in the container.

## Configuring Providers
Providers are described in `config/llm_config.json` under the `providers` list or can be added dynamically from the UI. When the app starts it mirrors this file into `/app/state/providers.json` and reads from that writable copy going forward.

### Add providers from the UI
1. Open **Configure APIs** and enter an API endpoint URL.
2. Click **Detect provider**. The backend inspects the host (OpenAI, Anthropic, Cohere, Google Generative Language/Gemini, or generic) and returns a request template plus any variable fields it needs.
3. Fill in the suggested inputs (e.g., model names or timeouts), tweak the JSON request template if necessary, and save.

The saved provider is written to `/app/state/providers.json` immediately and takes effect without a restart. Templates support placeholders like `{{prompt}}`, `{{model}}`, and `{{api_key}}` (you can even place them directly in the base URL for providers such as Google Gemini); any custom variables you add become available to the JSON template at runtime.

### Editing providers manually
You can continue editing `config/llm_config.json` directly. Add one object per upstream model:
```json
{
  "name": "openai",
  "display_name": "OpenAI Production",
  "base_url": "https://api.openai.com/v1/chat/completions",
  "model": "gpt-4o-mini",
  "timeout": 60,
  "mock": false,
  "description": "Primary OpenAI tenant"
}
```
Field notes:
- `name`: unique identifier (used by `/api/keys`). Keep it short and lowercase.
- `display_name`: UI label.
- `base_url`: provider endpoint. Works for OpenAI-compatible APIs too.
- `model`: any model string the provider expects.
- `timeout`: seconds before cancelling a slow request.
- `mock`: set `true` to return deterministic mock data instead of calling a live API.

After editing the file you can either restart the container or click **Reload config file** on the Configure APIs page (which calls `POST /api/reload-config`). The file contents will overwrite `/app/state/providers.json`.

## Supplying API Keys
Keys are stored separately in `/app/state/api_keys.json`, so they survive container restarts as long as the volume is mounted. Use one of these methods:
1. **UI workflow**: visit **Configure APIs**, enter the key matching each provider `name`, and hit **Save Keys**. The backend writes them to the JSON file on disk and loads them into memory.
2. **HTTP request**: send a Basic-authenticated `POST /api/keys` call:
   ```bash
   curl -u "$BASIC_AUTH_USER:$BASIC_AUTH_PASS" \
        -H "Content-Type: application/json" \
        -d '{"keys": {"openai": "sk-...", "cohere": "cohere-..."}}' \
        http://localhost:8080/api/keys
   ```
Use the Configure page whenever you replace keys; the backend logs which provider names were updated but never prints secrets. The JSON file is part of the bind-mounted `./state` directory by default.

## Useful Commands
- `docker-compose logs -f` – tail container logs alongside the in-UI log stream.
- `docker build -t your-handle/llm-committee:latest .` – build an image for Docker Hub.
- `docker push your-handle/llm-committee:latest` – publish the built image.

## Security & Deployment Notes
- Set `BASIC_AUTH_USER` / `BASIC_AUTH_PASS` to strong values before exposing the service.
- Serve the app behind HTTPS if you deploy beyond localhost.
- Provider definitions and API keys live under `/app/state`. Keep that directory on an encrypted disk or secret store if you deploy to a shared environment.
- The provided `docker-compose.yml` bind-mounts `./static` (read-only), `./logs`, `./config`, and `./state`. Adjust the paths if you prefer Docker volumes.

## Project Layout
```
.
├── auth.py                # HTTP Basic auth dependency
├── config/llm_config.json # Provider definitions (no secrets)
├── docker-compose.yml     # Local orchestration
├── Dockerfile             # Production container
├── main.py                # FastAPI application
├── requirements.txt
├── static/                # Tailwind styles + JS controllers
├── state/                 # Writable runtime state (providers + keys)
└── templates/             # Jinja templates
```
