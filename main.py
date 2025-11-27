import asyncio
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, unquote

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from auth import require_basic_auth

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "llm_config.json"
STATE_DIR = BASE_DIR / "state"
RUNTIME_CONFIG_PATH = STATE_DIR / "providers.json"
KEY_STORE_PATH = STATE_DIR / "api_keys.json"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"

STATE_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)

API_KEY_TOKEN_PATTERN = re.compile(r"{+\s*api_key\s*}+", re.IGNORECASE)


def _http_error_detail(response: httpx.Response) -> str:
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            error_payload = payload.get("error", payload)
            if isinstance(error_payload, dict):
                parts = []
                message = error_payload.get("message")
                status = error_payload.get("status")
                reason = error_payload.get("code")
                if message:
                    parts.append(str(message))
                if status:
                    parts.append(f"status={status}")
                if reason:
                    parts.append(f"code={reason}")
                if parts:
                    return " | ".join(parts)
            return json.dumps(payload)
    text = response.text.strip()
    return text or "No response body"


def _normalize_base_url_placeholder(value: str) -> str:
    decoded = unquote(value)
    normalized = API_KEY_TOKEN_PATTERN.sub("{{api_key}}", decoded)
    return normalized


def _normalize_provider_definition(provider: Dict[str, Any]) -> None:
    base_url = provider.get("base_url")
    if base_url:
        normalized = _normalize_base_url_placeholder(base_url)
        provider["base_url"] = normalized


class LLMRouter:
    def __init__(self, config_path: Path, runtime_path: Optional[Path] = None) -> None:
        self.config_path = config_path
        self.runtime_path = runtime_path
        self.providers: List[Dict[str, Any]] = []
        self.reload()

    def _active_path(self) -> Optional[Path]:
        if self.runtime_path and self.runtime_path.exists():
            return self.runtime_path
        if self.config_path.exists():
            return self.config_path
        return None

    def reload(self, force_default: bool = False) -> None:
        active_path: Optional[Path]
        if force_default and self.config_path.exists():
            active_path = self.config_path
        else:
            active_path = self._active_path()
        if not active_path:
            logger.warning("Configuration file %s not found", self.config_path)
            self.providers = []
            return
        try:
            with active_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.providers = payload.get("providers", [])
            for provider in self.providers:
                _normalize_provider_definition(provider)
            logger.info("Loaded %d LLM provider configurations", len(self.providers))
            if self.runtime_path:
                self._persist()
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse config: %s", exc)
            self.providers = []

    def _persist(self) -> None:
        if not self.runtime_path:
            return
        try:
            self.runtime_path.parent.mkdir(parents=True, exist_ok=True)
            with self.runtime_path.open("w", encoding="utf-8") as handle:
                json.dump({"providers": self.providers}, handle, indent=2)
        except OSError as exc:
            logger.error("Failed to persist provider config: %s", exc)

    def has_providers(self) -> bool:
        return bool(self.providers)

    def list_providers(self) -> List[Dict[str, Any]]:
        return self.providers

    def upsert_provider(self, provider: Dict[str, Any]) -> None:
        existing = next((idx for idx, item in enumerate(self.providers) if item.get("name") == provider.get("name")), None)
        if existing is None:
            self.providers.append(provider)
        else:
            self.providers[existing] = provider
        self._persist()

    def remove_provider(self, name: str) -> bool:
        initial_count = len(self.providers)
        self.providers = [provider for provider in self.providers if provider.get("name") != name]
        removed = len(self.providers) != initial_count
        if removed:
            self._persist()
        return removed


router = LLMRouter(CONFIG_PATH, RUNTIME_CONFIG_PATH)
app = FastAPI(title="LLM Committee", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

API_KEYS: Dict[str, str] = {}


PLACEHOLDER_PATTERN = re.compile(r"{{\s*([\w\-\.]+)\s*}}")
SINGLE_PLACEHOLDER_PATTERN = re.compile(r"^{{\s*([\w\-\.]+)\s*}}$")


class ProviderRequestPayload(BaseModel):
    method: str = Field("POST", description="HTTP method for the provider request")
    headers: List[Dict[str, str]] = Field(default_factory=list)
    body: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {"type": "json", "template": {"input": "{{prompt}}"}}
    )


class ProviderPayload(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)
    display_name: str = Field(..., min_length=2, max_length=100)
    base_url: str = Field(..., min_length=5)
    timeout: int = Field(default=60, ge=5, le=300)
    mock: bool = False
    description: Optional[str] = None
    request: Optional[ProviderRequestPayload] = None
    variables: Dict[str, Any] = Field(default_factory=dict)
    requires_api_key: bool = True


class ProviderIntrospectPayload(BaseModel):
    base_url: str = Field(..., min_length=5)


def _load_api_keys() -> Dict[str, str]:
    if not KEY_STORE_PATH.exists():
        return {}
    try:
        with KEY_STORE_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {str(k): str(v) for k, v in payload.items()}
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Unable to read API key store: %s", exc)
        return {}


def _persist_api_keys() -> None:
    try:
        KEY_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with KEY_STORE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(API_KEYS, handle, indent=2)
    except OSError as exc:
        logger.error("Unable to persist API keys: %s", exc)


API_KEYS.update(_load_api_keys())


class QueryPayload(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)


class KeyUpdatePayload(BaseModel):
    keys: Dict[str, Optional[str]]


def _render_template(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        single_match = SINGLE_PLACEHOLDER_PATTERN.match(value)
        if single_match:
            key = single_match.group(1)
            return context.get(key, value)

        def replacer(match: re.Match[str]) -> str:
            key = match.group(1)
            return str(context.get(key, match.group(0)))

        return PLACEHOLDER_PATTERN.sub(replacer, value)
    if isinstance(value, list):
        return [_render_template(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _render_template(val, context) for key, val in value.items()}
    return value


def _build_headers(headers_cfg: List[Dict[str, str]], context: Dict[str, Any]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for header in headers_cfg:
        key = header.get("key")
        value = header.get("value")
        if not key or value is None:
            continue
        headers[key] = _render_template(value, context)
    return headers


async def _call_openai_compatible(provider: Dict[str, Any], prompt: str, api_key: Optional[str], timeout: int) -> Any:
    if not api_key:
        raise RuntimeError(f"API key for {provider.get('name', 'unknown')} is not configured")
    base_url = provider.get("base_url")
    if not base_url:
        raise RuntimeError("Provider base_url missing")
    base_url = _normalize_base_url_placeholder(base_url)
    model = provider.get("model") or provider.get("variables", {}).get("model")
    if not model:
        raise RuntimeError("Model is required for OpenAI-compatible providers")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = []
    system_prompt = provider.get("variables", {}).get("system_prompt")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = {"model": model, "messages": messages}
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(base_url, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _http_error_detail(exc.response)
            raise RuntimeError(f"HTTP {exc.response.status_code}: {detail}") from exc
        return response.json()


async def _call_dynamic_provider(provider: Dict[str, Any], prompt: str, api_key: Optional[str], timeout: int) -> Any:
    request_cfg = provider.get("request") or {}
    base_url = provider.get("base_url")
    if not base_url:
        raise RuntimeError("Provider base_url missing")
    context = {
        "prompt": prompt,
        "api_key": api_key or "",
        "provider_name": provider.get("name", "provider"),
        **provider.get("variables", {}),
    }
    rendered_base_url = _render_template(base_url, context)
    api_key_value = context.get("api_key", "")
    if api_key_value:
        for token in ("{{api_key}}", "{api_key}", "%7Bapi_key%7D", "%7bapi_key%7d"):
            if token in rendered_base_url:
                rendered_base_url = rendered_base_url.replace(token, api_key_value)
    method = (request_cfg.get("method") or "POST").upper()
    headers_cfg = request_cfg.get("headers", [])
    headers = _build_headers(headers_cfg, context)
    body_cfg = request_cfg.get("body") or {"type": "json", "template": {"input": "{{prompt}}"}}
    json_payload = None
    data_payload = None
    template = body_cfg.get("template")
    if body_cfg.get("type", "json") == "json" and template is not None:
        json_payload = _render_template(template, context)
    elif template is not None:
        data_payload = _render_template(template, context)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.request(
                method,
                rendered_base_url,
                headers=headers,
                json=json_payload,
                data=data_payload,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _http_error_detail(exc.response)
            raise RuntimeError(f"HTTP {exc.response.status_code}: {detail}") from exc
        if "application/json" in response.headers.get("content-type", ""):
            return response.json()
        return response.text


def _slug_from_url(base_url: str, fallback: str = "provider") -> str:
    try:
        parsed = urlparse(base_url)
    except ValueError:
        return fallback
    host = parsed.netloc or fallback
    host = host.split(":")[0]
    slug = re.sub(r"[^a-z0-9]+", "-", host.lower()).strip("-")
    return slug or fallback


def _append_api_key_query(base_url: str) -> str:
    if "{{api_key}}" in base_url:
        return base_url
    try:
        parsed = urlparse(base_url)
    except ValueError:
        return base_url
    if "key=" in parsed.query:
        return base_url
    suffix = "&" if parsed.query else "?"
    if base_url.endswith(("?", "&")):
        suffix = ""
    return f"{base_url}{suffix}key={{api_key}}"


def _openai_schema(base_url: str) -> Dict[str, Any]:
    return {
        "provider_type": "openai-compatible",
        "suggested_name": "openai-" + _slug_from_url(base_url, "openai"),
        "suggested_display_name": "OpenAI Compatible",
        "requires_api_key": True,
        "allow_request_edit": False,
        "notes": "Payload matches OpenAI Chat Completions APIs.",
        "timeout": 60,
        "variables": {
            "model": "gpt-4o-mini",
            "system_prompt": "You are one member of the LLM Committee."
        },
        "variable_schema": [
            {"key": "model", "label": "Model name", "type": "text", "required": True, "placeholder": "gpt-4o-mini"},
            {"key": "system_prompt", "label": "System prompt", "type": "textarea", "required": False, "placeholder": "You are a helpful assistant."}
        ],
        "request": {
            "method": "POST",
            "headers": [
                {"key": "Content-Type", "value": "application/json"},
                {"key": "Authorization", "value": "Bearer {{api_key}}"}
            ],
            "body": {
                "type": "json",
                "template": {
                    "model": "{{model}}",
                    "messages": [
                        {"role": "system", "content": "{{system_prompt}}"},
                        {"role": "user", "content": "{{prompt}}"}
                    ]
                }
            }
        }
    }


def _anthropic_schema(base_url: str) -> Dict[str, Any]:
    return {
        "provider_type": "anthropic",
        "suggested_name": "anthropic-" + _slug_from_url(base_url, "anthropic"),
        "suggested_display_name": "Anthropic Messages",
        "requires_api_key": True,
        "allow_request_edit": False,
        "notes": "Uses Anthropic Messages format with Claude models.",
        "timeout": 60,
        "variables": {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1024
        },
        "variable_schema": [
            {"key": "model", "label": "Model", "type": "text", "required": True, "placeholder": "claude-3-sonnet-20240229"},
            {"key": "max_tokens", "label": "Max tokens", "type": "number", "required": True, "placeholder": "1024"}
        ],
        "request": {
            "method": "POST",
            "headers": [
                {"key": "Content-Type", "value": "application/json"},
                {"key": "x-api-key", "value": "{{api_key}}"},
                {"key": "anthropic-version", "value": "2023-06-01"}
            ],
            "body": {
                "type": "json",
                "template": {
                    "model": "{{model}}",
                    "max_tokens": "{{max_tokens}}",
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "{{prompt}}"}]}
                    ]
                }
            }
        }
    }


def _cohere_schema(base_url: str) -> Dict[str, Any]:
    return {
        "provider_type": "cohere",
        "suggested_name": "cohere-" + _slug_from_url(base_url, "cohere"),
        "suggested_display_name": "Cohere Generate",
        "requires_api_key": True,
        "allow_request_edit": False,
        "notes": "Targets Cohere's text generation endpoint.",
        "timeout": 45,
        "variables": {
            "model": "command-r",
            "max_tokens": 400
        },
        "variable_schema": [
            {"key": "model", "label": "Model", "type": "text", "required": True, "placeholder": "command-r"},
            {"key": "max_tokens", "label": "Max tokens", "type": "number", "required": False, "placeholder": "400"}
        ],
        "request": {
            "method": "POST",
            "headers": [
                {"key": "Content-Type", "value": "application/json"},
                {"key": "Authorization", "value": "Bearer {{api_key}}"}
            ],
            "body": {
                "type": "json",
                "template": {
                    "model": "{{model}}",
                    "prompt": "{{prompt}}",
                    "max_tokens": "{{max_tokens}}"
                }
            }
        }
    }


def _google_generative_schema(base_url: str) -> Dict[str, Any]:
    base_with_key = _append_api_key_query(base_url)
    return {
        "provider_type": "google-generative-language",
        "suggested_name": "gemini-" + _slug_from_url(base_url, "gemini"),
        "suggested_display_name": "Gemini (Generative Language)",
        "requires_api_key": True,
        "allow_request_edit": False,
        "notes": "Google Generative Language API requires the key as a query parameter; it has been appended automatically.",
        "timeout": 60,
        "variables": {},
        "variable_schema": [],
        "request": {
            "method": "POST",
            "headers": [
                {"key": "Content-Type", "value": "application/json"}
            ],
            "body": {
                "type": "json",
                "template": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": "{{prompt}}"}
                            ]
                        }
                    ]
                }
            }
        },
        "base_url": base_with_key,
    }


def _generic_schema(base_url: str) -> Dict[str, Any]:
    slug = _slug_from_url(base_url, "custom")
    return {
        "provider_type": "generic",
        "suggested_name": slug,
        "suggested_display_name": slug.replace("-", " ").title(),
        "requires_api_key": True,
        "allow_request_edit": True,
        "notes": "Edit the JSON request template to match your provider.",
        "timeout": 60,
        "variables": {
            "model": "",
            "extra": ""
        },
        "variable_schema": [
            {"key": "model", "label": "Model or deployment", "type": "text", "required": False},
            {"key": "extra", "label": "Extra parameter", "type": "text", "required": False}
        ],
        "request": {
            "method": "POST",
            "headers": [
                {"key": "Content-Type", "value": "application/json"},
                {"key": "Authorization", "value": "Bearer {{api_key}}"}
            ],
            "body": {
                "type": "json",
                "template": {
                    "model": "{{model}}",
                    "input": "{{prompt}}",
                    "extra": "{{extra}}"
                }
            }
        }
    }


def _detect_provider_template(base_url: str) -> Dict[str, Any]:
    try:
        host = urlparse(base_url).netloc.lower()
    except ValueError:
        host = ""
    if "openai" in host:
        return _openai_schema(base_url)
    if "anthropic" in host:
        return _anthropic_schema(base_url)
    if "cohere" in host:
        return _cohere_schema(base_url)
    if "generativelanguage" in host:
        return _google_generative_schema(base_url)
    return _generic_schema(base_url)


@app.on_event("startup")
async def startup_event() -> None:
    router.reload()
    API_KEYS.clear()
    API_KEYS.update(_load_api_keys())


def is_first_run() -> bool:
    return not router.has_providers()


@app.get("/")
async def home(request: Request, _: str = Depends(require_basic_auth)):
    if is_first_run():
        return RedirectResponse(url="/configure", status_code=status.HTTP_307_TEMPORARY_REDIRECT)
    context = {"request": request, "page": "home", "providers": router.list_providers()}
    return templates.TemplateResponse("home.html", context)


@app.get("/configure")
async def configure(request: Request, _: str = Depends(require_basic_auth)):
    context = {"request": request, "page": "configure", "providers": router.list_providers()}
    return templates.TemplateResponse("configure.html", context)


@app.get("/logs")
async def logs_page(request: Request, _: str = Depends(require_basic_auth)):
    context = {"request": request, "page": "logs"}
    return templates.TemplateResponse("logs.html", context)


@app.get("/health")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok", "providers": str(len(router.list_providers()))}


async def _call_provider(provider: Dict[str, Any], prompt: str) -> Any:
    provider_name = provider.get("name", "unknown")
    timeout = int(provider.get("timeout", 30))

    if provider.get("mock", False):
        await asyncio.sleep(min(2, timeout))
        return {
            "message": f"Mock response from {provider.get('display_name', provider_name)}",
            "echo": prompt[::-1],
        }

    requires_api_key = provider.get("requires_api_key", True)
    api_key = API_KEYS.get(provider_name)
    if requires_api_key and not api_key:
        raise RuntimeError(f"API key for {provider_name} is not configured")

    if provider.get("request"):
        return await _call_dynamic_provider(provider, prompt, api_key, timeout)

    return await _call_openai_compatible(provider, prompt, api_key, timeout)


async def _gather_provider_response(provider: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    provider_name = provider.get("name", "unknown")
    timeout = int(provider.get("timeout", 30))
    try:
        response = await asyncio.wait_for(_call_provider(provider, prompt), timeout=timeout)
        return {"status": "ok", "response": response}
    except asyncio.TimeoutError:
        logger.warning("Provider %s timed out", provider_name)
        return {"status": "timeout", "response": "Provider timed out"}
    except Exception as exc:  # broad for safety
        logger.exception("Provider %s failed", provider_name)
        return {"status": "error", "response": str(exc)}


@app.post("/api/query")
async def query_llms(payload: QueryPayload, _: str = Depends(require_basic_auth)) -> JSONResponse:
    if not router.has_providers():
        raise HTTPException(status_code=400, detail="No LLM providers configured")

    tasks = [
        _gather_provider_response(provider, payload.prompt)
        for provider in router.list_providers()
    ]

    results = await asyncio.gather(*tasks)
    response_payload = {
        provider.get("name", f"provider-{idx}"): result
        for idx, (provider, result) in enumerate(zip(router.list_providers(), results))
    }
    return JSONResponse({"responses": response_payload})


@app.post("/api/keys")
async def update_keys(payload: KeyUpdatePayload, _: str = Depends(require_basic_auth)) -> Dict[str, Any]:
    changed: List[str] = []
    for provider, raw_key in payload.keys.items():
        if raw_key is None:
            if provider in API_KEYS:
                API_KEYS.pop(provider)
                changed.append(provider)
            continue
        trimmed = raw_key.strip()
        if not trimmed:
            continue
        API_KEYS[provider] = trimmed
        changed.append(provider)
    if changed:
        _persist_api_keys()
        logger.info("Updated API keys for providers: %s", ", ".join(changed))
    return {"status": "ok", "configured": list(API_KEYS.keys()), "changed": changed}


@app.get("/api/config")
async def get_config(_: str = Depends(require_basic_auth)) -> Dict[str, Any]:
    return {"providers": router.list_providers(), "api_keys": list(API_KEYS.keys())}


@app.post("/api/providers/introspect")
async def introspect_provider(payload: ProviderIntrospectPayload, _: str = Depends(require_basic_auth)) -> Dict[str, Any]:
    schema = _detect_provider_template(payload.base_url)
    base_url = schema.get("base_url") or payload.base_url
    schema["base_url"] = _normalize_base_url_placeholder(base_url)
    return schema


@app.post("/api/providers")
async def create_or_update_provider(payload: ProviderPayload, _: str = Depends(require_basic_auth)) -> Dict[str, Any]:
    provider_data = payload.model_dump()
    _normalize_provider_definition(provider_data)
    existing = next((item for item in router.list_providers() if item.get("name") == provider_data["name"]), None)
    router.upsert_provider(provider_data)
    action = "created" if existing is None else "updated"
    logger.info("%s provider %s", action.capitalize(), provider_data["name"])
    return {"status": action, "provider": provider_data}


@app.delete("/api/providers/{name}")
async def delete_provider(name: str, _: str = Depends(require_basic_auth)) -> Dict[str, Any]:
    if not router.remove_provider(name):
        raise HTTPException(status_code=404, detail="Provider not found")
    if name in API_KEYS:
        API_KEYS.pop(name)
        _persist_api_keys()
    return {"status": "deleted", "name": name}


@app.get("/api/logs")
async def get_logs(_: str = Depends(require_basic_auth)) -> Dict[str, Any]:
    if not LOG_FILE.exists():
        return {"logs": []}
    try:
        with LOG_FILE.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()[-200:]
        return {"logs": lines}
    except Exception as exc:
        logger.error("Unable to read logs: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to read logs")


@app.post("/api/reload-config")
async def reload_config(_: str = Depends(require_basic_auth)) -> Dict[str, Any]:
    router.reload(force_default=True)
    return {"providers": router.list_providers(), "count": len(router.list_providers())}
