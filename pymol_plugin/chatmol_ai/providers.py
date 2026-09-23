"""LLM provider registry and OpenAI-compatible HTTP client."""

import json
import requests

# 1. Provider registry + LLMClient — multi-provider HTTP wrapper (no SDK)
# ---------------------------------------------------------------------------

# Each provider entry:
#   base_url  – chat completions endpoint
#   env_var   – environment variable for the API key
#   models    – list of (model_id, label) for the settings UI
#   extra_headers – provider-specific headers (optional)
PROVIDERS = {
    "openrouter": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "env_var": "OPENROUTER_API_KEY",
        "models": [
            ("openai/gpt-4o", "GPT-4o"),
            ("openai/gpt-5.2", "GPT-5.2"),
            ("openai/gpt-5.3-codex", "GPT-5.3 Codex"),
            ("google/gemini-3.1-pro-preview", "Gemini 3.1 Pro"),
            ("google/gemini-3-flash-preview", "Gemini 3 Flash"),
            ("deepseek/deepseek-chat", "DeepSeek V3"),
        ],
        "vision_models": [
            ("openai/gpt-4o", "GPT-4o"),
            ("openai/gpt-5.2", "GPT-5.2"),
            ("anthropic/claude-sonnet-4.6", "Claude Sonnet 4.6"),
            ("google/gemini-3-flash-preview", "Gemini 3 Flash"),
            ("google/gemini-3.1-pro-preview", "Gemini 3.1 Pro"),
        ],
        "extra_headers": {
            "HTTP-Referer": "https://github.com/ChatMol/ChatMol",
            "X-Title": "ChatMol",
        },
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/chat/completions",
        "env_var": "DEEPSEEK_API_KEY",
        "models": [
            ("deepseek-chat", "DeepSeek V3"),
            ("deepseek-reasoner", "DeepSeek R1"),
        ],
        "vision_models": [],
        "extra_headers": {},
    },
    "kimi": {
        "label": "Kimi (Moonshot)",
        "base_url": "https://api.moonshot.ai/v1/chat/completions",
        "env_var": "MOONSHOT_API_KEY",
        "models": [
            ("kimi-k2.5", "Kimi K2.5"),
        ],
        "vision_models": [
            ("kimi-k2.5", "Kimi K2.5"),
        ],
        "extra_headers": {},
    },
    "glm": {
        "label": "GLM (Zhipu)",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "env_var": "GLM_API_KEY",
        "models": [
            ("glm-5", "GLM-5"),
        ],
        "vision_models": [],
        "extra_headers": {},
    },
    "newapi": {
        "label": "Custom / NewAPI",
        "base_url": "",
        "env_var": "CHATMOL_CUSTOM_API_KEY",
        "models": [],
        "vision_models": [],
        "extra_headers": {},
    },
}


class LLMTransientError(RuntimeError):
    """Transient API/network errors that should be retried."""


class LLMFatalError(RuntimeError):
    """Non-retriable API errors."""


def _normalize_chat_completions_url(base_url):
    """Accept an API base URL (e.g. .../v1) or a full chat-completions endpoint."""
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return ""
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


class LLMClient:
    """Thin HTTP client for OpenAI-compatible chat-completion APIs."""

    def __init__(self, provider_name, api_key, base_url_override=""):
        prov = PROVIDERS.get(provider_name, PROVIDERS["openrouter"])
        raw_base_url = base_url_override or prov["base_url"]
        self.base_url = _normalize_chat_completions_url(raw_base_url)
        self.extra_headers = prov.get("extra_headers", {})
        self.api_key = api_key

    def _headers(self):
        h = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        h.update(self.extra_headers)
        return h

    def list_models(self):
        """Return model IDs from an OpenAI-compatible GET /models endpoint."""
        if not self.base_url:
            raise LLMFatalError("No Base URL configured.")
        if self.base_url.endswith("/chat/completions"):
            models_url = self.base_url[: -len("/chat/completions")] + "/models"
        else:
            models_url = self.base_url.rstrip("/") + "/models"
        try:
            resp = requests.get(models_url, headers=self._headers(), timeout=30)
        except requests.ConnectionError:
            raise LLMTransientError(f"Connection failed: cannot reach {models_url}")
        except requests.Timeout:
            raise LLMTransientError(f"Request timed out after 30s to {models_url}")
        if resp.status_code >= 400:
            detail = resp.text[:500] if resp.text else resp.reason
            raise LLMFatalError(
                f"Model list API error {resp.status_code} from {models_url}: {detail}"
            )
        data = resp.json()
        items = data.get("data", []) if isinstance(data, dict) else []
        model_ids = []
        for item in items:
            if isinstance(item, dict) and item.get("id"):
                model_ids.append(str(item["id"]))
            elif isinstance(item, str):
                model_ids.append(item)
        return sorted(set(model_ids), key=str.lower)

    def _check_response(self, resp):
        if resp.status_code >= 400:
            try:
                body = resp.json()
                err = body.get("error", {})
                if isinstance(err, dict):
                    detail = err.get("message", "") or err.get("msg", "")
                else:
                    detail = str(err)
            except Exception:
                detail = resp.text[:500] if resp.text else ""
            message = (
                f"API error {resp.status_code} from {self.base_url}: "
                f"{detail or resp.reason}"
            )
            if resp.status_code in (408, 409, 425, 429) or resp.status_code >= 500:
                raise LLMTransientError(message)
            raise LLMFatalError(message)

    def _parse_choices(self, data):
        if "error" in data:
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise LLMFatalError(f"API returned error: {msg}")
        if "choices" not in data or not data["choices"]:
            raise LLMFatalError(
                f"Unexpected API response (no choices): "
                f"{json.dumps(data, ensure_ascii=False)[:300]}"
            )
        return data["choices"][0]

    def chat_completion(
        self, model, messages, tools=None, temperature=0.01, max_tokens=4096
    ):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        try:
            resp = requests.post(
                self.base_url, headers=self._headers(), json=payload, timeout=120
            )
        except requests.ConnectionError:
            raise LLMTransientError(f"Connection failed: cannot reach {self.base_url}")
        except requests.Timeout:
            raise LLMTransientError(f"Request timed out after 120s to {self.base_url}")
        self._check_response(resp)
        data = resp.json()
        self._parse_choices(data)  # validate structure
        return data

    def vision_completion(self, model, text_prompt, image_base64):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ]
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 1024,
        }
        try:
            resp = requests.post(
                self.base_url, headers=self._headers(), json=payload, timeout=120
            )
        except requests.ConnectionError:
            raise LLMTransientError(f"Connection failed: cannot reach {self.base_url}")
        except requests.Timeout:
            raise LLMTransientError(f"Request timed out after 120s to {self.base_url}")
        self._check_response(resp)
        data = resp.json()
        choice = self._parse_choices(data)
        return choice["message"]["content"]


# ---------------------------------------------------------------------------
