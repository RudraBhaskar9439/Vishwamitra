from __future__ import annotations
import os
import json
import hashlib
import asyncio
from pathlib import Path
from typing import Any
from dataclasses import dataclass

try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError(
        "openai package required: pip install 'openai>=1.40.0'"
    ) from e


# Provider auto-detection table. Used only when API_BASE_URL / MODEL_NAME
# are not explicitly set in env. Keys checked in priority order.
_PROVIDER_DEFAULTS: list[tuple[str, str, str]] = [
    # (env_key, base_url, default_model)
    ("GROQ_API_KEY",      "https://api.groq.com/openai/v1",   "llama-3.3-70b-versatile"),
    ("TOGETHER_API_KEY",  "https://api.together.xyz/v1",      "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
    ("FIREWORKS_API_KEY", "https://api.fireworks.ai/inference/v1", "accounts/fireworks/models/llama-v3p3-70b-instruct"),
    ("HF_TOKEN",          "https://router.huggingface.co/v1", "meta-llama/Llama-3.1-70B-Instruct"),
    ("OPENAI_API_KEY",    "https://api.openai.com/v1",        "gpt-4o-mini"),
]


@dataclass
class LLMConfig:
    base_url: str
    model: str
    api_key: str
    provider_name: str


def _resolve_config() -> LLMConfig:
    """
    Pick provider config from env. Honor explicit API_BASE_URL/MODEL_NAME
    overrides if set; otherwise auto-detect from whichever provider key
    is present.
    """
    explicit_base = os.getenv("API_BASE_URL")
    explicit_model = os.getenv("MODEL_NAME")

    # If user has explicit base+model, just find any matching key
    if explicit_base and explicit_model:
        for env_key, _, _ in _PROVIDER_DEFAULTS:
            if os.getenv(env_key):
                return LLMConfig(
                    base_url=explicit_base,
                    model=explicit_model,
                    api_key=os.environ[env_key],
                    provider_name=f"explicit({env_key})",
                )
        raise RuntimeError(
            "API_BASE_URL/MODEL_NAME set but no provider API key found. "
            "Set one of: GROQ_API_KEY, TOGETHER_API_KEY, FIREWORKS_API_KEY, HF_TOKEN, OPENAI_API_KEY"
        )

    # Auto-detect from first available key
    for env_key, base_url, model in _PROVIDER_DEFAULTS:
        key_val = os.getenv(env_key)
        if key_val:
            return LLMConfig(
                base_url=explicit_base or base_url,
                model=explicit_model or model,
                api_key=key_val,
                provider_name=env_key,
            )

    raise RuntimeError(
        "No LLM provider key found in environment. "
        "Set one of: GROQ_API_KEY, TOGETHER_API_KEY, FIREWORKS_API_KEY, HF_TOKEN, OPENAI_API_KEY"
    )


class LLMClient:
    """
    Thin async wrapper around an OpenAI-compatible chat completion API.

    Features:
      - Provider auto-detection (Groq / Together / Fireworks / HF / OpenAI)
      - File-based response cache keyed on (model, system, user, json_mode)
      - JSON-mode helper that retries once on parse failure
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache_dir: str | Path | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        self.config = config or _resolve_config()
        self.client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[1] / "logs" / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------- cache helpers -------------------------
    def _cache_key(self, system: str, user: str, json_mode: bool) -> str:
        h = hashlib.sha256()
        h.update(self.config.model.encode())
        h.update(b"\x00")
        h.update(system.encode())
        h.update(b"\x00")
        h.update(user.encode())
        h.update(b"\x00")
        h.update(b"json" if json_mode else b"text")
        return h.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> str | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())["content"]
        except Exception:
            return None

    def _cache_put(self, key: str, content: str) -> None:
        try:
            self._cache_path(key).write_text(json.dumps({"content": content}))
        except Exception:
            pass  # cache is best-effort

    # --------------------------- core call ---------------------------
    async def chat(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.8,
        max_tokens: int = 1024,
        use_cache: bool = True,
    ) -> str:
        """
        Send a single-turn chat. Returns the assistant content string.

        json_mode requests structured JSON output via response_format
        when the provider supports it. Falls back silently if not.
        """
        key = self._cache_key(system, user, json_mode)
        if use_cache:
            cached = self._cache_get(key)
            if cached is not None:
                return cached

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = await self.client.chat.completions.create(**kwargs)
        except Exception as e:
            # If response_format is the issue, retry without it once.
            if json_mode and "response_format" in str(e):
                kwargs.pop("response_format", None)
                resp = await self.client.chat.completions.create(**kwargs)
            else:
                raise

        content = resp.choices[0].message.content or ""
        if use_cache:
            self._cache_put(key, content)
        return content

    async def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.8,
        max_tokens: int = 1024,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """
        Same as chat() but parses the result as JSON. Retries once with
        a stricter instruction if the first parse fails.
        """
        raw = await self.chat(
            system=system,
            user=user,
            json_mode=True,
            temperature=temperature,
            max_tokens=max_tokens,
            use_cache=use_cache,
        )
        try:
            return json.loads(_extract_json(raw))
        except Exception:
            # Retry once, no cache, with a sharper directive.
            stricter = user + (
                "\n\nIMPORTANT: Respond ONLY with a single valid JSON object. "
                "No prose, no code fences, no commentary."
            )
            raw2 = await self.chat(
                system=system,
                user=stricter,
                json_mode=True,
                temperature=max(0.2, temperature - 0.4),
                max_tokens=max_tokens,
                use_cache=False,
            )
            return json.loads(_extract_json(raw2))


def _extract_json(text: str) -> str:
    """Strip code fences / leading prose; return the largest JSON-looking blob."""
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        # remove a leading "json" hint if present
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    # Find first { and matching last }
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        return s[start : end + 1]
    return s


__all__ = ["LLMClient", "LLMConfig"]
