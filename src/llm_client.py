"""LLM provider abstraction.

Defines a single Protocol that any LLM backend must implement, plus a concrete
implementation for DeepSeek (via the OpenAI-compatible API).

Adding a new provider is one class:
- ClaudeClient -> wraps anthropic SDK
- OllamaClient -> wraps local Ollama HTTP API
- OpenAIClient -> wraps OpenAI directly

Switch providers via LLM_PROVIDER env var. Calling code stays the same.
"""

from __future__ import annotations

from typing import Protocol

from loguru import logger
from openai import OpenAI

from config import settings


class LLMClient(Protocol):
    """Interface every LLM provider must implement."""

    def generate(self, system_prompt: str, user_prompt: str, temperature: float | None = None) -> str:
        """Generate a single response. Returns the assistant's text reply."""
        ...


class DeepSeekClient:
    """DeepSeek backend using the OpenAI-compatible API.

    DeepSeek exposes /v1/chat/completions that's drop-in compatible with the
    OpenAI Python SDK. We just point base_url at DeepSeek's endpoint.
    """

    BASE_URL = "https://api.deepseek.com/v1"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        key = api_key or settings.deepseek_api_key
        if not key:
            raise ValueError(
                "DEEPSEEK_API_KEY is empty. Set it in your .env file. "
                "Get a key at https://platform.deepseek.com/api_keys"
            )

        self.client = OpenAI(api_key=key, base_url=self.BASE_URL)
        self.model = model or settings.llm_model
        logger.info(f"DeepSeekClient ready: model={self.model}")

    def generate(self, system_prompt: str, user_prompt: str, temperature: float | None = None) -> str:
        """Send a chat completion request to DeepSeek."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature if temperature is not None else settings.temperature,
            max_tokens=settings.max_tokens,
        )
        return response.choices[0].message.content or ""


def get_llm_client() -> LLMClient:
    """Factory: return the configured LLM client.

    Reads LLM_PROVIDER from settings and instantiates the right adapter.
    Add new providers as elif branches here.
    """
    provider = settings.llm_provider.lower()

    if provider == "deepseek":
        return DeepSeekClient()

    # Placeholders — fill in when you add these adapters later:
    # if provider == "claude":
    #     return ClaudeClient()
    # if provider == "ollama":
    #     return OllamaClient()

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider}. Supported: deepseek "
        "(claude and ollama can be added — see src/llm_client.py)"
    )
