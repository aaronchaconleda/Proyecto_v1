from __future__ import annotations

from src.config.settings import AppSettings
from src.llm.lmstudio_client import LMStudioClient
from src.llm.openai_client import OpenAIClient


def create_llm_client(settings: AppSettings):
    provider = settings.llm_provider.lower()
    if provider == "lmstudio":
        return LMStudioClient(settings.lmstudio_base_url, settings.lmstudio_api_key)
    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY no configurada. Define la clave en .env.")
        return OpenAIClient(settings.openai_base_url, settings.openai_api_key)
    raise ValueError(f"RAG_LLM_PROVIDER no soportado: {settings.llm_provider}")
