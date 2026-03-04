from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


class LMStudioClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def embed_texts(self, model: str, texts: List[str]) -> List[List[float]]:
        payload = {"model": model, "input": texts}
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=self._headers(),
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in data]

    def chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = 700,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def list_models(self) -> List[Dict[str, Any]]:
        base = self.base_url
        root = base[:-3] if base.endswith("/v1") else base
        candidate_urls = [
            f"{base}/models",
            f"{root}/api/v1/models",
            f"{root}/v1/models",
        ]

        for url in candidate_urls:
            try:
                response = requests.get(
                    url,
                    headers=self._headers(),
                    timeout=30,
                )
                if response.status_code == 200:
                    payload = response.json()
                    if isinstance(payload, dict) and "data" in payload:
                        return payload["data"]
                    if isinstance(payload, dict) and "models" in payload:
                        return payload["models"]
            except requests.RequestException:
                continue
        raise requests.HTTPError("No se pudo obtener el listado de modelos desde LM Studio.")
