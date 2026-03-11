from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


class OpenAIClient:
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
        self._raise_for_status_with_details(response)
        data = response.json()["data"]
        return [item["embedding"] for item in data]

    @staticmethod
    def _raise_for_status_with_details(response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            if detail:
                raise requests.HTTPError(f"{exc}. body={detail}", response=response) from exc
            raise

    def chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_completion_tokens"] = max_tokens

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=120,
        )
        self._raise_for_status_with_details(response)
        return response.json()

    def list_models(self) -> List[Dict[str, Any]]:
        response = requests.get(
            f"{self.base_url}/models",
            headers=self._headers(),
            timeout=30,
        )
        self._raise_for_status_with_details(response)
        payload = response.json()
        if isinstance(payload, dict) and "data" in payload:
            return payload["data"]
        if isinstance(payload, dict) and "models" in payload:
            return payload["models"]
        return []
