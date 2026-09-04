"""
Asynchronous Ollama API client for embeddings and streaming chat.
Enforces short keep_alive ('5m') so models unload from memory when idle,
respecting the 16 GB hardware budget.
"""

from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import httpx

from config import (
    EMBED_MODEL,
    EMBEDDING_BATCH_SIZE,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    get_active_llm_model,
)


class OllamaClient:
    """
    Async client for local Ollama server (localhost:11434).
    Supports batched embeddings and token-by-token streaming chat.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        keep_alive: str = OLLAMA_KEEP_ALIVE,
        embed_model: str = EMBED_MODEL,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive
        self.embed_model = embed_model
        self.timeout = timeout

    @property
    def llm_model(self) -> str:
        return get_active_llm_model()

    async def check_connection(self) -> Tuple[bool, List[str], str]:
        """
        Checks if Ollama is running and lists installed models.
        Returns: (is_connected, list_of_model_names, message)
        """
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(url)
                if res.status_code == 200:
                    data = res.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    return True, models, "Ollama is running"
                return False, [], f"Ollama returned HTTP {res.status_code}"
        except httpx.ConnectError:
            return False, [], "Cannot connect to Ollama at http://localhost:11434. Is Ollama running?"
        except Exception as e:
            return False, [], f"Ollama connection error: {str(e)}"

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Computes 768-dim embeddings for a list of text strings in modest batches
        to prevent memory spikes.
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        batch_size = max(1, EMBEDDING_BATCH_SIZE)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                # Modern Ollama batch endpoint: /api/embed
                embed_url = f"{self.base_url}/api/embed"
                payload = {
                    "model": self.embed_model,
                    "input": batch,
                    "keep_alive": self.keep_alive,
                }
                try:
                    res = await client.post(embed_url, json=payload)
                    if res.status_code == 200:
                        data = res.json()
                        embeddings = data.get("embeddings", [])
                        all_embeddings.extend(embeddings)
                        continue
                except Exception:
                    pass

                # Fallback to single /api/embeddings if batch endpoint is unavailable
                for text in batch:
                    legacy_url = f"{self.base_url}/api/embeddings"
                    legacy_payload = {
                        "model": self.embed_model,
                        "prompt": text,
                        "keep_alive": self.keep_alive,
                    }
                    res = await client.post(legacy_url, json=legacy_payload)
                    if res.status_code != 200:
                        raise RuntimeError(
                            f"Ollama embedding failed ({res.status_code}): {res.text}. "
                            f"Ensure '{self.embed_model}' is pulled via 'ollama pull {self.embed_model}'."
                        )
                    data = res.json()
                    all_embeddings.append(data.get("embedding", []))

        return all_embeddings

    async def embed_single(self, text: str) -> List[float]:
        """Embeds a single text string (e.g. search query)."""
        res = await self.embed_texts([text])
        if not res:
            raise RuntimeError("Empty embedding returned by Ollama.")
        return res[0]

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> AsyncGenerator[str, None]:
        """
        Streams response tokens from Ollama's chat API (/api/chat).
        Sets keep_alive so the LLM unloads after 5m of inactivity.
        """
        chat_url = f"{self.base_url}/api/chat"

        full_messages: List[Dict[str, str]] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload = {
            "model": self.llm_model,
            "messages": full_messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", chat_url, json=payload) as response:
                if response.status_code != 200:
                    error_detail = await response.aread()
                    raise RuntimeError(
                        f"Ollama chat error ({response.status_code}): {error_detail.decode('utf-8', errors='replace')}. "
                        f"Ensure model '{self.llm_model}' is pulled via 'ollama pull {self.llm_model}'."
                    )
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        import json
                        chunk = json.loads(line)
                        delta = chunk.get("message", {}).get("content", "")
                        if delta:
                            yield delta
                        if chunk.get("done", False):
                            break
                    except Exception:
                        continue
