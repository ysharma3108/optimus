"""One interface for every model: Claude, a local Ollama model, or a fake.

The pipeline only ever calls `client.complete(...)`, so the pilot can swap
models without touching anything else. Every call is written to the token
log with the counts the provider reports.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, List, Optional

from . import config
from .tokenlog import CallRecord, TokenLog


@dataclass
class Reply:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    seconds: float = 0.0
    stop_reason: str = ""  # "refusal" means the model declined; text is then empty


class Client:
    model: str = ""

    def __init__(self, log: Optional[TokenLog] = None):
        self.log = log

    def _send(self, system: str, messages: List[dict], max_tokens: int) -> Reply:
        raise NotImplementedError

    def complete(self, system: str, messages: List[dict], route: str, max_tokens: int = 2000,
                 request_id: str = "") -> Reply:
        t0 = time.perf_counter()
        reply = self._send(system, messages, max_tokens)
        reply.seconds = time.perf_counter() - t0
        if self.log is not None:
            self.log.add(CallRecord(
                model=reply.model, route=route, input_tokens=reply.input_tokens,
                output_tokens=reply.output_tokens, cache_read_tokens=reply.cache_read_tokens,
                cache_write_tokens=reply.cache_write_tokens, seconds=reply.seconds,
                request_id=request_id, outcome=reply.stop_reason,
            ))
        return reply


class AnthropicClient(Client):
    """Claude through the official SDK (`pip install anthropic`, and set
    ANTHROPIC_API_KEY). The system prompt is marked for prompt caching, so
    repeated instructions are billed at the cache-read rate."""

    def __init__(self, model: str, log: Optional[TokenLog] = None):
        super().__init__(log)
        import anthropic  # imported here so tests and offline use need no SDK

        self.model = model
        self.sdk = anthropic.Anthropic()

    def _send(self, system, messages, max_tokens):
        r = self.sdk.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
        )
        u = r.usage
        return Reply(
            text="".join(b.text for b in r.content if getattr(b, "type", "") == "text"),
            model=self.model,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            stop_reason=r.stop_reason or "",
        )


class OllamaClient(Client):
    """A local model served by Ollama (https://ollama.com) on this PC."""

    def __init__(self, model: str = config.MODEL_LOCAL, url: str = config.OLLAMA_URL,
                 log: Optional[TokenLog] = None, num_ctx: int = 8192):
        super().__init__(log)
        self.model, self.url, self.num_ctx = model, url.rstrip("/"), num_ctx

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())

    def _send(self, system, messages, max_tokens):
        r = self._post("/api/chat", {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": False,
            "options": {"num_ctx": self.num_ctx, "num_predict": max_tokens},
        })
        return Reply(text=r["message"]["content"], model=self.model,
                     input_tokens=r.get("prompt_eval_count", 0), output_tokens=r.get("eval_count", 0))

    def embed(self, texts: List[str], model: str = config.MODEL_LOCAL_EMBED) -> List[List[float]]:
        return self._post("/api/embed", {"model": model, "input": texts})["embeddings"]


class FakeClient(Client):
    """Replays scripted replies; used by the tests."""

    def __init__(self, replies: List[str] | Callable[[str, List[dict]], str], model: str = "fake",
                 log: Optional[TokenLog] = None):
        super().__init__(log)
        self.model = model
        self.replies = replies
        self.calls: List[dict] = []

    def _send(self, system, messages, max_tokens):
        self.calls.append({"system": system, "messages": messages})
        if callable(self.replies):
            text = self.replies(system, messages)
        else:
            text = self.replies[len(self.calls) - 1]
        n_in = len((system + json.dumps(messages)).split())
        return Reply(text=text, model=self.model, input_tokens=n_in, output_tokens=len(text.split()))
