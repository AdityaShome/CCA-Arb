"""OpenAI-compatible chat client (vLLM, Colab, or hosted API) with logging."""
from __future__ import annotations

import time
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from cca.logging.jsonl import JsonlLogger


@dataclass
class ModelSpec:
    name: str                      # model id sent to the server
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.7
    max_tokens: int = 512
    usd_per_1k_in: float = 0.0
    usd_per_1k_out: float = 0.0


class ChatClient:
    def __init__(self, spec: ModelSpec, logger: JsonlLogger | None = None):
        from openai import OpenAI  # lazy so tests need no server
        self.spec, self.logger = spec, logger
        self._c = OpenAI(base_url=spec.base_url, api_key=spec.api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat(self, messages, seed: int, **meta) -> str:
        s, t0 = self.spec, time.time()
        r = self._c.chat.completions.create(
            model=s.name, messages=messages, temperature=s.temperature,
            max_tokens=s.max_tokens, seed=seed)
        dt = time.time() - t0
        u = r.usage
        if self.logger:
            cost = (u.prompt_tokens * s.usd_per_1k_in
                    + u.completion_tokens * s.usd_per_1k_out) / 1000
            self.logger.llm_call(model=s.name, seed=seed, temperature=s.temperature,
                                 messages=messages, in_tokens=u.prompt_tokens,
                                 out_tokens=u.completion_tokens, latency_s=dt,
                                 cost_usd=cost, **meta)
        return r.choices[0].message.content or ""
