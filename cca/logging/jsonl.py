"""Append-only JSONL logger for LLM calls and tool calls (GUIDE rule 3)."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path


def prompt_hash(messages) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16]


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, kind: str, **fields) -> None:
        rec = {"ts": time.time(), "kind": kind, **fields}
        line = json.dumps(rec, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def llm_call(self, *, model, seed, temperature, messages, in_tokens, out_tokens,
                 latency_s, cost_usd=0.0, **extra):
        self.log("llm_call", model=model, seed=seed, temperature=temperature,
                 prompt_hash=prompt_hash(messages), in_tokens=in_tokens,
                 out_tokens=out_tokens, latency_s=latency_s, cost_usd=cost_usd, **extra)

    def tool_call(self, *, tool, args, ok, latency_s, **extra):
        self.log("tool_call", tool=tool, args=args, ok=ok, latency_s=latency_s, **extra)
