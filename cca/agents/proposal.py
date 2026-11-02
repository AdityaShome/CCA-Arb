"""Proposal schema + validation (GUIDE 6.2)."""
from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

DecisionType = Literal["plan", "tool_choice", "code_edit"]


class Action(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class Proposal(BaseModel):
    agent_id: str
    decision_type: DecisionType
    action: Action
    justification: str = ""
    self_confidence: float = Field(0.5, ge=0.0, le=1.0)


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of model text (handles code fences)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    cand = m.group(1) if m else None
    if cand is None:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(text)):
            depth += text[i] == "{"
            depth -= text[i] == "}"
            if depth == 0:
                cand = text[start:i + 1]
                break
    try:
        return json.loads(cand) if cand else None
    except json.JSONDecodeError:
        return None


def parse_proposal(text: str, agent_id: str, default_type: DecisionType = "tool_choice"
                   ) -> Proposal | None:
    obj = extract_json(text)
    if not isinstance(obj, dict):
        return None
    obj["agent_id"] = agent_id  # never trust model-claimed identity
    obj.setdefault("decision_type", default_type)
    try:
        return Proposal(**obj)
    except ValidationError:
        return None
