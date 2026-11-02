"""Proposal canonicalization + disagreement check (GUIDE 6.3)."""
from __future__ import annotations

import json
import re
from typing import Any

from cca.agents.proposal import Proposal

_WS = re.compile(r"\s+")


def _norm(v: Any) -> Any:
    if isinstance(v, str):
        return _WS.sub(" ", v.strip())
    if isinstance(v, dict):
        return {str(k).strip(): _norm(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, float) and v == int(v):
        return int(v)
    return v


def _norm_diff(text: str) -> str:
    """Formatting-insensitive diff: strip trailing whitespace, drop blank lines and headers' timestamps."""
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
    lines = [re.sub(r"\t.*$", "", ln) if ln.startswith(("---", "+++")) else ln for ln in lines]
    return "\n".join(ln for ln in lines if ln.strip())


def canonical(p: Proposal) -> str:
    tool = p.action.tool.strip().lower()
    args = _norm(p.action.args)
    if p.decision_type == "code_edit":
        for k in ("diff", "patch"):
            if isinstance(p.action.args.get(k), str):
                args[k] = _norm_diff(p.action.args[k])
    return json.dumps({"tool": tool, "args": args}, sort_keys=True, ensure_ascii=False)


def group_proposals(props: list[Proposal]) -> list[int]:
    """Group id per proposal; ids assigned in order of first appearance."""
    seen: dict[str, int] = {}
    return [seen.setdefault(canonical(p), len(seen)) for p in props]


def disagree(props: list[Proposal]) -> bool:
    return len(set(group_proposals(props))) > 1
