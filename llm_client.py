# llm_client.py — single-summary output with safe fallbacks
from __future__ import annotations
import os, json
from typing import Dict, Optional

OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL") or None
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# set to "0" or "false" to skip JSON schema if your model/account is slow with it
USE_JSON_SCHEMA: bool = os.getenv("OPENAI_USE_JSON", "1") not in ("0", "false", "False")

_client = None

def _get_client():
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            return None
        from openai import OpenAI
        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _client

# ---- Only one field: summary_md ----
_SUMMARY_SCHEMA: Dict = {
    "type": "object",
    "properties": {
        "summary_md": {"type": "string", "minLength": 120}
    },
    "required": ["summary_md"],
    "additionalProperties": False
}

def _fallback_markdown(raw: str) -> str:
    # Local fallback: produce a readable multi-paragraph summary only.
    import textwrap
    raw = (raw or "").strip()
    if not raw:
        return "## Summary\nWe couldn’t retrieve the full article text. Try opening the source for details."
    paras = [p.strip() for p in raw.split("\n") if p.strip()]
    chunks, cur, total = [], [], 0
    for p in paras:
        cur.append(p); total += len(p)
        if total > 900:
            chunks.append(" ".join(cur)); cur = []; total = 0
    if cur: chunks.append(" ".join(cur))
    body = "\n\n".join(textwrap.fill(c, width=100) for c in chunks[:6])
    return f"{body}".strip()

def summarize_structured(system_prompt: str, user_prompt: str, content: str) -> str:
    client = _get_client()
    article = (content or "")[:18000]

    if client is None:
        return _fallback_markdown(article)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "ARTICLE CONTENT (verbatim, may be long):\n\n"
                f"{article}\n\n"
                "----\nNow follow the instructions below for the requested summary."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    # 1) Try JSON schema for a single field `summary_md`
    if USE_JSON_SCHEMA:
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.35,
                messages=messages,
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "ArticleSummary",
                                                 "schema": _SUMMARY_SCHEMA, "strict": True}},
                max_tokens=1400,
            )
            data = json.loads(resp.choices[0].message.content)
            return (data.get("summary_md") or "").strip() or _fallback_markdown(article)
        except Exception:
            pass  # fall through

    # 2) Plain-text completion (model returns markdown body directly)
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.35,
            messages=messages,
            max_tokens=1400,
        )
        md = (resp.choices[0].message.content or "").strip()
        return md or _fallback_markdown(article)
    except Exception:
        return _fallback_markdown(article)
