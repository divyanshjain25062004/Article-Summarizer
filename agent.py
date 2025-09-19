"""Agent orchestration for the article summarizer.

This module promotes the previous fixed pipeline into a small agentic
framework.  The core pieces are:

* ``ToolSpec`` – declarative wrappers that describe callable capabilities.
* ``AgentState`` – runtime scratchpad, memory, and timing budget.
* ``ReactivePlanner`` – a lightweight planner that decides the next tool to
  invoke based on the working memory.
* ``Agent`` – an executor that loops until the planner signals completion.

The concrete ``ArticleSummarizerAgent`` composes search, ranking, and
summarisation tools to deliver the same end-to-end behaviour as the original
``agent_run`` helper while leaving room for future planners, tools, or
multi-agent supervision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import concurrent.futures
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol

from llm_client import summarize_structured
from prompts import SUMM_SYSTEM, make_user_prompt
from tools import (
    extract_main,
    fetch_page,
    is_probably_article,
    multi_source_search,
    rank_candidates,
)

# ---------------------------------------------------------------------------
# Thread-pool utilities

_EXEC = concurrent.futures.ThreadPoolExecutor(max_workers=6)


def run_with_timeout(fn: Callable[..., Any], timeout: float, *args: Any, **kwargs: Any) -> Any:
    """Run ``fn`` in the shared executor and raise on timeout."""

    fut = _EXEC.submit(fn, *args, **kwargs)
    try:
        return fut.result(timeout=timeout)
    except concurrent.futures.TimeoutError as exc:  # pragma: no cover - exercised via runtime timeouts
        raise TimeoutError(f"{fn.__name__} timed out after {timeout}s") from exc


# ---------------------------------------------------------------------------
# Agent framework primitives


class Tool(Protocol):
    """Protocol for executable tools."""

    name: str
    description: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - protocol definition
        ...


@dataclass(slots=True)
class ToolSpec:
    """Declarative description of a tool that can be invoked by the agent."""

    name: str
    description: str
    handler: Callable[..., Any]
    default_inputs: Dict[str, Any] = field(default_factory=dict)

    def __call__(self, **kwargs: Any) -> Any:
        inputs = dict(self.default_inputs)
        inputs.update(kwargs)
        return self.handler(**inputs)


class ToolRegistry:
    """Simple registry that exposes tools by name."""

    def __init__(self, tools: Iterable[ToolSpec] | None = None) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:  # pragma: no cover - developer error guard
            raise ValueError(f"tool {name!r} is not registered") from exc

    def list(self) -> List[ToolSpec]:
        return list(self._tools.values())


@dataclass(slots=True)
class AgentAction:
    """Planner directive returned by a planner."""

    thought: str
    tool: Optional[str]
    inputs: Dict[str, Any]
    stop: bool = False
    final_data: Optional[Dict[str, Any]] = None

    @classmethod
    def finish(cls, thought: str, final_data: Optional[Dict[str, Any]] = None) -> "AgentAction":
        return cls(thought=thought, tool=None, inputs={}, stop=True, final_data=final_data)


@dataclass(slots=True)
class ToolCall:
    """Record of a single tool execution."""

    name: str
    inputs: Dict[str, Any]
    thought: str
    started_at: float
    duration_s: float
    result: Any = None
    error: Optional[str] = None


@dataclass
class AgentState:
    """Mutable state shared between planner iterations."""

    goal: str
    config: Dict[str, Any]
    deadline: Optional[float]
    memory: List[ToolCall] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def time_left(self) -> float:
        if self.deadline is None:
            return float("inf")
        return max(0.0, self.deadline - time.time())


class Planner(Protocol):
    """Planner decides the next action for the agent."""

    def decide(self, state: AgentState) -> AgentAction:  # pragma: no cover - implemented by concrete planner
        ...


class Agent:
    """Generic agent loop that executes planner-selected tools."""

    def __init__(
        self,
        tools: ToolRegistry,
        planner: Planner,
        reducers: Optional[Dict[str, Callable[[AgentState, AgentAction, Any, Optional[str]], None]]] = None,
        max_steps: int = 12,
    ) -> None:
        self._tools = tools
        self._planner = planner
        self._reducers = reducers or {}
        self._max_steps = max_steps

    def run(self, goal: str, config: Dict[str, Any]) -> AgentState:
        deadline = None
        if "budget_s" in config:
            deadline = time.time() + float(config["budget_s"])
        state = AgentState(goal=goal, config=config, deadline=deadline)

        for _ in range(self._max_steps):
            if state.time_left() <= 0.0:
                break

            action = self._planner.decide(state)
            if action.stop:
                if action.final_data:
                    state.context.setdefault("final", {}).update(action.final_data)
                break

            tool = self._tools.get(action.tool or "")
            started = time.time()
            error: Optional[str] = None
            result: Any = None
            try:
                result = tool(**action.inputs)
            except Exception as exc:  # pragma: no cover - runtime safety
                error = f"{type(exc).__name__}: {exc}"

            duration = time.time() - started
            state.memory.append(
                ToolCall(
                    name=tool.name,
                    inputs=action.inputs,
                    thought=action.thought,
                    started_at=started,
                    duration_s=duration,
                    result=result,
                    error=error,
                )
            )

            reducer = self._reducers.get(tool.name)
            if reducer:
                reducer(state, action, result, error)

        return state


# ---------------------------------------------------------------------------
# Concrete planner and tool reducers for the summariser


def _fallback_summary(title: str, snippet: str, url: str) -> Dict[str, str]:
    snippet_text = (snippet or "").strip()
    if snippet_text:
        body = f"{snippet_text}\n\nVisit the source link for the full context."
    else:
        body = (
            "We couldn’t reliably fetch the full article right now. Open the source "
            "to read the complete details."
        )
    return {"title": title or "Untitled", "url": url, "summary": body}


def _build_summary_input(article_text: str, snippet: str) -> tuple[str, bool]:
    primary = (article_text or "").strip()
    snippet_text = (snippet or "").strip()

    if primary and len(primary) >= 320:
        return primary, True

    pieces: List[str] = []
    if primary:
        pieces.append(primary)
    if snippet_text and snippet_text not in primary:
        pieces.append(snippet_text)

    combined = "\n\n".join(pieces).strip()
    return combined, len(combined) >= 220


def _summarise_candidate(
    query: str,
    candidate: Dict[str, Any],
    time_left: float,
    fetch_timeout: float,
    summarise_timeout: float,
) -> Dict[str, Any]:
    url = candidate["url"]
    title_guess = (candidate.get("title") or "Untitled").strip()
    snippet = (candidate.get("snippet") or "").strip()

    if not is_probably_article(url, title_guess):
        return {"skipped": True, "reason": "not_article"}

    if time_left < 3.0:
        return {"skipped": False, "article": _fallback_summary(title_guess, snippet, url), "used_fallback": True}

    try:
        html = run_with_timeout(fetch_page, min(fetch_timeout, time_left - 1e-3), url)
    except Exception as exc:
        return {
            "skipped": False,
            "article": _fallback_summary(title_guess, snippet, url),
            "used_fallback": True,
            "error": f"fetch: {type(exc).__name__}: {exc}",
        }

    try:
        title, text = extract_main(html, base_url=url)
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "skipped": False,
            "article": _fallback_summary(title_guess, snippet, url),
            "used_fallback": True,
            "error": f"extract: {type(exc).__name__}: {exc}",
        }

    combined_text, has_enough = _build_summary_input(text, snippet)
    if not has_enough or not combined_text:
        return {"skipped": False, "article": _fallback_summary(title or title_guess, snippet, url), "used_fallback": True}

    user_prompt = make_user_prompt(query, title or title_guess, url, mode="long")

    try:
        summary = run_with_timeout(
            summarize_structured,
            min(summarise_timeout, max(1.0, time_left - 1e-3)),
            SUMM_SYSTEM,
            user_prompt,
            combined_text,
        )
    except Exception as exc:
        return {
            "skipped": False,
            "article": _fallback_summary(title or title_guess, snippet, url),
            "used_fallback": True,
            "error": f"summarise: {type(exc).__name__}: {exc}",
        }

    return {
        "skipped": False,
        "article": {"title": title or title_guess, "url": url, "summary": summary},
        "used_fallback": False,
    }


class ReactivePlanner:
    """Rule-based planner that mirrors the traditional pipeline."""

    def __init__(self, top_k: int, max_candidates: int) -> None:
        self._top_k = top_k
        self._max_candidates = max_candidates

    def decide(self, state: AgentState) -> AgentAction:
        ctx = state.context

        if "results" not in ctx:
            return AgentAction(
                thought="Search multiple sources for candidate articles.",
                tool="search",
                inputs={
                    "query": state.goal,
                    "max_results_per_source": state.config.get("max_results_per_source", 30),
                    "domain_only": state.config.get("domain_only"),
                    "sources": state.config.get("sources"),
                },
            )

        if "ranked" not in ctx:
            return AgentAction(
                thought="Rank the gathered candidates to focus on the most relevant articles.",
                tool="rank",
                inputs={
                    "query": state.goal,
                    "items": ctx["results"],
                    "top_n": self._max_candidates,
                },
            )

        summaries: List[Dict[str, Any]] = ctx.setdefault("summaries", [])
        cursor: int = ctx.setdefault("cursor", 0)
        ranked: List[Dict[str, Any]] = ctx["ranked"]

        if len(summaries) >= self._top_k or cursor >= len(ranked):
            return AgentAction.finish(
                thought="Finished assembling summaries.",
                final_data={"articles": summaries[: self._top_k]},
            )

        candidate = ranked[cursor]
        return AgentAction(
            thought=f"Summarise candidate #{cursor + 1}.",
            tool="summarise",
            inputs={
                "query": state.goal,
                "candidate": candidate,
                "time_left": state.time_left(),
                "fetch_timeout": state.config.get("fetch_timeout", 10.0),
                "summarise_timeout": state.config.get("summarise_timeout", 18.0),
            },
        )


def _search_reducer(state: AgentState, action: AgentAction, result: Any, error: Optional[str]) -> None:
    if error:
        state.context["results"] = []
        state.context.setdefault("errors", []).append({"tool": "search", "message": error})
        return
    if not isinstance(result, list):  # pragma: no cover - defensive
        state.context["results"] = []
        state.context.setdefault("errors", []).append({"tool": "search", "message": "invalid result"})
        return
    state.context["results"] = result


def _rank_reducer(state: AgentState, action: AgentAction, result: Any, error: Optional[str]) -> None:
    if error:
        state.context.setdefault("errors", []).append({"tool": "rank", "message": error})
        state.context["ranked"] = state.context.get("results", [])[: action.inputs.get("top_n", 10)]
        return
    if not isinstance(result, list):  # pragma: no cover - defensive
        state.context.setdefault("errors", []).append({"tool": "rank", "message": "invalid result"})
        state.context["ranked"] = state.context.get("results", [])[: action.inputs.get("top_n", 10)]
        return
    state.context["ranked"] = result


def _summarise_reducer(state: AgentState, action: AgentAction, result: Any, error: Optional[str]) -> None:
    ctx = state.context
    summaries: List[Dict[str, Any]] = ctx.setdefault("summaries", [])
    ctx["cursor"] = ctx.get("cursor", 0) + 1

    if error:
        summaries.append(
            _fallback_summary(
                (action.inputs.get("candidate", {}).get("title") or "Untitled"),
                action.inputs.get("candidate", {}).get("snippet", ""),
                action.inputs.get("candidate", {}).get("url", ""),
            )
        )
        state.context.setdefault("errors", []).append({"tool": "summarise", "message": error})
        return

    if not isinstance(result, dict):  # pragma: no cover - defensive
        summaries.append(
            _fallback_summary(
                (action.inputs.get("candidate", {}).get("title") or "Untitled"),
                action.inputs.get("candidate", {}).get("snippet", ""),
                action.inputs.get("candidate", {}).get("url", ""),
            )
        )
        state.context.setdefault("errors", []).append({"tool": "summarise", "message": "invalid result"})
        return

    if result.get("skipped"):
        return

    article = result.get("article")
    if article:
        summaries.append(article)
    else:  # pragma: no cover - defensive
        summaries.append(
            _fallback_summary(
                (action.inputs.get("candidate", {}).get("title") or "Untitled"),
                action.inputs.get("candidate", {}).get("snippet", ""),
                action.inputs.get("candidate", {}).get("url", ""),
            )
        )


# ---------------------------------------------------------------------------
# Wiring it all together


SEARCH_TOOL = ToolSpec(
    name="search",
    description="Search the web, news, Reddit, and Twitter for relevant URLs.",
    handler=multi_source_search,
)

RANK_TOOL = ToolSpec(
    name="rank",
    description="Rank candidate articles by textual relevance and recency.",
    handler=lambda query, items, top_n: rank_candidates(query, items, top_n=top_n),
)

SUMMARISE_TOOL = ToolSpec(
    name="summarise",
    description="Fetch, extract, and summarise an article candidate.",
    handler=_summarise_candidate,
)


class ArticleSummarizerAgent(Agent):
    """Concrete agent wiring for the article summarisation workflow."""

    def __init__(self, top_k: int, max_candidates: int = 12) -> None:
        tools = ToolRegistry([SEARCH_TOOL, RANK_TOOL, SUMMARISE_TOOL])
        planner = ReactivePlanner(top_k=top_k, max_candidates=max_candidates)
        reducers = {
            "search": _search_reducer,
            "rank": _rank_reducer,
            "summarise": _summarise_reducer,
        }
        super().__init__(tools=tools, planner=planner, reducers=reducers, max_steps=32)


def agent_run(
    query: str,
    top_k: int = 3,
    domain_only: Optional[str] = None,
    sources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Entry point used by the FastAPI server and CLI."""

    budget_s = 32.0
    agent = ArticleSummarizerAgent(top_k=top_k, max_candidates=12)
    state = agent.run(
        goal=query,
        config={
            "budget_s": budget_s,
            "top_k": top_k,
            "domain_only": domain_only,
            "sources": sources,
            "max_results_per_source": 30,
            "fetch_timeout": 10.0,
            "summarise_timeout": 18.0,
        },
    )

    articles = list(state.context.get("summaries", []))
    ranked = state.context.get("ranked") or state.context.get("results", [])

    seen_urls = {art.get("url") for art in articles}
    for cand in ranked:
        if len(articles) >= top_k:
            break
        url = cand.get("url")
        if not url or url in seen_urls:
            continue
        articles.append(
            _fallback_summary(
                cand.get("title", "Untitled"),
                cand.get("snippet", ""),
                url,
            )
        )
        seen_urls.add(url)

    articles = articles[:top_k]

    Path("logs").mkdir(exist_ok=True)
    trace = {
        "query": query,
        "config": state.config,
        "context": state.context,
        "steps": [
            {
                "tool": call.name,
                "thought": call.thought,
                "inputs": call.inputs,
                "duration_s": round(call.duration_s, 3),
                "error": call.error,
            }
            for call in state.memory
        ],
    }
    trace_path = Path("logs/last_trace.json")
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

    return {"query": query, "articles": articles, "trace_path": str(trace_path)}


__all__ = [
    "Agent",
    "AgentAction",
    "AgentState",
    "ArticleSummarizerAgent",
    "ToolSpec",
    "ToolRegistry",
    "agent_run",
]

